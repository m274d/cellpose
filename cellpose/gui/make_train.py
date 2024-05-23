import sys, os, argparse, glob, pathlib, time
import numpy as np
from natsort import natsorted
from tqdm import tqdm
from cellpose import utils, models, io, core, version_str, transforms


def main():
    parser = argparse.ArgumentParser(description="cellpose parameters")

    input_img_args = parser.add_argument_group("input image arguments")
    input_img_args.add_argument("--dir", default=[], type=str,
                                help="folder containing data to run or train on.")
    input_img_args.add_argument(
        "--image_path", default=[], type=str, help=
        "if given and --dir not given, run on single image instead of folder (cannot train with this option)"
    )
    input_img_args.add_argument(
        "--look_one_level_down", action="store_true",
        help="run processing on all subdirectories of current folder")
    input_img_args.add_argument("--img_filter", default=[], type=str,
                                help="end string for images to run on")
    input_img_args.add_argument(
        "--channel_axis", default=None, type=int,
        help="axis of image which corresponds to image channels")
    input_img_args.add_argument("--z_axis", default=None, type=int,
                                help="axis of image which corresponds to Z dimension")
    input_img_args.add_argument("--t_axis", default=None, type=int,
                                help="axis of image which corresponds to T dimension")
    input_img_args.add_argument(
        "--chan", default=0, type=int, help=
        "channel to segment; 0: GRAY, 1: RED, 2: GREEN, 3: BLUE. Default: %(default)s")
    input_img_args.add_argument(
        "--chan2", default=0, type=int, help=
        "nuclear channel (if cyto, optional); 0: NONE, 1: RED, 2: GREEN, 3: BLUE. Default: %(default)s"
    )
    input_img_args.add_argument("--invert", action="store_true",
                                help="invert grayscale channel")
    input_img_args.add_argument(
        "--all_channels", action="store_true", help=
        "use all channels in image if using own model and images with special channels")
    input_img_args.add_argument("--anisotropy", required=False, default=1.0, type=float,
                                help="anisotropy of volume in 3D")

    training_args = parser.add_argument_group("training arguments")
    training_args.add_argument(
        "--mask_filter", default="_masks", type=str, help=
        "end string for masks to ignore in folder for creating slices. Default: %(default)s"
    )

    # crop settings
    slice_args = parser.add_argument_group("slice arguments")
    slice_args.add_argument(
        "--use_XZYZ", action="store_true", help=
        "create XZ and YZ slice samples"
    )
    slice_args.add_argument("--sharpen_radius", required=False, default=0.0,
                                type=float, help="tile normalization")
    slice_args.add_argument("--tile_norm", required=False, default=0.0, type=float,
                                help="tile normalization")
    slice_args.add_argument("--nimg_per_tif", required=False, default=30, type=int,
                                help="number of slices to save")
    slice_args.add_argument("--crop_size", required=False, default=256, type=int,
                                help="size of random crop to save")

    args = parser.parse_args()

    # find images
    if len(args.img_filter) > 0:
        imf = args.img_filter
    else:
        imf = None

    image_names = io.get_image_files(args.dir, args.mask_filter, imf=imf,
                                     look_one_level_down=args.look_one_level_down)

    channels = [args.chan, args.chan2]
    np.random.seed(0)
    
    os.makedirs(os.path.join(args.dir, "train/"), exist_ok=True)
    for name in image_names:
        name0 = os.path.splitext(os.path.split(name)[-1])[0]
        X = io.imread(name)
        X = transforms.convert_image(X, channels, channel_axis=args.channel_axis, 
                                       z_axis=args.z_axis, do_3D=True)
        Lz, Ly, Lx = X.shape[:-1]
        # number of slices per view
        nimg = (args.nimg_per_tif // 3) if args.use_XZYZ else args.nimg_per_tif

        if args.anisotropy != 1:
            rescaling = [[1, 1], [args.anisotropy, 1], [1 * args.anisotropy, 1]]
        else:
            rescaling = [1, 1, 1]
        
        # threshold to help find crops with ROIs
        ranges = np.ptp(X[...,0], axis=(-2,-1))
        range_threshold = np.percentile(ranges, 5)
            
        pm = [(0, 1, 2, 3), (1, 0, 2, 3), (2, 0, 1, 3)]
        ipm = [(3, 0, 1, 2), (3, 1, 0, 2), (3, 1, 2, 0)]
        
        sli = ["XY", "XZ", "YZ"]
        for p in range(3 if args.use_XZYZ else 1):
            xsl = X.copy().transpose(pm[p])
            ranges = np.ptp(xsl[...,0], axis=(-2,-1))
            xsl_valid = ranges > range_threshold
            xsl = xsl[xsl_valid]
            imgs = xsl[np.random.permutation(xsl.shape[0])[:nimg]]
            imgs = transforms.resize_image(imgs, rsz=rescaling[p])
            shape = imgs.shape
            print(shape)
            for k, img in enumerate(imgs):
                if args.tile_norm:
                    img = transforms.normalize99_tile(img, blocksize=args.tile_norm)
                if args.sharpen_radius:
                    img = transforms.smooth_sharpen_img(img,
                                                        sharpen_radius=args.sharpen_radius)
                take_crop = True
                while take_crop:
                    ly = np.random.randint(0, max(1, shape[1] - args.crop_size))
                    lx = np.random.randint(0, max(1, shape[2] - args.crop_size))
                    img0 = img[ly:ly + args.crop_size, lx:lx + args.crop_size]
                    if np.ptp(img0[:,:,0]) > range_threshold:
                        io.imsave(os.path.join(args.dir, "train", f"{sli[p]}_{name0}_{k}.tif"),
                                img0)
                        take_crop = False

if __name__ == "__main__":
    main()
