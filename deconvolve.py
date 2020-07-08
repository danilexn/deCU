#!/usr/bin python3

# Script by Daniel Leon-Perinan at CABD
# MIT License, 2020
# Version 1.0.5

from pycudadecon import RLContext, rl_decon, TemporaryOTF
from contextlib import contextmanager
from tqdm import tqdm
import pynvml as nvml
import numpy as np
import tifffile
import mrc
import os
import argparse
import sys
import time


class Image:
    def load_image(self, route):
        extension = os.path.splitext(route)[1]
        self.wavelengths, size, shape, image = opener[extension](route)
        try:
            self.nt, self.nz, self.nw, self.nx, self.ny = shape
        except (Exception):
            os.system(Exception)
            exit(1)
        self.image = image
        self.pxx, self.pxy, self.pxz = size

    def __init__(self, route):
        self.load_image(route)
        super().__init__()

    def frame(self, t, w):
        if self.nt <= 1 and self.nw > 1:
            im = self.image[w, 0 : self.nz, :, :]
        elif self.nw <= 1:  # Fixed for only 1 channel!
            if self.nt > 1:
                im = self.image[t, 0 : self.nz, :, :]
            else:
                im = self.image[0 : self.nz, :, :]
        else:
            if self.image.shape[1] == self.nz:
                im = self.image[t, 0 : self.nz, w, :, :]
            elif self.image.shape[1] == self.nw:
                im = self.image[t, w, 0 : self.nz, :, :]
        return im

    # Create a new function to duplicate the Z-stack and 
    # perform deconvolution. Then, remove the extra timeframe


def open_tif(route):
    if args.waves == None:
        raise ValueError("No wavelengths were provided. Please refer to --help")
    pxx, pxz = args.xyimage, args.zimage
    image = tifffile.imread(route)
    if len(image.shape) < 5:
        raise ValueError("Dimensions are wrong. You must provide a 4D stack")
    nt, nz, nw, nx, ny = image.shape
    return args.waves, [pxx, pxx, pxz], [nt, nz, nw, nx, ny], image


def open_dv(route):
    image = mrc.imread(route)
    header = image.Mrc.hdr
    nt, nw = header.NumTimes, header.NumWaves
    nx, ny, nsecs = header.Num
    nz = int(nsecs / nt / nw)
    pxx, pxy, pxz = header.d[0:3]
    return header.wave[0:4], [pxx, pxy, pxz], [nt, nz, nw, nx, ny], image


def join_spinning(route, channels):
    """
    This function allows joining a set of Spinning-Disk folder
    structure to a .tif file consisting of a Z-Stack.

    You must provide the route as a foldername, it is
    mandatory that inside the folder, channel names (subfolders)
    must be provided. This set of subfolders contains the Z-stacks
    in individual files for each timepoint. They will be joined in a
    two step process: 
        1. A stack for a single channel is reconstructed in memory
        2. The stack is merged into a single file
    This process takes place in RAM: please, make sure enough memory is 
    available!

    Args:
        route ([string]): folder name containing files to reconstruct a movie
        channels ([list]): list of channel names (same as subsequent folder names)

    Returns:
        filename [string]: The generated 4D stack file route
    """
    # Creates an empty list that will contain the final stack
    movieStack = []

    # Iterates over the user-provided list of channels
    for channel in channels:  # argument provided as spinning

        # Creates an empty list that will contain the channel stack
        channelStack = []

        # Route for the channel stack (individual timepoitns)
        channelPath = os.path.join(route, channel)

        # Individual files are parsed. Timepoint order in filename (..._t1, ..._tn)
        files = [
            os.path.join(channelPath, f)
            for f in os.listdir(channelPath)
            if os.path.isfile(os.path.join(channelPath, f))
        ]

        # Select timepoints based on *_{tN}.TIF notation
        frames = [
            int("".join(os.path.basename(os.path.splitext(f)[0]).split("_")[-1][1:]))
            for f in files
        ]
        # Sort the timepoints
        frames.sort()

        # Select a basename to iterate over timepoints
        filenameBase = "_".join(os.path.splitext(files[0])[0].split("_")[:-1])

        # Store the exact file extension
        extension = os.path.splitext(files[1])[1]

        # Iterating over files, construct the channelStack list
        for frame in frames:
            image = tifffile.imread("{}_t{}{}".format(filenameBase, frame, extension))
            channelStack.append(image)

        # Stores the channel inside the final movie
        movieStack.append(channelStack)

    # Transposes the array to maintain the classical TCZXY ordering (4D stack)
    movieStack = np.array(movieStack)
    movieStack = np.swapaxes(movieStack, 0, 1)
    movieStack = np.swapaxes(movieStack, 1, 2)

    # Stores the file with a new, derived name
    filename = "{}_stack.tif".format(route)
    tifffile.imsave("{}".format(filename), movieStack)

    # Returns the generated stack image fileroute
    return filename


# https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python/17954769#17954769
@contextmanager
def stdout_redirected(to=os.devnull):
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()
        os.dup2(to.fileno(), fd)
        sys.stdout = os.fdopen(fd, "w")

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "a") as file:
            _redirect_stdout(to=file)
        try:
            yield
        finally:
            _redirect_stdout(to=old_stdout)


def log(message, fname=None, show=True):
    if show:
        print("{}".format(message))
    if fname is not None:
        with open(fname, "a" if os.path.exists(fname) else "w") as f:
            f.write("{}\n".format(message))


def getGPUinfo():
    # This works with Nvidia GPUs, only. nvidia-smi must be installed
    smiOutput = os.popen("nvidia-smi").read()
    smiSplitted = smiOutput.splitlines()
    return "\n".join(smiSplitted[1:10])


def deconvolve_all():
    log("# Starting deconvolution with settings provided")
    # Implements CUDA and .dv (DeltaVision) file reading (thanks to tlambert01)
    filter_paths = args.psf

    log("# GPU information")
    log(getGPUinfo())

    for f in args.source:  # Must be fixed to implement SpinningDisk files
        logfile = "{}.log".format(f)  # Create .log file for each movie
        start = time.time()

        if os.path.isdir(f):
            channels = args.spinning
            log(
                "# Reconstructing and saving .tif file from folder structure with channels {}".format(
                    channels
                ),
                logfile,
            )
            f = join_spinning(f, channels)

        extension = os.path.splitext(f)[1]

        output_name = os.path.basename(os.path.splitext(f)[0])
        image_loaded = Image(f)
        nt, nz, nw = image_loaded.nt, image_loaded.nz, image_loaded.nw
        nx, ny = image_loaded.nx, image_loaded.ny

        nw_f = min(len(filter_paths), nw)

        log("\n\n# Processing file {}".format(f))
        log(
            "# Starting deconvolution at {}".format(
                time.strftime("%b %d %Y %H:%M:%S", time.localtime(start))
            )
        )

        log("####################################################", logfile, show=False)
        log("CUDA Deconvolution, by DLP (CABD), using pyCudaDecon", logfile, show=False)
        log("####################################################", logfile, show=False)
        log("\n# Deconvolution settings", logfile, show=False)
        log("## Projection {}".format(args.project), logfile, show=False)
        log("## Channels {}".format(nw), logfile, show=False)
        log("## File {}".format(f), logfile, show=False)
        log(
            "## XY pixel size {}".format(round(image_loaded.pxx, 3)),
            logfile,
            show=False,
        )
        log(
            "## Z-stack depth {}".format(round(image_loaded.pxz, 3)),
            logfile,
            show=False,
        )
        log("## GPU information", logfile, show=False)
        log(getGPUinfo(), logfile, show=False)
        log(
            "## Time start {}".format(
                time.strftime("%b %d %Y %H:%M:%S", time.localtime(start)),
            ),
            logfile,
            show=False,
        )
        log("\n# Deconvolution process steps", logfile, show=False)

        for w in range(0, nw_f):
            if filter_paths[w] == "_":
                continue
            log(
                "\n## Deconvolving channel {} with PSF {}. {} frame{}".format(
                    w, filter_paths[w], nt, "s" if nt > 1 else ""
                ),
                logfile,
            )
            image_cat = []
            with TemporaryOTF(filter_paths[w]) as otf:
                with stdout_redirected(to=logfile):
                    with RLContext([nz, nx, ny], otf.path) as ctx:
                        for t in tqdm(range(0, nt)):
                            os.system("echo ## Frame {}".format(t))
                            image = image_loaded.frame(t, w)
                            result = rl_decon(
                                image,
                                output_shape=ctx.out_shape,
                                dxdata=image_loaded.pxx,
                                dxpsf=args.xyfilter,
                                dzdata=image_loaded.pxz,
                                dzpsf=args.zfilter,
                                wavelength=image_loaded.wavelengths[w],
                                n_iters=args.i,
                                na=args.na,
                                nimm=args.refractive,
                            )
                            result = result.astype(np.uint16)
                            image_cat.append(result)
                        image_result = np.array(image_cat)
            fname_save = ("{0}_{1}_D3D{2}").format(
                output_name, str(image_loaded.wavelengths[w]), extension
            )
            log("## Saving file as {}".format(fname_save), logfile)
            saveformat[extension](
                fname_save, image_result,
            )
            # Added new functionality for Z-stack maximum projection
            if args.project:
                log("## Creating maximum projection", logfile)
                min_plane = args.planes[0]
                max_plane = args.planes[1]
                if image_loaded.nt <= 1:
                    if len(image_result.shape) == 4:
                        image_max = np.max(
                            image_result[0, min_plane:max_plane, :, :], axis=0
                        )
                    else:
                        image_max = np.max(
                            image_result[min_plane:max_plane, :, :], axis=0
                        )
                else:
                    image_max = np.max(
                        image_result[:, min_plane:max_plane, :, :], axis=1
                    )
                fmax_save = ("{0}_{1}_PRJ_D3D{2}").format(
                    output_name, str(image_loaded.wavelengths[w]), extension
                )
                log("## Saving maximum projection as {}".format(fmax_save), logfile)
                saveformat[extension](
                    fmax_save, image_max,
                )
        end = time.time()
        elapsed = end - start
        log(
            "# Deconvolution ended at {}, elapsed {}".format(
                time.strftime("%b %d %Y %H:%M:%S", time.localtime(end)),
                "{} m : {} s".format(int(elapsed / 60), int(elapsed % 60)),
            ),
            logfile,
        )


def cmdline_args():
    p = argparse.ArgumentParser(
        description="""
        CUDA deconvolution of (multi)frame .dv or .tif(f) files
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--source",
        nargs="+",
        required=True,
        help="Source file(s) to deconvolve (or folder if SpinningDisk)",
    )
    p.add_argument(
        "--psf",
        nargs="+",
        required=True,
        help="PSF file(s) (one per wavelength). Provide _ for channel negative selection",
    )
    p.add_argument(
        "--zfilter",
        nargs=1,
        default=0.1,
        help="Z-depth of the given filter(s) (PSF) in microns",
    )
    p.add_argument(
        "--xyfilter",
        nargs=1,
        default=0.1,
        help="XY-pixel size of the given filter(s) (PSF) in microns",
    )
    p.add_argument(
        "--zimage",
        nargs=1,
        default=0.1,
        type=float,
        help="Z-depth of the given image stack in microns",
    )
    p.add_argument(
        "--xyimage",
        nargs=1,
        default=0.1,
        type=float,
        help="XY-pixel size of the given image in microns",
    )
    p.add_argument(
        "--na",
        nargs=1,
        default=1.4,
        help="Numerical aperture units for selected objective",
    )
    p.add_argument(
        "--refractive", nargs=1, default=1.5, help="Refractive index of interface media"
    )
    p.add_argument(
        "--spinning",
        nargs="+",
        type=str,
        help="Folder names for each channel in a SpinningDisk type file structure",
    )
    p.add_argument(
        "-i", nargs=1, default=15, help="Iterations of the Richardson-Lucy algorithm"
    )
    p.add_argument(
        "-p", "--project", action="store_true", help="Project the z-stack in a new file"
    )
    p.add_argument(
        "--planes",
        type=int,
        nargs=2,
        default=[0, -1],
        help="Select range for z-stack projection, two numbers, space delimited",
    )
    p.add_argument("-w", "--waves", nargs="+", help="Select wavelengths to process")

    return p.parse_args()

# Dictionary for opening format functions
opener = {".tif": open_tif, ".tiff": open_tif, ".dv": open_dv}

# Dictionary for saving functions
saveformat = {
    ".tif": tifffile.imsave,
    ".TIF": tifffile.imsave,
    ".tiff": tifffile.imsave,
    ".TIFF": tifffile.imsave,
    ".dv": mrc.imsave,
    ".DV": mrc.imsave,
}


if __name__ == "__main__":
    args = cmdline_args()
    deconvolve_all()
