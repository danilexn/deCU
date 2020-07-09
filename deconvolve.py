#!/usr/bin python3

# Script by Daniel Leon-Perinan at CABD
# MIT License, 2020
# Version 1.0.5

from pycudadecon import RLContext, rl_decon, TemporaryOTF
from contextlib import contextmanager
from tqdm import tqdm
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
            if len(self.image.shape) == 5:
                im = self.image[t, 0 : self.nz, w, :, :]
            elif self.nt > 1:
                im = self.image[t, 0 : self.nz, :, :]
            else:
                im = self.image[0 : self.nz, :, :]
                # More than one z is mandatory!
        else:
            if self.image.shape[1] == self.nz:
                im = self.image[t, 0 : self.nz, w, :, :]
            elif self.image.shape[1] == self.nw:
                im = self.image[t, w, 0 : self.nz, :, :]
        return im

    # TODO implement a duplication function


def open_tif(route):
    """[summary]

    Args:
        route (string): route for the .tiff file being processed

    Raises:
        ValueError: when wavelengths (even if one) are not provided
        ValueError: when a 4D stack is not provided

    Returns:
        tuple: complying the order in the Image class, returns the tuple
        of dimensions (TZWXY), as well as pixel size and other features
    """
    # Check that wavelengths have been provided through arguments
    if args.waves == None:
        raise ValueError("No wavelengths were provided. Please refer to --help")

    # Get pixel and stack size from arguments (defaults are set if none)
    pxx, pxz = args.xyimage, args.zimage

    # Read the image using tifffile
    image = tifffile.imread(route)

    # Check that the image is a 4D stack
    if len(image.shape) < 5:
        raise ValueError("Dimensions are wrong. You must provide a 4D stack")

    # Detuple the image properties
    nt, nz, nw, nx, ny = image.shape

    return args.waves, [pxx, pxx, pxz], [nt, nz, nw, nx, ny], image


def open_dv(route):
    """Opens a DeltaVision .dv image stack, and generates a tuple of 
    settings using mrc

    Args:
        route (string): route for the .dv file being processed

    Returns:
        tuple: complying the order in the Image class, returns the tuple
        of dimensions (TZWXY), as well as pixel size and other features
    """
    # Load the .dv file into memory with mrc
    image = mrc.imread(route)

    # Parse the header
    header = image.Mrc.hdr

    # Parse the number of timepoints and wavelengths
    nt, nw = header.NumTimes, header.NumWaves

    # Parse the number of XY pixels (image resolution)
    nx, ny, nsecs = header.Num

    # Parse the number of z planes
    nz = int(nsecs / nt / nw)

    # Parse the pixel and stack sizes in microns
    pxx, pxy, pxz = header.d[0:3]
    return header.wave[0:4], [pxx, pxy, pxz], [nt, nz, nw, nx, ny], image


def join_spinning(route, channels):
    """Joins a set of Spinning-Disk folder
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
        [string]: The generated 4D stack file route
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
    # TODO: this is a bit wanky, will be fixed
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
    """Redirects the CUDAdecon output to the provided route, avoiding displaying it
    in the display (iterations, lambda values...). Instead, it is better to store it
    in a log.

    Args:
        to (string, optional): text file to write into. Defaults to os.devnull.
    """
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()
        os.dup2(to.fileno(), fd)
        sys.stdout = os.fdopen(fd, "w")

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        # Changed to append
        with open(to, "a") as file:
            _redirect_stdout(to=file)
        try:
            yield
        finally:
            _redirect_stdout(to=old_stdout)


def log(message, fname=None, show=True):
    """Logging function used throughout the program

    Args:
        message (string): what is to be logged
        fname (string, optional): route of the log file, if not provided, no file will be written. Defaults to None.
        show (bool, optional): to be displayed onscreen. Defaults to True.
    """

    # Check function arguments
    if show:
        print("{}".format(message))
    if fname is not None:
        # Append the message content into the file if it exists, or create it from scratch
        with open(fname, "a" if os.path.exists(fname) else "w") as f:
            f.write("{}\n".format(message))


def getGPUinfo():
    """Gets GPU information with logging purposes. Please note that
    CUDA library must be installed, with nvidia-smi available.

    Returns:
        stirng: formatted output of the nvidia-smi command
    """    
    # Call a new nvidia-smi process and read its output
    smiOutput = os.popen("nvidia-smi").read()

    # Clean output contents
    smiSplitted = smiOutput.splitlines()

    return "\n".join(smiSplitted[1:10])

def max_projection(im, minimum=0, maximum=-1):
    """Returns the Z maximum projection of an image, between the
    provided minimum and maximum slices

    Args:
        im (np.array): image data as numpy array, any format. 4D stack!
        minimum (int, optional): minimum slice in Z projection. Defaults to 0.
        maximum (int, optional): maximum slice in Z projection. Defaults to -1.

    Returns:
        np.array: z maximum projected image
    """    
    if im.shape == 5 and np.size(im, axis = 0) <= 1:
        if len(im.shape) == 4:
            result = np.max(
                im[0, minimum:maximum, :, :], axis=0
            )
        else:
            result = np.max(
                im[minimum:maximum, :, :], axis=0
            )
    else:
        result = np.max(
            im[:, minimum:maximum, :, :], axis=1
        )
    
    return result


def deconvolve_all():
    """Main deconvolution function, where all logging, file exploration and
    processing is wrapped. Continuous references to external libraries...
    Many thanks to tlambert01 for the libs mrc and pyCUDAdecon
    """    

    log("# Starting deconvolution")
    log("# GPU information")
    log(getGPUinfo())

    # Get the paths for each filter to be used, in the same order
    filter_paths = args.psf

    # Iterate over the list of files provided
    # Note that queuing can be done using a simple .sh script to call the
    # application with different parameters
    for f in args.source: 

        # Log file naming
        logfile = "{}.log".format(f)

        # Start a timer for profiling time elapsed
        start = time.time()

        # Check whether source provided is SpinningDisk-like
        if os.path.isdir(f):
            channels = args.spinning
            log(
                "# Reconstructing and saving .tif file from folder structure with channels {}".format(
                    channels
                ),
                logfile,
            )
            f = join_spinning(f, channels)

        # Store the file extension to save in the same format
        extension = os.path.splitext(f)[1]

        # Generate an output name based on the source name
        outname = os.path.basename(os.path.splitext(f)[0])

        # Load the image
        log("# Loading the file {} and creating log".format(f), logfile)
        im = Image(f)

        # Calculate an appropriate number of wavelengths to process
        nw = min(len(filter_paths), im.nw)
 
        # Create the header for the current file
        log("\n\n# Processing file {}".format(f))
        log(
            "# Starting deconvolution at {}".format(
                time.strftime("%b %d %Y %H:%M:%S", time.localtime(start))
            )
        )
        log("####################################################", logfile, show=False)
        log("deCU v1.0.5, by DLP (CABD), 2020", logfile, show=False)
        log("####################################################", logfile, show=False)
        log("\n# Deconvolution settings", logfile, show=False)
        log("## Projection {}".format(args.project), logfile, show=False)
        log("## Channels {}".format(im.nw), logfile, show=False)
        log("## File {}".format(f), logfile, show=False)
        log(
            "## XY pixel size {}".format(round(im.pxx, 3)),
            logfile,
            show=False,
        )
        log(
            "## Z-stack depth {}".format(round(im.pxz, 3)),
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
        
        # Iterate over the range of wavelengths
        for w in range(0, nw):
            # Skip wavelengths marked as null
            if filter_paths[w] == "_":
                continue

            log(
                "\n## Deconvolving channel {} with PSF {}. {} frame{}".format(
                    w, filter_paths[w], im.nt, "s" if im.nt > 1 else ""
                ),
                logfile,
            )

            # Provisional list to store multiframe images
            stack = []

            # Loading all libraries for the deconvolution process
            with TemporaryOTF(filter_paths[w]) as otf:
                with stdout_redirected(to=logfile):
                    with RLContext([im.nz, im.nx, im.ny], otf.path) as ctx:
                        # Iterate over all frames in the image
                        for t in tqdm(range(0, im.nt)):

                            # Using echo to save into the logfile the frame no.
                            os.system("echo ## Frame {}".format(t))

                            # Retrieve current frame
                            image = im.frame(t, w)

                            # Perform pyCUDAdecon deconvolution
                            result = rl_decon(
                                image,
                                output_shape=ctx.out_shape,
                                dxdata=im.pxx,
                                dxpsf=args.xyfilter,
                                dzdata=im.pxz,
                                dzpsf=args.zfilter,
                                wavelength=im.wavelengths[w],
                                n_iters=args.i,
                                na=args.na,
                                nimm=args.refractive,
                            )

                            # Convert the result to uint16 to preserve format
                            result = result.astype(np.uint16)

                            # Append the timepoint to the stack
                            stack.append(result)

                        # Generate a np array from the stack
                        imResult = np.array(stack)

            # Generate a definitive saving name
            savename = ("{0}_{1}_D3D{2}").format(
                outname, str(im.wavelengths[w]), extension
            )

            log("## Saving file as {}".format(savename), logfile)
            saveformat[extension](
                savename, imResult,
            )

            # Check whether max-projection has to occur
            # TODO implement more types of projection
            if args.project:
                log("## Creating maximum projection", logfile)
                imProjected = max_projection(imResult, minimum = args.planes[0], maximum = args.planes[1])
                savename = ("{0}_{1}_PRJ_D3D{2}").format(
                    outname, str(im.wavelengths[w]), extension
                )
                log("## Saving maximum projection as {}".format(savename), logfile)
                saveformat[extension](
                    savename, imProjected,
                )
        
        # Stop profiling timer and calculate time elapsed
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
    """Parses the command line arguments as program parameters

    Returns:
        argparse.args: comprehensive list of arguments
    """    
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


# Dictionary for opening functions
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
