#!/usr/bin/env python
# -*- coding: utf-8 -*-

## ./main.py -i ../data/pics_kolonie_4/ -p

# ------------------------------------------------------------------------------
#                                  MODULES IMPORT
# ------------------------------------------------------------------------------

# os, sys, benchmark and args handling
import os, sys, time, getopt

# numerical computing and pandas
import numpy as np
from numpy import *

# OpenCV
import cv2

from itertools import izip_longest

# SciKit
import skimage
from skimage import transform as tf
from skimage.segmentation import clear_border

import warnings

import math, operator
from PIL import Image
from PIL import ImageChops

from functions import *


# ------------------------------------------------------------------------------
#                                      MAIN
# ------------------------------------------------------------------------------
def main(argv):
    start_time = time.time() # use timeit for small snippets of code
    #    pm = __import__("functions")
    #    print(dir(pm))
    try:
        opts, args = getopt.getopt(argv, "hi:o:p", ["input=", "param=", "help", "output"])
        if not opts:
            usage()
            print "\t[Error]\tPlease supply a directory with \n"
            sys.exit()
    except getopt.GetoptError, err:
        print str(err)
        os.system('cls' if os.name == 'nt' else 'clear')
        usage()
        print "\t[Error]\tPlease supply a folder\n"
        sys.exit()
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-i", "--input"):
            if os.path.isdir(arg):
                idir = arg
                print "\t[Input] \t Directory supplied\t", idir
        elif opt == '-p':
            image_treatment = True
            header()
            print "\t[Input]\t Image filetype selected"

            # ------------------------------------------------------------------
            #                             LOAD DATA
            # ------------------------------------------------------------------

            # Put filenames in a list (luminance, red, green, blue)
            flist, f_red, f_green, f_blue = get_imlist(idir, 1)

            ## ADD CONTROL IF supplied file doesn't exist in get_imlist()

            print "\t> %d files found\t" %len(flist)
            #flist = get_imlist("../data/pics_kolonie_4_3",1)

            # Load luminance images
            c_imgs, size_var = load_img(flist)
            print "\t> N frames imported\t", len(c_imgs)

            # 1st frame properties (rows, cols)/(height, width)
            rows, cols, channels = size_var.shape
            print "\t Images I of size ", size_var.shape

            elapsed_time(start_time)

            # Load RGB channels
            print "\t Loading all RGB Channels .."
            red_imgs, sv = load_img(f_red)
            green_imgs, sv = load_img(f_green)
            blue_imgs, sv = load_img(f_blue)

            elapsed_time(start_time)


            # Silent mode : warnings masked from this point on
            warnings.filterwarnings("ignore")

            # ------------------------------------------------------------------
            #                           PREPROCESSING
            # ------------------------------------------------------------------

            gray, denoise, clahe, ada, dilate, bin = preprocess(flist, c_imgs)
            elapsed_time(start_time)


            # ------------------------------------------------------------------
            #                           REGISTRATION
            # ------------------------------------------------------------------

            # Pre-alignment Average RMS of bin imgs.
            mrms = rms_call(bin, flist)
            print "\t> [Pre-Alignment]\tAverage RMS of consecutive frames\t",mrms
            print "\t> Image closeness\t%s" %(100-mrms*100/rows)

            # Contour based approach wrapped in a function
            anchors1, cnts, fltr_cnts, bad_cnts, areas, perim, fltr_areas, ratio, dist =\
                contour_approach(flist, bin, rows, cols)

            # Image alignment
            print "\tAligning %d frames of all channels.." %len(c_imgs)*4
            align_c_imgs = register(anchors1, c_imgs, flist, rows, cols)
            red = register(anchors1, red_imgs, f_red, rows, cols)
            green = register(anchors1, green_imgs, f_green, rows, cols)
            blue = register(anchors1, blue_imgs, f_blue, rows, cols)

            # Preprocessing on newly aligned imgs.
            gray, denoise, clahe, ada, dilate, bin = preprocess(flist,\
                                                                align_c_imgs)

            # Re used post alignment for masking bad cnts
            anchors2, cnts, fltr_cnts, bad_cnts, areas, perim, fltr_areas, ratio, dist =\
                contour_approach(flist, bin, rows, cols)

            # Mask bad contours for an appropriate evaluation of performance
            mask = np.ones((rows, cols), dtype="uint8") * 255
            cbin = remove_bad_cnts(bin, bad_cnts, rows, cols)

            # Bitwise comparison to prepare for masking of contours
            #histb = layer(flist, bin, gray)


            # ------------------------------------------------------------------
            #                      PERFORMANCE EVALUATION
            # ------------------------------------------------------------------

            # Post-alignment Average RMS of bin imgs.
            mrms = rms_call(cbin, flist)
            print "\t> [Post-Alignment]\tAverage RMS of consecutive frames\t",mrms
            print "\t> Image closeness\t%s" %(100-mrms*100/rows)

            elapsed_time(start_time)
            

            # ------------------------------------------------------------------
            #                         HANDLE OUTPUT
            # ------------------------------------------------------------------


            # Exporting images
            # Define arrays to output str. folder name and each of len(flist)
            out = [align_c_imgs, gray, denoise, clahe, ada, dilate, cbin]
            dirs = ["1_align", "2_gray", "3_denoise", "4_clahe", "5_ada", \
                    "6_dilate", "7_bin"]
            export(flist, dirs, out)

        elif opt== '-v':
            print "\t[Input]\t Video filetype selected"
            # Put filenames in a list
            flist = get_imlist(idir,c)
            print "\t[Error]\tVideo handling not yet implemented"
        elif opt in ('-o', '--output'):
            outfile = arg
        else:
            usage()
            assert False, "[Error]\tUnhandled option\n"

    print '\tProcessed options [{0}] in Directory [{1}]\n'.format(','.join(argv),idir)
    elapsed_time(start_time)



# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main(sys.argv[1:])

