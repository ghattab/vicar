
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


# ------------------------------------------------------------------------------
#                                   USAGE
# ------------------------------------------------------------------------------

def usage():
    """ Returns usage information """
    os.system('cls' if os.name == 'nt' else 'clear')
    usage = """\ns
        -h --help                 Prints this
        -i --input                Supplies a directory containing the data
        -p --param                Flag for treating images
        """
    print usage


def header():
    """ Returns usage information """
    os.system('cls' if os.name == 'nt' else 'clear')
    usage = """\n
        ------------------------------------------------------------

        VICAR, (VI)sual (C)ues (A)daptive (R)egistration - Georges Hattab
        Creation date\t141117
        Last update\t160229
        MIT License (MIT)

        ------------------------------------------------------------\n
        """
    print usage

# ------------------------------------------------------------------------------
#                                   OTHER
# ------------------------------------------------------------------------------

def elapsed_time(start_time):
    print("\t--- %4s seconds ---\n" %(time.time()-start_time))


def is_none(object):
    """ This function checks if an object is Nonetype or has no length """
    if (object is None) or (len(object) == 0):
        return True
    else:
        return False


def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)


def create_dir(dir):
    """ Directory creation """
    if not os.path.exists(dir):
        os.makedirs(dir)
    print "\t> Directory '%s' created" %(str(dir))


# ------------------------------------------------------------------------------
#                                   INPUT
# ------------------------------------------------------------------------------

def get_imlist(path, choice):
    """ Returns a list of filenames for all compatible extensions in a directory
	    Args : path of the directory and the supplied user choice
        Handles a directory containing i_ext as extension and 
        c1, c2, c3, c4 as luminance, red, green, blue channels respectively
		Returns the list of files to be treated"""
    i_ext = tuple([".tif",".jpg",".jpeg",".png"])
    v_ext = tuple([".3g2", ".3gp", ".asf", ".asx", ".avi", ".flv", ".m2ts", \
				   ".mkv", ".mov", ".mp4", ".mpg", ".mpeg", ".rm", ".swf", \
				   ".vob", ".wmv"])
    print '\t\t', path
    if(choice == 1):
        print "\t[Input]\t Image filetype selected"
        f = [os.path.join(path,f) for f in os.listdir(path) \
             if f.lower().endswith(i_ext)]
        ext = str.split(f[0],".")[-1] # recovered extension
        #(check if all filenames have the same extension)
        # recover all dimensions
        f_lum = [ l for l in f if l.endswith("c1"+"."+ext) ]
        f_red = [ r for r in f if r.endswith("c2"+"."+ext) ]
        f_green = [ g for g in f if g.endswith("c3"+"."+ext) ]
        f_blue = [ b for b in f if b.endswith("c4"+"."+ext) ]
        return f_lum, f_red, f_green, f_blue
    elif(choice == 2):
       print '[Input]\t Video filetype selected'
    else:
        print "\t\n[Error]\t Please try again."
# >>>>>>>>>> TO DO # Convert to frames and verify
    return [os.path.join(path,f) for f in os.listdir(path)\
            if f.lower().endswith(v_ext)]


def load_img(flist):
    """ Loads images in a list of arrays
	    Args : list of files
	    Returns list of all the ndimage arrays """
    rgb_imgs = []
    for i in flist:
        rgb_imgs.append(cv2.imread(i, -1)) # flag <0 to return img as is
    print "\t> Batch import of N frames\t", len(rgb_imgs)
    size_var = cv2.imread(i)    # (height, width, channels)
    return rgb_imgs, size_var

# ------------------------------------------------------------------------------
#                                   TRANSFORMS
# ------------------------------------------------------------------------------

def rgb_to_gray(img):
    """ Converts an RGB image to greyscale, where each pixel
        now represents the intensity of the original image.
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def invert(img):
    cimg = copy(img)
    return cv2.bitwise_not(cimg, cimg)


def ubyte(img):
    return cv2.convertScaleAbs(img, alpha=(255.0/65535.0))


# ------------------------------------------------------------------------------
#                               PREPROCESSING
# ------------------------------------------------------------------------------

def preprocess(flist, imgs):
    """ Preprocesses RGB imgs
            Args :  flist, list of strings' imgs filenames
                    imgs, loaded list of imgs arrays
            Returns 6 lists of img arrays, each corresponding to a preprocess. step
        """
    gclahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))
    gray, denoise, clahe, ada, dilate, clear, bin = [], [], [], [], [], [], []
    kernel = np.ones((3,3), np.uint8)

    # mask creation using a circle drawn from the center of each frame
    mask = np.ones(imgs[0].shape[:2], dtype="uint8") * 255
    rows, cols = imgs[0].shape[:2]
    cv2.circle(mask, (rows/2,cols/2), rows*1/5*3, (0,0,0), -1)
    circle = 255 - mask

    # Apply for each each image file in flist
    for i in range(len(flist)):
        # RGB (uint16 3C) to GRAY (uint16)
        gray_img = rgb_to_gray(imgs[i])
        gray.append(gray_img)


        # Denoise Bilateral (takes 32F or uint8)
        gray_ubyte = ubyte(gray_img)
        denoise_img = cv2.bilateralFilter(gray_ubyte, 10, 75, 75)
        denoise.append(denoise_img)

        # CLAHE (uint8 -> uint8)
        clahe_img = gclahe.apply(denoise_img)
        clahe.append(clahe_img)

        # Adaptive Thresholding (uint8)
        ada_img = cv2.adaptiveThreshold(clahe_img, 255, \
                                        cv2.ADAPTIVE_THRESH_MEAN_C, \
                                        cv2.THRESH_BINARY, 11, 2)
        ada.append(ada_img)

        # Dilation (uint8 -> uint8)
        dilation = cv2.dilate(ada_img, kernel, iterations = 1)
        dilate.append(dilation)

        # Clear borders
        clear_img = clear_border(dilation)
        clear.append(clear_img)

        # Mask with circle
        bin_img = cv2.bitwise_or(clear_img, clear_img, mask=circle)
        bin.append(bin_img)

    return gray, denoise, clahe, ada, dilate, bin



# ------------------------------------------------------------------------------
#                               CONTOURS
# ------------------------------------------------------------------------------

def contour_approach(flist, bin, rows, cols):
    # Detection of visual cues by Border following [Suzuki85] algorithm
    # --- Suzuki, S. and Abe, K., Topological Structural Analysis
    # --- of Digitized Binary Images by Border Following.
    # --- CVGIP 30 1, pp 32-46 (1985)
    c = find_contours(flist, bin)

    # Remove all contours of length 1
    for i in range(len(c)):
        c[i] = [x for x in c[i] if len(x)!= 1]

    contours = c

    # Perimeter-area ratio based filtering
    areas = find_area(flist, contours)
    perim = find_perimeter(flist, contours)
    ratio = find_ratio(perim, areas)
    dist = find_euclidean(ratio)
    #
    n, fltr_cnts, bad_cnts = filter_contours(flist, contours, dist)
    fltr_areas = find_area(flist, fltr_cnts)

    # Find anchors and align images
    anchors = find_anchors(flist, n, fltr_cnts)
    return anchors, contours, fltr_cnts, bad_cnts, areas, perim, fltr_areas, ratio, dist


def get_contours(image):
    """ Finds the outer contours of a binary image and returns 
        a shape-approximation of them. 
        Since we are only looking for outer contours, no object hierarchy exists
    """
    (contours, hierarchy) = cv2.findContours(image, \
                                             mode=cv2.cv.CV_RETR_TREE, \
                                             method=cv2.cv.CV_CHAIN_APPROX_SIMPLE)
    return contours


def find_contours(flist, images):
    """ for each frame
    """
    cnts, len_cnts = [], []
    for i in range(len(flist)):
        # find contours
        contours = get_contours(images[i])
        cnts.append(contours)
        # number of contours found
        len_cnts = len(contours)
        print (len_cnts)
    return cnts


def get_centers(contours):
    """ Finds the moments or the centroids of a list of contours returned by
        the find_contours (or cv2.findContours) function.
        If any moment of the contour is 0, the centroid is not computed. 
        Therefore the number of centroids returned by this function 
        may be smaller than the number of contours passed in.

        The return value from this function is a list of (x,y) pairs, 
        where each (x, y) pair denotes the center of a contour.
    """
    centers = []
    for contour in contours:
        moments = cv2.moments(contour, True)
        # If any moment is 0, discard the entire contour. This is
        # to prevent division by zero.
        if (len(filter(lambda x: x==0, moments.values())) > 0):
            continue
        center = (moments['m10']/moments['m00'] , moments['m01']/moments['m00'])
        # Convert floating point contour center into an integer so that
        # we can display it later.
        center = map(lambda x: int(round(x)), center)
        centers.append(center)
    return centers


def find_centers(flist, contours):
    """ for each frame
    """
    centers = []
    for i in range(len(flist)):
        cc = get_centers(contours[i])
        centers.append(cc)
    return centers


def get_area(contours):
    """ Computes the area of a contour using the Green Theorem
        contours = contours[i]
    """
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    return area


def find_area(flist, contours):
    """ for n frames
    """
    n_area = []
    for i in range(len(flist)):
        n_area.append(get_area(contours[i]))
    return n_area


def get_perimeter(contours):
    """ Computes the perimeter of a contours using arclength
        contours = contours[i]
        """
    perimeter = []
    for i in range(len(contours)):
        perimeter.append(cv2.arcLength(contours[i], True))
    return perimeter


def find_perimeter(flist, contours):
    """ for n frames
        """
    n_perimeter = []
    for i in range(len(flist)):
        n_perimeter.append(get_perimeter(contours[i]))
    return n_perimeter


def find_ratio(perim, areas):
    ratio = []
    for i in range(len(perim)):
        ratio.append([math.ceil(x*100)/100 \
                      for x in list(np.array(perim[i]) / np.array(areas[i]))])
    return ratio


def check_equal(list):
    """ Determines whether two lists are equal
    """
    return list[1:] == list[:-1]


def get_euclidean(a, b):
    dist = np.linalg.norm(a-b)
    return dist


def find_euclidean(ratio):
    """ Find the euclidean distance between the min(ratio) and each list elmnt
        """
    dist = []
    for i in range(len(ratio)):
        dist.append([ get_euclidean( min(ratio[i]), ratio[i][j] ) \
                     for j in range( len(ratio[i]) ) ])
    return dist


def filter_contours(flist, contours, dist):
    """ Filter out all visual cues with a eucl_dist(P/A ratio) smaller than .05
    """
    n, fltr_cnts, bad_cnts = [], [], []
    # loop over each frame to verify condition
    for i in range(len(flist)):
        fltr_areas = list(np.array(contours[i])[np.where(np.array(dist[i]) < .05)])
        bad_areas = list(np.array(contours[i])[np.where(np.array(dist[i]) > .05)])
        fltr_cnts.append(fltr_areas)
        bad_cnts.append(bad_areas)
        n.append(len(fltr_areas))
    return n, fltr_cnts, bad_cnts


def remove_bad_cnts(bin, bad_cnts, rows, cols):
    """ Remove bad contours by drawing it on the mask
        Args :  bin, binary images resulting from preprocessing
                bad_cnts, contours filtered out as bad
                mask, initialised before function call
        Returns a list of images deprived of bad contours
    """
    cbin = []
    for i in range(len(bad_cnts)):
        # draw a bad contour on the mask
        mask = np.ones((rows, cols), dtype="uint8") * 255
        for c in bad_cnts[i]:
            cv2.drawContours(mask, [c], -1, 0, -1)
        tmp = cv2.bitwise_and(255*bin[i], 255*bin[i], mask=mask)
        cbin.append(tmp)
    return cbin


# ------------------------------------------------------------------------------
#                               REGISTRATION
# ------------------------------------------------------------------------------

def pull_anchors(flist, n, fltr_contours):
    """ Returns 3 anchor points for frames having equal n contours
    """
    # amount of contours found in frame 0
    if n == 1:
        ctrl_pts = []
        # bounding rectangle to get 4 coords.
        for i in range(len(flist)):
            rect = cv2.minAreaRect(fltr_contours[i])
            box = cv2.cv.BoxPoints(rect)
            # rect = ( center (x,y), (width, height), angle of rotation )
            # x1, y1 ; x1 + width, y1 + height
            x1, y1 = box[0]
            x2, y2 = box[1]
            x3, y3 = box[2]
            pts = np.array((x1, y1), \
                           (x2, y2), \
                           (x3, y3))
            # -- get data array sorted by n-th column (n= 1)
            pts = pts[np.argsort(pts[:,1])]
            ctrl_pts.append(pts)
        return ctrl_pts
    # 2 cnts
    elif n == 2:
        # initial random indexes to extract 4 corners
        # 2 from each bounding rectangle
        ctrl_pts = []
        for i in range(len(flist)):
            rect1 = cv2.minAreaRect(fltr_contours[i][0])
            box1 = cv2.cv.BoxPoints(rect1)
            rect2 = cv2.minAreaRect(fltr_contours[i][1])
            box2 = cv2.cv.BoxPoints(rect2)
            # coordinates 2 from each
            x1, y1 = box1[0]
            x2, y2 = box1[1]
            x3, y3 = box2[0]
            pts = np.array((x1, y1), \
                           (x2, y2), \
                           (x3, y3))
            # -- get data array sorted by n-th column (n= 1)
            pts = pts[np.argsort(pts[:,1])]
            ctrl_pts.append(pts)
        return ctrl_pts
    else :
        # get all centers
        center_contours = find_centers(flist, fltr_contours)
        print center_contours
        pts = []
        for i in range(len(flist)):
            coords = copy(center_contours[i])
            # sort by argsort col n=1 and grab first 3 pts
            tmp = coords[np.argsort(coords[:,1])]
            tmp = concatenate((tmp[:2], tmp[-1:]),axis=0)
            pts.append(tmp)
        return pts


def retrieve_ind(n):
    """ Create a sublist of each groups of frames containing diff. n contours
        Args : n, single-level list of n contours
        Returns the indexes in a two levels list
        Used for treating m+i, n+j, etc frames independently
    """
    tmp = []
    for i in range(len(n)-1):
        if n[i] != n[i+1]:
            tmp.append(i)
    # First range of elements to first index
    ind, r = [], []
    r = list(xrange(tmp[0]))
    r.append(tmp[0])
    ind.append(r)
    # Second range to end of index list
    for i in range(len(tmp)-1):
        e1 = tmp[i]+1
        e2 = tmp[i+1]+1
        ind.append(list(xrange(e1,e2)))
    # Third range from last index in list to length of n cnts list
    t3 = []
    e3 = tmp[i+1]+1
    e4 = len(n)
    ind.append(list(xrange(e3,e4)))
    return ind


def find_anchors(flist, n, fltr_contours):
    """ Returns control points
        a list of indexes grouping frames indexes in a sublist by n cnts
        Arg : n, number of contours per frame (list)
    """
    # if equality of contours in n frames i.e.: n = [x, x, x, x]
    if check_equal(n):
        pts = pull_anchors(flist, n[0], fltr_contours)
        return pts
    else :
        # Unpack in a list of lists (2 levels)
        pts = []
        ind = retrieve_ind(n)
        for i in range(len(ind)):
            pts += pull_anchors(flist[ind[i][0]:ind[i][-1]+1], \
                               n[ind[i][0]:ind[i][-1]+1], \
                               fltr_contours[ind[i][0]:ind[i][-1]+1])
        print "\n>>",len(pts)
        print pts
        return pts


def register(anchors, imgs, flist, rows, cols):
    """
    """
    # use first frame as reference
    src = np.float32(anchors[0])
    align_imgs = []
    # append first image serving as anchor
    align_imgs.append(imgs[0])
    # loop over each RGB img to transform (skip 1st frame)
    for i in range(len(flist[1:])):
        M = cv2.getAffineTransform(np.float32(anchors[i+1]), src)
        dst = cv2.warpAffine(imgs[i+1], M, (cols,rows))
        align_imgs.append(dst) # .astype(np.float32)
    return align_imgs

# ------------------------------------------------------------------------------
#                               CARTESIAN PLAN
# ------------------------------------------------------------------------------

def reshape_coord(center_contours):
    """ Decomposes list of (x,y) into 2 x and y lists, as follows
        [ 'xi, yi', 'xi+1, yi+1', 'xn, yn'] -> [xi, xi+1, xn] & [yi, yi+1, yn]
    """
    x, y = [], []
    for i, j in enumerate(center_contours[:-1]):
        x.append(j[0])
        y.append(j[1])
    return x, y




# ------------------------------------------------------------------------------
#                            REGISTRATION ACCURACY
# ------------------------------------------------------------------------------

def rmsdiff(im1, im2):
    "Calculates the root-mean-square difference between two images"
    im1 = Image.fromarray(ubyte(im1))
    im2 = Image.fromarray(ubyte(im2))
    diff = ImageChops.difference(im1, im2)
    h = diff.histogram()
    sq = (value*(idx**2) for idx, value in enumerate(h))
    # sq = (value*((idx%256)**2) for idx, value in enumerate(h))
    sum_of_squares = sum(sq)
    rms = math.sqrt(sum_of_squares/float(im1.size[0] * im1.size[1]))
    return rms


def rms_call(align_imgs,flist):
    l = []
    for i in range(len(flist)-1):
        tmp = rmsdiff(align_imgs[0],align_imgs[i])
        l.append(tmp)
        print "[%s]" %tmp
    mean = reduce(lambda x, y: x + y, l) / len(l)
    return mean


# ------------------------------------------------------------------------------
#                                   OUTPUT
# ------------------------------------------------------------------------------

def export(flist, dirs, out):
    # Enumerate elements in dirs
    for j in enumerate(dirs):
        create_dir("../"+j[1]) # create all dirs
        for i in range(len(flist)):
            f = "../"+j[1]+ "/" +j[1].split(" - ")[-1]+ "_" + flist[i].split("/")[-1]
            tmp = out[j[0]][i]
            #            print f, tmp
            cv2.imwrite(f, tmp)
