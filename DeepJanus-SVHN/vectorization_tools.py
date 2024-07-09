import xml.etree.ElementTree as ET
import potrace
import numpy as np
from config import IMG_CHN
from copy import deepcopy
import numpy as np
import cv2


def print_bw(bw):
    from matplotlib import pyplot as plt
    v = bw
    v = np.expand_dims(v, axis=-1)
    v = np.repeat(v, 3, axis=2)
    v = v.astype('uint8')
    v = v.reshape(32, 32, 3)
    plt.imsave("channel_bw.png", v, cmap='gray')


def bitmap_count(image, threshold):
    bkp = deepcopy(image)
    bkp = np.asarray(bkp)
    #bkp = bkp[ 5:-5,:]
    count_b = 0
    count_w = 0
    for x in np.nditer(bkp):
        if x > threshold:
            count_w += 1
        else:
            count_b += 1
    #print("black"+str(count_b))
    #print("white"+str(count_w))
    return count_w > count_b + 120


def preprocess(npdata):
    bw = np.asarray(npdata).copy()
    bw = bw.astype('uint8')
    # Pixel range is 0...255, 256/2 = 128
    import cv2
    tshd, bw = cv2.threshold(bw, 128, 255, cv2.THRESH_OTSU)

    if bitmap_count(bw, tshd):
        #print("HERE")
        bw = 255 - bw
    #print(_)
    #exit()
    #bw[bw < 30] = 0  # Black
    #bw[bw >= 30] = 255  # White
    #bw[bw < 128] = 0  # Black
    #bw[bw >= 128] = 255  # White
    #print_bw(bw)
    # Normalization
    bw = bw / 255.0
    return bw

def preprocess_negative(npdata):
    bw = np.asarray(npdata).copy()
    # Pixel range is 0...255, 256/2 = 128
    bw[bw >= 128] = 0  # Black
    bw[bw < 128] = 255  # White
    print_bw(bw)
    # Normalization
    bw = bw / 255.0
    return bw

def createSVGpath(path):
    path_desc = ""
    # Iterate over path curves
    for curve in path:
        path_desc = path_desc + " M " + str(curve.start_point[0]) + "," + str(curve.start_point[1])
        for segment in curve:
            if segment.is_corner:
                path_desc = path_desc + " L " + str(segment.c[0]) + "," + str(segment.c[1]) \
                            + " L " + str(segment.end_point[0]) + "," + str(
                    segment.end_point[1])
            else:
                path_desc = path_desc + " C " + str(segment.c1[0]) + "," + str(segment.c1[1]) + " " + str(
                    segment.c2[0]) + "," + str(segment.c2[1]) + " " + str(segment.end_point[0]) + "," + str(
                    segment.end_point[1])
    return path_desc + " Z"


def create_svg_xml(desc):
    root = ET.Element("svg")
    root.set("version", "1.0")
    root.set("xmlns", "http://www.w3.org/2000/svg")
    root.set("height", str(32))
    root.set("width", str(32))
    path = ET.SubElement(root, "path")
    path.set("d", desc)
    tree = ET.ElementTree(root)
    tree = tree.getroot()
    xml_str = ET.tostring(tree, encoding='unicode', method='xml')
    return xml_str


def vectorize(image):
    if IMG_CHN > 1:

        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #image = np.mean(image, axis=2)
    array = preprocess(image)
    #array = preprocess_negative(image)
    #if not np.any(array):
    #    array = preprocess_negative(image)
    #    print("HERE")
    # use Potrace lib to obtain a SVG path from a Bitmap
    # Create a bitmap from the array
    bmp = potrace.Bitmap(array)
    # Trace the bitmap to a path
    path = bmp.trace()
    desc = createSVGpath(path)
    return create_svg_xml(desc)

def vectorize3d(image):
    image = np.mean(image, axis=2)
    return vectorize(image)
