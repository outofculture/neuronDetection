"""Show bounding boxes for detected neurons using the AND_yolo3 model."""
import argparse
import numpy as np
from PIL import Image
import scipy.stats
from yolo import YOLO


def main(filename):
    image = Image.open(filename)
    image = normalize(image)
    image.show(filename)
    my_yolo = YOLO()
    boxes = my_yolo.get_boxes(image)
    print(boxes)
    r_image = my_yolo.detect_image(image)
    r_image.show("with boxes")


def normalize(image: Image, min_in=None, max_in=None):
    if min_in is None:
        # min_in = scipy.stats.scoreatpercentile(image, 5)
        min_in = np.min(image)
        # min_in = 0
    if max_in is None:
        # max_in = scipy.stats.scoreatpercentile(image, 90)
        max_in = np.max(image)
        # max_in = 65535
    min_out = 0
    max_out = 255  # maxmimum intensity (output)
    image = (image - np.uint16(min_in)) * (
            ((max_out - min_out) / (max_in - min_in)) + min_out
    )
    # image = (image - np.uint16(min_in)) * (
    #     (max_out / (max_in - min_in))
    # )
    image = scipy.ndimage.zoom(image, 3)
    offset_w = int(image.shape[0] * 0.3)
    offset_h = int(image.shape[1] * 0.3)
    margin_w = int(image.shape[0] * 0.4)
    margin_h = int(image.shape[1] * 0.4)
    image = image[offset_w:offset_w + margin_w, offset_h:offset_h + margin_h]
    # image = np.rot90(image)
    return Image.fromarray(image.astype(np.uint8))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument("input", help="16-bit? tiff image file")

    args = argparser.parse_args()
    main(args.input)
