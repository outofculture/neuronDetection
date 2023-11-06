"""Show bounding boxes for detected neurons using the AND_yolo3 model."""
import argparse
import numpy as np
from PIL import Image
import scipy.stats
from yolo import YOLO


if "__main__" == __name__:
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument("input", help="tiff image file")

    args = argparser.parse_args()
    image = Image.open(args.input)
    # normalize to 8-bit
    image -= np.min(image)
    image = image / np.max(image)  #  scipy.stats.scoreatpercentile(image, 90)
    image *= 255
    image = Image.fromarray(image.astype(np.uint8))
    image.show()

    my_yolo = YOLO()
    r_image = my_yolo.detect_image(image)
    r_image.show()
