import numpy as np
from PIL import Image
#from init_seg_output import make_init_seg
from util import init_seg_needed, output, save_init_seg
from rag_builder import build_rag
from skimage.segmentation import slic
from skimage.future import graph

def main(image, weight_mode = 'similarity', compactness = 30, n_segments = 100):
    '''
    Performs an initial segmentation of input image into regions, then performs
    an n-cut segmentation on the region adjacency graph corresponding
    to these. Data for the initial segmentation is saved, so that it does not
    need to be performed unnecessarily.
    '''

    image_title = sum(np.cumsum([ord(c) for c in image])) # Primitive hash of image title to integer.

    image = "./images/" + image

    img = Image.open(image)
    img = np.array(img)

    isn = init_seg_needed(img, image_title, compactness, n_segments)
    # isn = True
    # isn = False

    if isn:
        labels = slic(img, compactness=compactness, n_segments=n_segments)
        save_init_seg(labels, img, image_title, compactness, n_segments)
    else:
        labels = np.load('labels.npy')

    g = build_rag(labels, weight_mode)

    labels2 = graph.cut_normalized(labels, g)

    output(img, labels)
    output(img, labels2)

if __name__=='__main__':
    image = "landscape.jpg"
    main(image)
    # main(image, weight_mode = 'laplacian')
