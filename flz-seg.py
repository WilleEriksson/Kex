from skimage import segmentation as sgm
import numpy as np
from random import randrange
from PIL import Image

def nseg(inputlist):
    # Determines the number of segments
    return max(max(inputlist.max(1)),max(inputlist.max(0)))

def getimg(loaded,w,h):
    # Converts loaded image into np-array of 3-tuples

    img = np.zeros((h, w, 3))

    for j in range(h):
        for i in range(w):
            img[j][i]=loaded.getpixel((i,j))

    return img

def getcolors(seg):
    # Randomly generates unqiue colors for every segment
    colors = []

    for i in range(nseg(seg)+1):
        c = (randrange(256),randrange(256),randrange(256))
        colors.append(c)

    return colors

def getresult(seg,w,h):
    # Returns an 'RGB' image where every segment is uniquely colored
    result = Image.new('RGB', (w, h))
    colors = getcolors(seg)

    for j in range(h):
        for i in range(w):
            c=colors[seg.item(j*w+i)]
            result.putpixel((i,j),c)

    return result

def main(image,mode='show'):
    '''
    Performs a Felzenswalb segmenation on an input image
    and returns an image where every segment is uniquely colored.
    '''
    print("Preparing...")

    loaded = Image.open(image) # Using library PIL

    loaded = loaded.convert(mode='RGB')
    w,h = loaded.size

    img = getimg(loaded,w,h)

    print("Segmenting...")
    seg = sgm.felzenszwalb(img,1,0.8,1000) # Using the sci-kit image library

    print("Coloring...")

    result = getresult(seg,w,h)

    if mode == 'show':
        result.show("segmented.png")
    elif mode == 'save':
        result.save("segmented.png")

if __name__ == "__main__":
    image = "kub.png"
    image = "./images/" + image
    main(image)
