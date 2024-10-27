import cv2
import numpy as np
from keras import datasets
import matplotlib.pyplot as plt

(train_imgs, train_lbls), (test_imgs, test_lbls) = datasets.fashion_mnist.load_data()

def main():
    image = GetGrayImage()
    bw = BinarizeImage(image)
    ShowMontage((image, bw))
    ShowHistogram(image)
    ShowHistogram(bw)
    ShowSignal(image)
    ShowSignal(bw)

def GetGrayImage(index: int | None=None, image_set=train_imgs):
    '''
    Get an image from the image set.  
    - If it's color (i.e. 3-D array), convert to grayscale (i.e. 2-D array)
    - If index is None, use a random image
    '''
    
    rng = np.random.default_rng()

    if(index is None):
        image = np.array(image_set[rng.integers(1, len(train_imgs))])
    else:
        image = np.array(image_set[index])

    if(len(np.shape(image)) == 3): # 3-D (color image (RGB)) - convert to Grayscale (2-D)
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else: 
        return image

def BinarizeImage(image=train_imgs[42]):
    '''Converts an image to binary (i.e. pixel > threshold == 255 (white); else == 0 (black))'''
    
    th, bw = cv2.threshold(image, 128, 255, cv2.THRESH_OTSU) 
    return bw

def ShowHistogram(image):
    '''
    Plot the histogram for the specified image.
    - Notice, there are a few options for getting the histogram.
    - For example, here I'm trying out:
        1. Numpy
        2. CV
        3. MatplotLib

    - MatplotLib's _hist()_ function handles both computing and plotting of the histogram
    - With the others, they are separate operations
    '''
    
    # Numpy histogram()
    nHist, edges = np.histogram(image, bins=256, range=(0,256))
    nHist = nHist.T
    plt.plot(nHist, color='r')

    # CV calcHist()
    cvHist = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.plot(cvHist, color='b')

    # MatlibPlot hist()
    plt.hist(image.ravel(), 256, (0,255))

    plt.xlim([0, 256])
    plt.ylim([0,256])
    plt.show()

def ShowSignal(image):
    '''Shows the image signal (aka row sums)'''

    image_signal = np.sum(image, 1)
    plt.plot(image_signal)
    plt.show()

def ShowMontage(images):
    '''
    Creates a montage of the provided images.
    - Note the use of Numpy's _concatenate()_ function.  
      - In this case the images are simply appended to each other, increasing the image width.
      - TODO: Append images such that the width and height are adjusted, keeping the montage square-ish
    '''

    montage = np.concatenate(images, axis=1)
    plt.imshow(montage, cmap="gray")
    plt.show()


if (__name__ == "__main__"):
    main()
