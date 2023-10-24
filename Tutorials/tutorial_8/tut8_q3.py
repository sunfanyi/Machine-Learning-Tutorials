# -*- coding: utf-8 -*-
# @File    : tut8_q3.py
# @Time    : 01/12/2022
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

# Question 3: Data Processing (Image)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def show_image_greyscale(img):
    plt.figure()
    plt.imshow(img, cmap=plt.cm.gray)
    plt.colorbar()
    plt.axis('off')
    # plt.show()


def fourier_processing(imgBw):
    npy, npx = imgBw.shape
    fim = np.fft.fft2(imgBw)

    img = np.fft.fftshift(np.abs(fim))
    plt.figure()
    plt.imshow(img)
    # show the colour scale:
    plt.colorbar()
    # set the colour limits to something so we can see the image better
    plt.clim(0, 1e3)
    # plt.show()

    # make a separate copy of the image to work on:
    fim2 = fim.copy()
    # set everything in all except the first row to zero
    fim2[1:npy, :] = 0
    # set everything in the first row to zero except the first 61 points and the
    # last 60
    fim2[0, 61:npx - 60] = 0

    # inverse fourier transform after processing
    img2 = np.real(np.fft.ifft2(fim2))
    show_image_greyscale(img2)


def edge_detection(imgBw):
    diffx = np.diff(imgBw, axis=1)
    diffy = np.diff(imgBw, axis=0)

    # to make their dimensions equal
    npy, npx = imgBw.shape
    diffx = diffx[0:npy - 1, :]
    diffy = diffy[:, 0:npx - 1]

    edgeIm = np.sqrt(np.square(diffx) + np.square(diffy))

    return edgeIm


def thresholding(t, img):
    edgeThresh = (img > t).astype('int')

    return edgeThresh


if __name__ == '__main__':
    img = mpimg.imread('window.png')

    # just extract one channel - technically this is red, but
    # it doesn't matter since they are all equal (raw image is in greyscale)
    imgBw = np.squeeze(img[:, :, 0])

    show_image_greyscale(imgBw)

    fourier_processing(imgBw)

    edgeIm = edge_detection(imgBw)
    show_image_greyscale(edgeIm)

    t = 0.35
    edgeThresh = thresholding(t, edgeIm)
    show_image_greyscale(edgeThresh)
    plt.show()


