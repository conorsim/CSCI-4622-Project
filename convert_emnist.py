"""
File for conversions of the EMNIST dataset to different feature representations

arguments:
-v: version of feature representation
"""

import argparse
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist
from skimage.filters import threshold_otsu
from skimage.morphology import binary_erosion

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--version', required=True, type=int, default=6, help="Version of feature representation")
args = parser.parse_args()

###########################

"""
Feature representation 1:

Columns are...
0 - area
1 - perimeter
2 - radius mean
3 - radius std
4 - theta mean
5 - theta std
6 - range in x direction
7 - range in y direction
8-17 - coefficients of 9-degree polynomial fit on r and theta
"""

def feature_transform_1(X_train, X_test):
    for i in range(2):
        if i == 0:
            str = 'train'
            X = X_train
        else:
            str = 'test'
            X = X_test
        FR = np.zeros((X.shape[0], 18))
        write_path = os.getcwd()+f'/feature_representations/feature_representation_1_{str}.npy'
        with open(write_path, 'w') as csvfile:
            incr = X.shape[0] // 10
            for j in range(X.shape[0]):
                # print statement to view progress
                if j % incr == 0 and i == 0: print(f"{round(100*(j/X.shape[0]), 2)}% of the way through the training set")
                if j % incr == 0 and i == 1: print(f"{round(100*(j/X.shape[0]), 2)}% of the way through the testing set")

                # get binary image and binary image outline
                x = X[j,:].reshape(28,28)
                x = np.divide(x, 255.)
                t = threshold_otsu(x)
                x = np.where(x < t, 0, 1)
                er = binary_erosion(x)
                x_er = x - er

                # find area and permiter of binary image
                area = np.sum(x)
                perimeter = np.sum(x_er)

                # tranform coordinates of binary image to relative polar coordinates
                ii, jj = np.where(x_er == 1)
                it = ii / 28
                jt = jj / 28
                cen = np.mean(it), np.mean(jt)
                it = it - cen[0]
                jt = jt - cen[1]
                r = [np.sqrt(k**2 + l**2) for k,l in list(zip(it, jt))]
                theta = [np.arctan(l/k) if k!= 0 else 0 for k,l in list(zip(it, jt))]

                # find mean and standard deviation of polar coordinates
                r_mean = np.mean(r)
                r_std = np.std(r, ddof=1)
                theta_mean = np.mean(theta)
                theta_std = np.std(theta, ddof=1)

                # find the ranges in x and y direction
                x_range = np.max(jt) - np.min(jt)
                y_range = np.max(it) - np.min(it)

                # find 10-degree polynomial coefficients on polar coordinates
                coeffs = np.polyfit(r, theta, 9)

                # represent features as numpy array
                FR[j, 0] = area
                FR[j, 1] = perimeter
                FR[j, 2] = r_mean
                FR[j, 3] = r_std
                FR[j, 4] = theta_mean
                FR[j, 5] = theta_std
                FR[j, 6] = x_range
                FR[j, 7] = y_range
                FR[j, 8:] = coeffs

            # normalize features and write to np array
            print("Writing to disk")
            norm_FR = np.zeros(FR.shape)
            for k in range(FR.shape[1]):
                col = FR[:, k]
                new_col = (col - np.mean(col)) / np.std(col, ddof=1)
                norm_FR[:, k] = new_col
            np.save(write_path, norm_FR)

###########################

versions = [1]
try:
    args.version in versions
except ValueError:
    print(f"Version {args.version} is not supported. Supported versions are {versions}")

funcs = [feature_transform_1]
func = funcs[args.version - 1]

print(f"Converting data to feature representation {args.version}")

Xtrain, ytrain = loadlocal_mnist(
    images_path='emnist/emnist-letters-train-images-idx3-ubyte',
    labels_path='emnist/emnist-letters-train-labels-idx1-ubyte')

Xtest, ytest = loadlocal_mnist(
    images_path='emnist/emnist-letters-test-images-idx3-ubyte',
    labels_path='emnist/emnist-letters-test-labels-idx1-ubyte')

if not os.path.exists(os.getcwd()+'/feature_representations'):
    os.system('mkdir feature_representations/')

if not os.path.exists(os.getcwd()+'/feature_representations/ytrain.npy'):
    np.save(os.getcwd()+'/feature_representations/ytrain.npy', ytrain)

if not os.path.exists(os.getcwd()+'/feature_representations/ytest.npy'):
    np.save(os.getcwd()+'/feature_representations/ytest.npy', ytest)

func(Xtrain, Xtest)
