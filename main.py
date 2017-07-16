#!/usr/bin/env python3

import numpy as np
import torch

import matplotlib.pyplot as plt

if __name__ == '__main__':
    print('hello')

    X = np.random.multivariate_normal([4,1], [[2, 0.5],[0.5, 0.5]], size=100)

    plt.figure()
    plt.scatter(X[:,0], X[:,1]) 
    plt.show()
