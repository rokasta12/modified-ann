from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
import nn

data = datasets.make_blobs(n_samples=1000, centers=2, random_state=2)
X = data[0].T
y = np.expand_dims(data[1], 1).T

print(X)
