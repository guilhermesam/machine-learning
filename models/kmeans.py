import numpy as numpy
import pandas as pd

class KMeans(object):
    def __init__(self, n_clusters, iterations):
        self.n_clusters = n_clusters
        self.iterations = iterations

    