import numpy as np
import pandas as pd


class MyKMeans:
    def __init__(self, n_clusters, tol, iterations, logging=False, log_interval=1):
        self.n_clusters = n_clusters
        self.iterations = iterations
        self.tol = tol
        self.centroids = {}
        self.logging = logging
        self.log_interval = log_interval
        self.classifications = {}

        self.__str__ = "KMeans(n_clusters={}, iterations={}, tol={})".format(
            self.n_clusters, self.iterations, self.tol)


    def __initialize_random_centroids(self, data):
        random_indexes = np.random.choice(data.shape[0], size=self.n_clusters, replace=False)
        for i in range(self.n_clusters):
            self.centroids[i] = data[random_indexes[i]]


    def __initialize_dict_of_empty_lists(self, length):
        this_dict = {}
        for i in range(length):
            this_dict[i] = list()
        return this_dict


    def fit(self,data):
        self.__initialize_random_centroids(data)

        for epoch in range(self.iterations):

            self.classifications = self.__initialize_dict_of_empty_lists(self.n_clusters)

            for row in data:
                distances_to_centroid = [np.linalg.norm(row - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances_to_centroid.index(min(distances_to_centroid))
                self.classifications[classification].append(row)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)


    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[c]) for c in self.centroids]
        return distances.index(min(distances))


if __name__ == "__main__":

    model = MyKMeans(n_clusters=4, tol=0.1, iterations=10)
    data = np.loadtxt('data/customer_churn/customer_churn_processed.txt')

    model.fit(data)

    print(model.classifications)
