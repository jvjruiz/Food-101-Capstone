import numpy as np
from numpy.linalg import norm


class Kmeans:
    '''Implementing Kmeans algorithm.'''

    def __init__(self, n_clusters, max_iter=100, random_state=123):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def initialize_centroids(self, X):
        # randomly choose points from dataset and assign centroids

    def compute_centroids(self, X, labels):
        # compute the new centroids based on the mean of all points per label

    def compute_distance(self, X, centroids):
        # compute distance between each point and each centroid

    def find_closest_cluster(self, distance):
        # find the closest centroid

    def compute_sse(self, X, labels, centroids):
        # computer sum of squared error
    
    def fit(self, X):
        # intiialize centroids
        # per iteration
        # calculate the distance between centroids and all other points
        # assign centroids based on closest centroid
        # find the mean of all centroids
        # assign new centroids
    
    def predict(self, X):
        # find the closest cluster