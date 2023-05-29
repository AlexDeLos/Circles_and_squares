import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances, manhattan_distances
from sklearn.preprocessing import normalize

SAFE_DIVISION_EPSILON = 1e-8
def calculate_forces(genotype, strength=0.00000005):
    if strength <= 0:
        return np.zeros(genotype.shape)
    points = np.reshape(genotype, (-1, 2))
    differences = points[:, np.newaxis] - points
    distances = euclidean_distances(points, squared=True) + SAFE_DIVISION_EPSILON
    np.fill_diagonal(distances, strength) #"strength" is actually the distances to self
    weights = normalize(1/distances)
    return np.einsum('ijk,ij->ik', differences, weights).flatten()

def plot_forces(genotype, strength=0.00000005):
    points = np.reshape(genotype, (-1, 2))
    forces = np.reshape(calculate_forces(genotype, strength=strength), (-1, 2))
    for p0, p1, f0, f1 in zip(points[:, 0], points[:, 1], forces[:, 0], forces[:, 1]):
        plt.arrow(p0, p1, f0, f1, head_width=0.01, head_length=0.02, fc='k', ec='k')
