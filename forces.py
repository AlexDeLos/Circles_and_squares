import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize

SAFE_DIVISION_EPSILON = 1e-8
#place the next function in comments to use the old version
'''def calculate_forces_full(genotype, strength=0.00000005):
    """
    Calculate forces by considering all neighbours
    """
    if strength <= 0:
        return np.zeros(genotype.shape)
    points = np.reshape(genotype, (-1, 2))
    differences = points[:, np.newaxis] - points
    distances = euclidean_distances(points, squared=True) + SAFE_DIVISION_EPSILON
    np.fill_diagonal(distances, strength) #"strength" is actually the distances to self
    weights = normalize(1/distances)
    return np.einsum('ijk,ij->ik', differences, weights).flatten()'''

#Mutate the strength parameter below to change the strength of the forces
def mutate_force_strength(strength, mutate_rate=0.05):
    #if np.random.rand() <= mutate_rate:
    return strength + np.random.randn() * 0.01
    #return strength

def calculate_forces(genotype, strength=0.25):
    """
    Calculate forces by only considering the closest neighbours
    """
    if strength <= 0:
        return np.zeros(genotype.shape)
    points = np.reshape(genotype, (-1, 2))
    distances = euclidean_distances(points, squared=True)
    np.fill_diagonal(distances, 10)
    closest_neighbours = np.argmin(distances, axis=-1)
    weights = 1/(distances[tuple(zip(*enumerate(closest_neighbours)))] + SAFE_DIVISION_EPSILON)
    strength = mutate_force_strength(strength)
    return strength*weights[:, np.newaxis] * (points - points[closest_neighbours])

def plot_forces(genotype, strength=0.25):
    points = np.reshape(genotype, (-1, 2))
    forces = np.reshape(calculate_forces(genotype, strength=strength), (-1, 2))
    for p0, p1, f0, f1 in zip(points[:, 0], points[:, 1], forces[:, 0], forces[:, 1]):
        plt.arrow(p0, p1, f0, f1, head_width=0.02, head_length=0.02, fc='k', ec='k')

if __name__ == "__main__":
    arr = np.array([
        0.4,   0,
        0,      0.5,
        1.1,    0,
        1,      0.95
    ])
    plot_forces(arr)
    plt.scatter(arr[0:len(arr):2], arr[1:len(arr):2])
    plt.show()