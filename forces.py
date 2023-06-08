from enum import Enum

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize


class ForcesConfig:
    SAFE_DIVISION_EPSILON = 1e-8

    class Strategy(Enum):
        SINGLE_FORCE_SCALES = 1,        # 1 scale factor for all forces
        MULTIPLE_FORCE_SCALES = 2       # n scale factors, 1 per circle

    def __init__(self, force_strength=0.2, mutation_rate=1, probability_to_apply_forces=1,
                 number_of_neighbours=1, strategy=Strategy.SINGLE_FORCE_SCALES,
                 force_epsilon=0.0000001):
        self.force_strength = force_strength                                #initial force_strength
        self.mutation_rate = mutation_rate                                  #probability to rescale the force_strength
        self.probability_to_apply_forces = probability_to_apply_forces      #probability to ignore a force
        self.number_of_neighbours = number_of_neighbours                    #number of nearest neighbours to base the forces on
        self.strategy = strategy                                            #decides how to mutate the force strengths
        self.force_epsilon = force_epsilon                                  #minimum force scale

    def calculate_forces(self, genotype):
        """
        Calculate forces applied to genotype
        """
        if self.probability_to_apply_forces == 1:
            if self.number_of_neighbours >= len(genotype)//2 - 1:
                return self._calculate_forces_full(genotype)
            else:
                return self._calculate_forces_nearest_neighbours(genotype)

        forces = np.zeros(len(genotype))
        apply_forces = np.random.choice([True, False], size=len(genotype)//2,
                                          p = [self.probability_to_apply_forces, 1-self.probability_to_apply_forces])

        if any(apply_forces):
            indices = np.zeros(len(genotype), dtype=np.bool)
            indices[0::2] = apply_forces
            indices[1::2] = apply_forces

            if self.number_of_neighbours >= len(genotype)//2 - 1:
                forces[indices] = self._calculate_forces_full(genotype[indices]).flatten()
            else:
                forces[indices] = self._calculate_forces_nearest_neighbours(genotype[indices]).flatten()

        return forces

    def _calculate_forces_nearest_neighbours(self, genotype):
        """
        Calculate forces applied to genotype considering only the number_of_neighbours nearest neighbours
        """
        HIGH_VALUE = 10
        if self.force_strength <= 0:
            return np.zeros(genotype.shape)
        points = np.reshape(genotype, (-1, 2))
        distances = euclidean_distances(points, squared=True)
        np.fill_diagonal(distances, HIGH_VALUE)
        forces = np.zeros(points.shape)
        for _ in range(self.number_of_neighbours):
            closest_neighbours = np.argmin(distances, axis=-1)
            indices = tuple(zip(*enumerate(closest_neighbours)))
            weights = 1 / (distances[indices] + ForcesConfig.SAFE_DIVISION_EPSILON)
            distances[indices] = HIGH_VALUE
            forces += self.force_strength * weights[:, np.newaxis] * (points - points[closest_neighbours])
        return forces

    def _calculate_forces_full(self, genotype):
        """
        Calculate forces by considering all neighbours
        """

        if self.force_strength <= 0:
            return np.zeros(genotype.shape)
        points = np.reshape(genotype, (-1, 2))
        differences = points[:, np.newaxis] - points
        distances = euclidean_distances(points, squared=True) + ForcesConfig.SAFE_DIVISION_EPSILON
        np.fill_diagonal(distances, self.force_strength)  # "strength" is actually the distances to self
        weights = normalize(1 / distances)
        return np.einsum('ijk,ij->ik', differences, weights).flatten()

        # if self.strategy == ForcesParams.Strategy.ONE_PER_INDIVIDUAL:
        #     self.length = 1
        #     self.force_strengths = [self.initial_force_strength]
        # elif self.strategy == ForcesParams.Strategy.ONE_PER_CIRCLE:
        #     self.length = n_circles
        #     self.force_strengths = [self.initial_force_strength]*n_circles
        # else:
        #     raise ValueError("Provided strategy parameter was not an instance of ForcesParams.Strategy")

    # , strategy=Strategy.ONE_PER_INDIVIDUAL
    # class Strategy(Enum):
    #     """Enum used to distinguish different types of strategies for the forces pararms.
    #
    #     These strategies are used to determine the mechanism which each individual can use to control
    #     its own mutability. The three strategies which are included are:
    #
    #     - ONE_PER_INDIVIDUAL: 1 force strength per individual
    #     - ONE_PER_CIRCLE: n force strengths per individual. 1 per circle
    #     """
    #     ONE_PER_INDIVIDUAL = 1
    #     ONE_PER_CIRCLE = 2
    #
    # def get_force_strength(self):
    #     return self.initial_force_strength
    #     #todo
    #
    #     # if self.strategy == ForcesParams.Strategy.ONE_PER_INDIVIDUAL:
    #     #     return self.force_strengths[0]
    #     # elif self.strategy == ForcesParams.Strategy.ONE_PER_CIRCLE:
    #     #     #todo
    #     #     return self.force_strengths[0]
    #     # else:
    #     #     raise ValueError("Provided strategy parameter was not an instance of ForcesParams.Strategy")
    #
    # def mutate_forces(self):
    #     if self.strategy == ForcesParams.Strategy.ONE_PER_INDIVIDUAL:
    #         self._mutate_forces_one_per_individual()
    #     elif self.strategy == ForcesParams.Strategy.ONE_PER_CIRCLE:
    #         self._mutate_forces_one_per_circle()
    #     else:
    #         raise ValueError("Provided strategy parameter was not an instance of ForcesParams.Strategy")
    #
    # def _mutate_forces_one_per_individual(self):
    #     scale_factor = np.random.randn() * np.sqrt(1 / (2 * self.length))
    #     self.force_strengths = max(self.force_strengths[0] * np.exp(scale_factor), self._EPSILON)
    #
    # def _mutate_forces_one_per_circle(self):
    #     # todo
    #     raise NotImplementedError("")
    #     # global_scale_factor = np.random.randn() * np.sqrt(1 / (2 * self.length))
    #     # scale_factors = [np.random.randn() * np.sqrt(1 / 2 * np.sqrt(self.length))
    #     #                  for _ in range(self.length)]
    #     # self.force_strengths = [max(np.exp(global_scale_factor + scale_factors[i])
    #     #                 * self.force_strengths[i], self._EPSILON)
    #     #                   for i in range(self.length)]

if __name__ == "__main__":
    _genotype = np.array([
        0.4, 0,
        0, 0.5,
        1.1, 0,
        1, 0.95,
        0, 2.1,
        2, 2,
        2, 0.1,
        2, -0.2
    ])
    n_circles = len(_genotype)//2
    fig, axs = plt.subplots(1, n_circles, figsize = (3*n_circles, 3))
    for k in range(n_circles):
        _points = np.reshape(_genotype, (-1, 2))
        _forces = np.reshape(ForcesConfig(number_of_neighbours=k).calculate_forces(_genotype), (-1, 2))
        for p0, p1, f0, f1 in zip(_points[:, 0], _points[:, 1], _forces[:, 0], _forces[:, 1]):
            axs[k].arrow(p0, p1, f0, f1, head_width=0.02, head_length=0.02, fc='k', ec='k')
        axs[k].scatter(_genotype[0:len(_genotype):2], _genotype[1:len(_genotype):2])
        axs[k].set_title(f"{k}-nearest_neighbours")
    plt.setp(axs, xlim=[-1, 3], ylim=[-1, 3])
    plt.show()
