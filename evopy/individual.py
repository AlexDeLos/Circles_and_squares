"""Module containing the individuals of the evolutionary strategy algorithm."""
import numpy as np
from matplotlib import pyplot as plt

from evopy.strategy import Strategy
from evopy.utils import random_with_seed
from forces import ForcesConfig


class Individual:
    """The individual of the evolutionary strategy algorithm.

    This class handles the reproduction of the individual, using both the genotype and the specified
    strategy.

    For the full variance reproduction strategy, we adopt the implementation as described in:
    [1] Schwefel, Hans-Paul. (1995). Evolution Strategies I: Variants and their computational
        implementation. G. Winter, J. Perieaux, M. Gala, P. Cuesta (Eds.), Proceedings of Genetic
        Algorithms in Engineering and Computer Science, John Wiley & Sons.
    """
    _BETA = 0.0873
    _EPSILON = 0.01

    def __init__(self, genotype, strategy, strategy_parameters, bounds=None, random_seed=None, forces_config: ForcesConfig=None, forces_scale_param=None, inertia=None, mutation_rate=1):
        """Initialize the Individual.

        :param genotype: the genotype of the individual
        :param strategy: the strategy chosen to reproduce. See the Strategy enum for more
                         information
        :param strategy_parameters: the parameters required for the given strategy, as a list
        :param forces_scale_param: factor of 1 that can be mutated to scale the force strength
        """
        self.genotype = genotype
        self.length = len(genotype)
        self.random_seed = random_seed
        self.random = random_with_seed(self.random_seed)
        self.fitness = None
        self.constraint = None
        self.bounds = bounds
        self.strategy = strategy
        self.strategy_parameters = strategy_parameters
        self.age = 0
        self.inertia = np.zeros(len(genotype)) if inertia is None else inertia
        if not isinstance(strategy, Strategy):
            raise ValueError("Provided strategy parameter was not an instance of Strategy.")
        if strategy == Strategy.SINGLE_VARIANCE and len(strategy_parameters) == 1:
            self.reproduce = self._reproduce_single_variance
        elif strategy == Strategy.MULTIPLE_VARIANCE and len(strategy_parameters) == self.length:
            self.reproduce = self._reproduce_multiple_variance
        elif strategy == Strategy.FULL_VARIANCE and len(strategy_parameters) == self.length * (
                self.length + 1) / 2:
            self.reproduce = self._reproduce_full_variance
        else:
            raise ValueError("The length of the strategy parameters was not correct.")

        self.forces_config = forces_config

        if not self.forces_config is None:
            if forces_scale_param is None:
                if self.forces_config.strategy == ForcesConfig.Strategy.SINGLE_FORCE_SCALES:
                    self.forces_scale_param = 1
                if self.forces_config.strategy == ForcesConfig.Strategy.MULTIPLE_FORCE_SCALES:
                    self.forces_scale_param = np.array([1 for _ in range(self.length // 2)])
            else:
                self.forces_scale_param = forces_scale_param
        else:
            self.forces_scale_param = None

        self.mutation_rate = mutation_rate

    def evaluate(self, fitness_function):
        """Evaluate the genotype of the individual using the provided fitness function.

        :param fitness_function: the fitness function to evaluate the individual with
        :return: the value of the fitness function using the individuals genotype
        """
        self.fitness = fitness_function(self.genotype)

        return self.fitness

    def _mutate_forces(self):
        """ Mutates the forces_scale_param """
        if self.forces_config.strategy == ForcesConfig.Strategy.SINGLE_FORCE_SCALES:
            if self.random.rand() <= self.forces_config.mutation_rate:
                scale_factor = self.random.randn() * np.sqrt(1 / (2 * self.length))
                self.forces_scale_param = max(self.forces_scale_param * np.exp(scale_factor), self.forces_config.force_epsilon)
        elif self.forces_config.strategy == ForcesConfig.Strategy.MULTIPLE_FORCE_SCALES:
            global_scale_factor = self.random.randn() * np.sqrt(1 / (2 * self.length))
            scale_factors = [self.random.randn() * np.sqrt(1 / (2 * self.length))
                             for _ in range(len(self.forces_scale_param))]
            self.forces_scale_param = np.array([max(np.exp(global_scale_factor + scale_factors[i])
                                  * self.forces_scale_param[i], self.forces_config.force_epsilon)
                              for i in range(len(self.forces_scale_param))])

    def _distribution_mean(self):
        """
        Shifts the genotype according to forces
        :return: mean of the sample distribution
        """

        mean = self.genotype
        if not self.forces_config is None:
            # Mutate the forces_scale_param
            if self.random.rand() <= self.forces_config.mutation_rate:
                scale_factor = self.random.randn() * np.sqrt(1 / (2 * self.length))
                self.forces_scale_param = max(self.forces_scale_param * np.exp(scale_factor), 0.000001)
            
            
            change_vector = self.forces_scale_param*self.forces_config.calculate_forces(self.genotype).flatten()
            change = np.zeros(self.length)
            
            for i in range(int((self.length/2))):
                x = [change_vector[i] +self.inertia[i], change_vector[i+ int(self.length/2)] +self.inertia[i+ int(self.length/2)]]
                y = x/(np.linalg.norm(x)+0.0000001)
                change[i] = y[0]
                change[i+ int(self.length/2)] = y[1]
            self.inertia = change

            return mean + change
        



            # inertia
            # change_vector = self.forces_scale_param*self.forces_config.calculate_forces(self.genotype).flatten() + self.inertia
            #
            # # normalize it
            # change = np.zeros(self.length)

            # result = mean + change
            # self.inertia = change
            # return result
        return mean

    def _handle_oob_indices(self, new_genotype):
        # Originally: "Randomly sample out of bounds indices"
        #oob_indices = (new_genotype < self.bounds[0]) | (new_genotype > self.bounds[1])
        #new_genotype[oob_indices] = self.random.uniform(self.bounds[0], self.bounds[1], size=np.count_nonzero(oob_indices))
        # Clip out of bounds indices instead
        return np.clip(new_genotype, self.bounds[0], self.bounds[1])

    def _reproduce_single_variance(self):
        """Create a single offspring individual from the set genotype and strategy parameters.

        This function uses the single variance strategy.

        :return: an individual which is the offspring of the current instance
        """
        if self.random.rand() <= self.mutation_rate:
            new_genotype = self._distribution_mean() + self.strategy_parameters[0] * self.random.randn(self.length)
        else:
            new_genotype = self._distribution_mean()
        scale_factor = self.random.randn() * np.sqrt(1 / (2 * self.length))
        new_parameters = [max(self.strategy_parameters[0] * np.exp(scale_factor), self._EPSILON)]
        new_genotype = self._handle_oob_indices(new_genotype)
        return Individual(new_genotype, self.strategy, new_parameters, forces_config=self.forces_config, bounds=self.bounds, random_seed=self.random, mutation_rate=self.mutation_rate, inertia=self.inertia, forces_scale_param=self.forces_scale_param)

    def _reproduce_multiple_variance(self):
        """Create a single offspring individual from the set genotype and strategy.

        This function uses the multiple variance strategy.

        :return: an individual which is the offspring of the current instance
        """
        new_genotype = self._distribution_mean() + [self.strategy_parameters[i] * self.random.randn()
                                        for i in range(self.length)]
        global_scale_factor = self.random.randn() * np.sqrt(1 / (2 * self.length))
        scale_factors = [self.random.randn() * np.sqrt(1 / 2 * np.sqrt(self.length))
                         for _ in range(self.length)]
        new_parameters = [max(np.exp(global_scale_factor + scale_factors[i])
                              * self.strategy_parameters[i], self._EPSILON)
                          for i in range(self.length)]
        new_genotype = self._handle_oob_indices(new_genotype)
        return Individual(new_genotype, self.strategy, new_parameters, forces_config=self.forces_config, bounds=self.bounds, mutation_rate=self.mutation_rate, inertia=self.inertia, forces_scale_param=self.forces_scale_param)

    # pylint: disable=invalid-name
    def _reproduce_full_variance(self):
        """Create a single offspring individual from the set genotype and strategy.

        This function uses the full variance strategy, as described in [1]. To emphasize this, the
        variable names of [1] are used in this function.

        :return: an individual which is the offspring of the current instance
        """
        global_scale_factor = self.random.randn() * np.sqrt(1 / (2 * self.length))
        scale_factors = [self.random.randn() * np.sqrt(1 / 2 * np.sqrt(self.length))
                         for _ in range(self.length)]
        new_variances = [max(np.exp(global_scale_factor + scale_factors[i])
                             * self.strategy_parameters[i], self._EPSILON)
                         for i in range(self.length)]
        new_rotations = [self.strategy_parameters[i] + self.random.randn() * self._BETA
                         for i in range(self.length, len(self.strategy_parameters))]
        new_rotations = [rotation if abs(rotation) < np.pi
                         else rotation - np.sign(rotation) * 2 * np.pi
                         for rotation in new_rotations]
        T = np.identity(self.length)
        for p in range(self.length - 1):
            for q in range(p + 1, self.length):
                j = int((2 * self.length - p) * (p + 1) / 2 - 2 * self.length + q)
                T_pq = np.identity(self.length)
                T_pq[p][p] = T_pq[q][q] = np.cos(new_rotations[j])
                T_pq[p][q] = -np.sin(new_rotations[j])
                T_pq[q][p] = -T_pq[p][q]
                T = np.matmul(T, T_pq)
        new_genotype = self.genotype + T @ self.random.randn(self.length)
        new_genotype = self._handle_oob_indices(new_genotype)
        return Individual(new_genotype, self.strategy, new_variances + new_rotations, forces_config=self.forces_config, bounds=self.bounds, mutation_rate=self.mutation_rate, inertia=self.inertia, forces_scale_param=self.forces_scale_param)

    def plot_distribution_mean(self):
        points = np.reshape(self.genotype, (-1, 2))
        delta = np.reshape(self._distribution_mean()-self.genotype, (-1, 2))
        for p0, p1, f0, f1 in zip(points[:, 0], points[:, 1], delta[:, 0], delta[:, 1]):
            plt.arrow(p0, p1, f0, f1, head_width=0.02, head_length=0.02, fc='k', ec='k')