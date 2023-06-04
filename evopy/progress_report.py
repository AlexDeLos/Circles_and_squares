"""Module containing the ProgressReport class, used to report on the progress of the optimizer."""
from evopy.individual import Individual

class ProgressReport:
    """Class representing a report on an intermediate state of the learning process."""

    def __init__(self, generation, evaluations, best_individual: Individual, best_genotype, best_fitness, avg_fitness, std_fitness, time_elapsed, is_final_report):
        """Initializes the report instance.

        :param generation: number identifying the reported generation
        :param best_genotype: the genotype of the best individual of that generation
        :param best_fitness: the fitness of the best individual of that generation
        """
        self.generation = generation
        self.evaluations = evaluations
        self.best_individual = best_individual
        self.best_genotype = best_genotype
        self.best_fitness = best_fitness
        self.avg_fitness = avg_fitness
        self.std_fitness = std_fitness
        self.time_elapsed = time_elapsed
        self.is_final_report = is_final_report
