import matplotlib
# import random

from fitness_plots import FitnessPlots, TrackableVariable
from forces import plot_forces

matplotlib.use('Qt5Agg')

import math
from evopy import EvoPy, Strategy
from evopy import ProgressReport
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import numpy as np


###########################################################
#                                                         #
# EvoPy framework from https://github.com/evopy/evopy     #
# Read documentation on github for further information.   #
#                                                         #
# Adjustments by Renzo Scholman:                          #
#       Added evaluation counter (also in ProgressReport) #
#       Added max evaluation stopping criterion           #
#       Added random repair for solution                  #
#       Added target fitness value tolerance              #
#                                                         #
# Original license stored in LICENSE file                 #
#                                                         #
# Install required dependencies with:                     #
#       pip install -r requirements.dev.txt               #
#                                                         #
###########################################################

# np/scipy CiaS implementation is faster for higher problem dimensions, i.e, more than 11 or 12 circles.
def circles_in_a_square_scipy(individual):
    points = np.reshape(individual, (-1, 2))
    dist = euclidean_distances(points)
    np.fill_diagonal(dist, 1e10)
    return np.min(dist)


# Pure python implementation is faster for lower problem dimensions
def circles_in_a_square(individual):
    n = len(individual)
    distances = []
    for i in range(0, n - 1, 2):
        for j in range(i + 2, n, 2):
            distances.append(math.sqrt(math.pow((individual[i] - individual[j]), 2)
                                       + math.pow((individual[i + 1] - individual[j + 1]), 2)))
    return min(distances)


class CirclesInASquare:
    def __init__(self, n_circles, output_statistics=False, plot_sols=False, print_sols=False, save_sols=False, number_of_runs=1):
        self.evopy = None
        self.print_sols = print_sols
        self.output_statistics = output_statistics
        self.plot_best_sol = plot_sols
        self.save_sols = save_sols
        self.n_circles = n_circles
        self.force_strength = None
        self.fig = None
        self.ax = None
        assert 2 <= n_circles <= 20

        if self.plot_best_sol or self.save_sols:
            self.set_up_plot()

        if self.output_statistics:
            self.statistics_header()

        self.number_of_runs = number_of_runs
        self.fitness_plots = FitnessPlots(number_of_runs)

    def set_up_plot(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("$x_0$")
        self.ax.set_ylabel("$x_1$")
        self.ax.set_title("Best solution in generation 0")
        if self.plot_best_sol:
            self.fig.show()

    def statistics_header(self):
        if self.print_sols:
            print("Generation Evaluations Best-fitness (Best individual..)")
        else:
            print("Time Elapsed Generation Evaluations Best-fitness")

    def statistics_callback(self, report: ProgressReport):
        if self.output_statistics:
            formatted_time = str(report.time_elapsed).split(".")[0] + "s"
            output = "{:>10s} {:>10d} {:>11d} {:>12.8f} {:>12.8f} {:>12.8f}".format(formatted_time, report.generation,
                                                                                    report.evaluations,
                                                                                    report.best_fitness, report.avg_fitness,
                                                                                    report.std_fitness)

            if self.print_sols:
                output += " ({:s})".format(np.array2string(report.best_genotype))
            print(output)

        if self.plot_best_sol or self.save_sols:
            points = np.reshape(report.best_genotype, (-1, 2))
            self.ax.clear()
            self.ax.scatter(points[:, 0], points[:, 1], clip_on=False, color="black")
            if self.force_strength > 0:
                plot_forces(points, strength=self.force_strength)
            self.ax.set_xlim((0, 1))
            self.ax.set_ylim((0, 1))
            self.ax.set_title(f"Best solution in generation {report.generation} (Fitness {report.best_fitness})")
            if self.plot_best_sol:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            if self.save_sols and (report.generation == 0 or report.is_final_report):
                plt.savefig(f"results/{self.fitness_plots.get_current_state().replace('.', '').replace('/', '')}@Generation{report.generation}")

        self.add_to_fitness_plots(report)

    def add_to_fitness_plots(self, report: ProgressReport):
        if self.fitness_plots:
            self.fitness_plots.add(report)

    def get_target(self):
        values_to_reach = [
            1.414213562373095048801688724220,  # 2
            1.035276180410083049395595350499,  # 3
            1.000000000000000000000000000000,  # 4
            0.707106781186547524400844362106,  # 5
            0.600925212577331548853203544579,  # 6
            0.535898384862245412945107316990,  # 7
            0.517638090205041524697797675248,  # 8
            0.500000000000000000000000000000,  # 9
            0.421279543983903432768821760651,  # 10
            0.398207310236844165221512929748,
            0.388730126323020031391610191835,
            0.366096007696425085295389370603,
            0.348915260374018877918854409001,
            0.341081377402108877637121191351,
            0.333333333333333333333333333333,
            0.306153985300332915214516914060,
            0.300462606288665774426601772290,
            0.289541991994981660261698764510,
            0.286611652351681559449894454738
        ]

        return values_to_reach[self.n_circles - 2]

    def run_evolution_strategies(self, generations=1000, num_children=1, max_age=0, strategy=Strategy.SINGLE_VARIANCE,
                                 population_size=30, max_evaluations=1e5, use_warm_start = True, force_strength=0):
        if self.plot_best_sol or self.save_sols:
            #quick and dirty way to reset the plot when the same runner is reused with new parameters
            self.set_up_plot()

        callback = self.statistics_callback

        best_solutions = []

        self.force_strength = force_strength

        for current_run in range(self.number_of_runs):
            self.fitness_plots.set_run(current_run)

            self.evopy = EvoPy(
                circles_in_a_square if self.n_circles < 12 else circles_in_a_square_scipy,  # Fitness function
                self.n_circles * 2,  # Number of parameters
                reporter=callback,  # Prints statistics at each generation
                maximize=True,
                generations=generations,
                population_size=population_size,
                bounds=(0, 1),
                target_fitness_value=self.get_target(),
                max_evaluations=max_evaluations,
                num_children=num_children,
                max_age=max_age,
                strategy=strategy,
                warm_start = self.getWarmStart(population_size) if use_warm_start else None,
                force_strength = self.force_strength
            )

            best_solutions.append(self.evopy.run())

        if self.plot_best_sol or self.save_sols:
            plt.close()

        if len(best_solutions) == 1:
            return best_solutions[0]
        return best_solutions

    def toBaseN(self, x:int, base:int):
        newNumber = []
        while not x == 0:
            newD = x%base
            newNumber.insert(0,newD)
            x = int((x -newD)/base)
        return newNumber

    def getWarmStart(self, population_size):
        """
        Returns a set of positions in cordinates
        """
        return np.asarray([
            self.getWarmStartIndividual()
            for _ in range(population_size)
        ])

    def getWarmStartIndividual(self):
        n = int(self.n_circles**(1/2)) # how many we can fit in each grid cleanly
        result = []
        fun = lambda x: int(x)/(n-1)
        for el in range(n**2):
            i = self.toBaseN(el,n) #turn it to base whatever
            while len(i) < 2:
                i.insert(0,'0')
            result.append(list(map(fun,i)))

        # fill in the rest:
        while len(result) < self.n_circles:
            result.append([np.random.uniform(),np.random.uniform()])

        # transform to the format they should be
        result = sum(result, [])

        # add random noise to it
        result = result + np.random.normal(loc=0, scale=1/(10*(n-1)), size=self.n_circles*2)
        result = np.clip(result, 0, 1)

        return result



def main():
    """
    Original main function
    """
    circles = 10
    runner = CirclesInASquare(circles, plot_sols=True, output_statistics=True)
    runner.run_evolution_strategies(generations=1000, max_age=1000, num_children=4)

def fitness_plots_from_backup(number_of_error_bars=float("inf")):
    circles = 10
    runner = CirclesInASquare(circles)
    runner.fitness_plots = FitnessPlots.from_backup()
    runner.fitness_plots.show(number_of_error_bars=number_of_error_bars)

def experiment1():
    """
    Shows severals plots for the `num_children` and `max_age`
    """
    circles = 10
    runner = CirclesInASquare(circles, plot_sols=False, number_of_runs=10)
    for num_children in [1, 2, 3, 4]:
        runner.fitness_plots.set_subplot(f"Number of Children = {num_children}")
        for max_age in [0, 1, 5, 1000]:
            runner.fitness_plots.set_line(f"Max Age = {max_age}")
            runner.run_evolution_strategies(generations=1000, num_children=num_children, max_age=max_age)
    runner.fitness_plots.show()


def experiment2():
    """
    Shows several plots for different `num_children` and `strategy`
    """
    circles = 10
    runner = CirclesInASquare(circles, plot_sols=False)
    for num_children in [1, 2, 3, 4]:
        runner.fitness_plots.set_subplot(f"Number of Children = {num_children}")
        for strategy in [Strategy.SINGLE_VARIANCE, Strategy.MULTIPLE_VARIANCE, Strategy.FULL_VARIANCE]:
            runner.fitness_plots.set_line(strategy.name)
            runner.run_evolution_strategies(generations=1000, num_children=num_children, max_age=5, strategy=strategy)
    runner.fitness_plots.show()


def experiment3():
    """
    Shows several plots for different `max_age` and `strategy`
    """
    circles = 10
    runner = CirclesInASquare(circles, plot_sols=False)
    for strategy in [Strategy.SINGLE_VARIANCE, Strategy.MULTIPLE_VARIANCE, Strategy.FULL_VARIANCE]:
        runner.fitness_plots.set_subplot(strategy.name)
        for max_age in [0, 1, 5, 1000]:
            runner.fitness_plots.set_line(f"Max Age = {max_age}")
            runner.run_evolution_strategies(generations=1000, num_children=2, max_age=max_age, strategy=strategy)
    runner.fitness_plots.show()


def experiment4():
    """
    Shows several plots for different `strategy` and `population_size`
    """
    circles = 10
    runner = CirclesInASquare(circles, plot_sols=False)
    for strategy in [Strategy.SINGLE_VARIANCE, Strategy.MULTIPLE_VARIANCE, Strategy.FULL_VARIANCE]:
        runner.fitness_plots.set_subplot(strategy.name)
        for population_size in [10, 30, 60, 100]:
            runner.fitness_plots.set_line(f"Population size = {population_size}")
            runner.run_evolution_strategies(generations=1000, num_children=1, max_age=1000, population_size=population_size,
                                            strategy=strategy)
    runner.fitness_plots.show()



def experiment5():
    """
    Shows plot for a non random initialization
    """
    circles = 10
    runner = CirclesInASquare(circles, plot_sols=False, number_of_runs=10)
    for strategy in [Strategy.SINGLE_VARIANCE, Strategy.MULTIPLE_VARIANCE, Strategy.FULL_VARIANCE]:
        runner.fitness_plots.set_subplot(strategy.name)
        for population_size in [10, 30, 60, 100]:
            runner.fitness_plots.set_line(f"Population size = {population_size}")
            runner.run_evolution_strategies(generations=1000, num_children=1, max_age=1000, population_size=population_size,
                                            strategy=strategy, use_warm_start = True)
    runner.fitness_plots.show()


def experiment6():
    """
    Shows 2 plots comparing random initialization vs warm start initialization
    """
    circles = 9
    runner = CirclesInASquare(circles, plot_sols=False, save_sols=True, number_of_runs=10)
    for use_warm_start in [False, True]:
        if use_warm_start:
            runner.fitness_plots.set_subplot(f"Warm Start")
        else:
            runner.fitness_plots.set_subplot(f"Random Initialization")
        for population_size in [50]:
            runner.fitness_plots.set_line(f"Population Size = {population_size}")
            runner.run_evolution_strategies(generations=1000000, num_children=4, max_age=1000000, population_size=population_size,
                                            strategy=Strategy.SINGLE_VARIANCE, use_warm_start= use_warm_start)
    runner.fitness_plots.show()


def experiment7():
    """
    Shows 3 plots comparing random initialization and warm start initialization with several mutation strategies
    """
    circles = 9
    runner = CirclesInASquare(circles, plot_sols=False, save_sols=True, number_of_runs=10)
    for strategy in [Strategy.SINGLE_VARIANCE, Strategy.MULTIPLE_VARIANCE, Strategy.FULL_VARIANCE]:
        runner.fitness_plots.set_subplot(strategy.name)
        for use_warm_start in [False, True]:
            if use_warm_start:
                runner.fitness_plots.set_line(f"Warm Start")
            else:
                runner.fitness_plots.set_line(f"Random Initialization")
            runner.run_evolution_strategies(generations=1000000, num_children=4, max_age=1000000, population_size=100,
                                            strategy=strategy, use_warm_start= use_warm_start)
    runner.fitness_plots.show()

def experiment8():
    """
    Compares different parameters for forces
    """
    circles = 10
    runner = CirclesInASquare(circles, plot_sols=False, save_sols=True, number_of_runs=10)
    for force_strength in [0.01, 0.0001, 0]:
        runner.fitness_plots.set_subplot(f"Force Strength = {str(force_strength)}")
        for use_warm_start in [False, True]:
            if use_warm_start:
                runner.fitness_plots.set_line(f"Warm Start")
            else:
                runner.fitness_plots.set_line(f"Random Initialization")
            runner.run_evolution_strategies(generations=100, num_children=2, max_age=1000000, population_size=50,
                                            strategy=Strategy.SINGLE_VARIANCE, force_strength=force_strength,
                                            use_warm_start= use_warm_start)
    runner.fitness_plots.show()


if __name__ == "__main__":
    # NOTE: locally create an empty "results" folder in the root of the repo
    experiment8()
