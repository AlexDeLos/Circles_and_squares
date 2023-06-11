import matplotlib
# import random

from fitness_plots import FitnessPlots, TrackableVariable
from forces import ForcesConfig

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
    def __init__(self, n_circles, output_statistics=False, plot_sols=False, print_sols=False, save_sols=False,
                 number_of_runs=1):
        self.evopy = None
        self.print_sols = print_sols
        self.output_statistics = output_statistics
        self.plot_best_sol = plot_sols
        self.save_sols = save_sols
        self.n_circles = n_circles
        self.forces_config = None
        self.fig = None
        self.ax = None
        assert 2 <= n_circles <= 20

        # if self.plot_best_sol or self.save_sols:
        #     self.set_up_plot()

        if self.output_statistics:
            self.statistics_header()

        self.number_of_runs = number_of_runs
        self.fitness_plots = FitnessPlots(number_of_runs, hline=self.get_target())

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
                                                                                    report.best_fitness,
                                                                                    report.avg_fitness,
                                                                                    report.std_fitness)

            if self.print_sols:
                output += " ({:s})".format(np.array2string(report.best_genotype))
            print(output)

        if self.plot_best_sol or self.save_sols:
            points = np.reshape(report.best_genotype, (-1, 2))
            self.ax.clear()
            self.ax.scatter(points[:, 0], points[:, 1], clip_on=False, color="black")
            if not self.forces_config is None:
                report.best_individual.plot_distribution_mean()
                # plot_forces(points, strength=report.best_individual.force_strength)
            self.ax.set_xlim((0, 1))
            self.ax.set_ylim((0, 1))
            self.ax.set_title(f"Best solution in generation {report.generation} (Fitness {report.best_fitness})")
            if self.plot_best_sol:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            if self.save_sols and (report.generation == 0 or report.is_final_report):
                plt.savefig(
                    f"results/{self.fitness_plots.get_current_state().replace('.', '').replace('/', '')}@Generation{report.generation}")

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
                                 population_size=30, max_evaluations=1e5, use_warm_start=True, forces_config=None,
                                 mutation_rate=1):
        if self.plot_best_sol or self.save_sols:
            # quick and dirty way to reset the plot when the same runner is reused with new parameters
            self.set_up_plot()

        callback = self.statistics_callback

        best_solutions = []

        self.forces_config = forces_config

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
                target_fitness_value=self.get_target() + 100,
                # Don't stop at the target fitness_value, this ruins the plot
                max_evaluations=max_evaluations,
                num_children=num_children,
                max_age=max_age,
                strategy=strategy,
                warm_start=self.getWarmStart(population_size) if use_warm_start else None,
                forces_config=self.forces_config,
                mutation_rate=mutation_rate
            )

            best_solutions.append(self.evopy.run())

        if self.plot_best_sol or self.save_sols:
            plt.close()

        if len(best_solutions) == 1:
            return best_solutions[0]
        return best_solutions

    def toBaseN(self, x: int, base: int):
        newNumber = []
        while not x == 0:
            newD = x % base
            newNumber.insert(0, newD)
            x = int((x - newD) / base)
        return newNumber

    def get_warm_start(self, population_size):
        """
        Returns a set of positions in cordinates
        """
        return np.asarray([
            self.get_warm_start_individual() if self.n_circles < 6 else self.get_warm_start_individual_honey_comb()
            for _ in range(population_size)
        ])

    def get_warm_start_individual(self):
        n = int(self.n_circles ** (1 / 2))  # how many we can fit in each grid cleanly
        result = []
        fun = lambda x: int(x) / (n - 1)
        for el in range(n ** 2):
            i = self.toBaseN(el, n)  # turn it to base whatever
            while len(i) < 2:
                i.insert(0, '0')
            result.append(list(map(fun, i)))

        # fill in the rest:
        while len(result) < self.n_circles:
            result.append([np.random.uniform(), np.random.uniform()])

        # transform to the format they should be
        result = sum(result, [])

        # add random noise to it
        result = result + np.random.normal(loc=0, scale=1 / (10 * (n - 1)), size=self.n_circles * 2)
        result = np.clip(result, 0, 1)

        return result

    def get_warm_start_individual_honey_comb(self):
        """
        Function to create a warm start for an individual following the honeycomb pattern.
        """
        num_columns = math.floor(self.n_circles ** 0.5)
        # The `ceil` here ensures there is always enough room for all the points,
        # sometimes there is room leftover though.
        # TODO: Think about if this is preferred over making less honeycombs and randomly adding leftover points.
        num_rows = math.ceil(self.n_circles / num_columns)

        margin_x = 1 / (num_columns - 0.5)
        margin_y = 1 / (num_rows - 1)

        result = []
        for i in range(self.n_circles):
            # The row the new point will be placed
            row = int(len(result) / num_columns)

            # The coordinates of the new point
            y = 1 - row * margin_y
            x = (i % num_columns) * margin_x
            # If we are on an uneven row, add a bit of margin
            if (row % 2) == 1:
                x += 0.5 * margin_x

            result.append([x, y])

        result = sum(result, [])
        result += np.random.normal(loc=0, scale=0.1 * margin_x, size=self.n_circles * 2)
        result = np.clip(result, 0, 1)
        return result


def main():
    """
    Original main function
    """
    circles = 10
    runner = CirclesInASquare(circles, plot_sols=True, output_statistics=True)

    runner.run_evolution_strategies(generations=1000000, num_children=3,
                                    max_age=1000000, population_size=100,
                                    strategy=Strategy.SINGLE_VARIANCE,
                                    forces_config=ForcesConfig(force_strength=0.01,
                                                               number_of_neighbours=1),
                                    use_warm_start=True, mutation_rate=0.1)


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
            runner.run_evolution_strategies(generations=1000, num_children=1, max_age=1000,
                                            population_size=population_size,
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
            runner.run_evolution_strategies(generations=1000, num_children=1, max_age=1000,
                                            population_size=population_size,
                                            strategy=strategy, use_warm_start=True)
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
            runner.run_evolution_strategies(generations=1000000, num_children=4, max_age=1000000,
                                            population_size=population_size,
                                            strategy=Strategy.SINGLE_VARIANCE, use_warm_start=use_warm_start)
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
                                            strategy=strategy, use_warm_start=use_warm_start)
    runner.fitness_plots.show()


def experiment8():
    """
    Compares different parameters for forces
    """
    circles = 10
    runner = CirclesInASquare(circles, plot_sols=False, save_sols=False, number_of_runs=10)
    for force_strength in [0.01, 1, 0]:
        runner.fitness_plots.set_subplot(f"Force Strength = {str(force_strength)}")
        for use_warm_start in [True]:
            if use_warm_start:
                runner.fitness_plots.set_line(f"Warm Start")
            else:
                runner.fitness_plots.set_line(f"Random Initialization")
            runner.run_evolution_strategies(generations=100, num_children=2, max_age=1000000, population_size=50,
                                            strategy=Strategy.SINGLE_VARIANCE,
                                            forces_config=ForcesConfig(force_strength=0.01),
                                            use_warm_start=use_warm_start)
    runner.fitness_plots.show()


def experiment9():
    """
    Comparing different mutation rates for different population sizes
    """
    circles = 10
    runner = CirclesInASquare(circles, plot_sols=False, save_sols=False, number_of_runs=10)
    for population_size in [25, 50, 75]:
        runner.fitness_plots.set_subplot(f"Population Size = {str(population_size)}")
        for mutation_rate in [0, 0.1, 0.3, 1]:
            runner.fitness_plots.set_line(f"mutation_rate = {str(mutation_rate)}")
            runner.run_evolution_strategies(generations=100, num_children=2, max_age=1000000,
                                            population_size=population_size,
                                            strategy=Strategy.SINGLE_VARIANCE,
                                            forces_config=ForcesConfig(force_strength=0.01),
                                            use_warm_start=True, mutation_rate=mutation_rate)
    runner.fitness_plots.show()


def experiment10():
    """
    Comparing Forces vs No Forces
    """
    circles = 10
    runner = CirclesInASquare(circles, plot_sols=False, save_sols=False, number_of_runs=10)
    for force_strength in [1, 0.1, 0.01, 0.001]:
        runner.fitness_plots.set_subplot(f"initial_force_strength = {str(force_strength)}")
        for forces_mutation_rate in [0, 0.1, 0.3, 0.7, 1]:
            runner.fitness_plots.set_line(f"forces_mutation_rate = {str(forces_mutation_rate)}")
            runner.run_evolution_strategies(generations=1000000, num_children=2, max_age=1000000, population_size=75,
                                            strategy=Strategy.SINGLE_VARIANCE,
                                            forces_config=ForcesConfig(force_strength=force_strength,
                                                                       mutation_rate=forces_mutation_rate,
                                                                       number_of_neighbours=1),
                                            use_warm_start=True, mutation_rate=1)
    runner.fitness_plots.show()


def experiment11():
    """
    New experiment
    """
    circles = 10
    runner = CirclesInASquare(circles, plot_sols=False, save_sols=True, number_of_runs=10)
    for force_strength in [0.1, 0.01, 0.001, 0]:
        runner.fitness_plots.set_subplot(f"initial_force_strength = {str(force_strength)}")
        for forces_mutation_rate in [0, 1]:
            runner.fitness_plots.set_line(f"forces_mutation_rate = {str(forces_mutation_rate)}")
            runner.run_evolution_strategies(generations=1000000, num_children=2, max_age=1000000, population_size=75,
                                            strategy=Strategy.SINGLE_VARIANCE,
                                            forces_config=ForcesConfig(force_strength=force_strength,
                                                                       mutation_rate=forces_mutation_rate,
                                                                       number_of_neighbours=1),
                                            use_warm_start=True, mutation_rate=1)
    runner.fitness_plots.show()


def experiment12():
    """
    Shows several plots for different `num_children` and `population_size`
    """
    circles = 10
    runner = CirclesInASquare(circles, plot_sols=False)
    for population_size in [25, 50, 75]:
        runner.fitness_plots.set_subplot(population_size)
        for num_children in [1, 2, 3, 4]:
            runner.fitness_plots.set_line(f"Number of children = {num_children}")
            runner.run_evolution_strategies(generations=1000, num_children=num_children, max_age=1000,
                                            population_size=population_size,
                                            strategy=Strategy.SINGLE_VARIANCE)
    runner.fitness_plots.show()


def experiment13():
    """
    Shows several plots for different `num_children` and `population_size`
    """
    circles = 10
    runner = CirclesInASquare(circles, plot_sols=False)
    for num_children in [1, 2, 3, 4]:
        runner.fitness_plots.set_subplot(num_children)
        for population_size in [25, 50, 75]:
            runner.fitness_plots.set_line(f"Population Size = {population_size}")
            runner.run_evolution_strategies(generations=100000, num_children=num_children, max_age=100000,
                                            population_size=population_size,
                                            strategy=Strategy.SINGLE_VARIANCE)
    runner.fitness_plots.show()

def experiment14():
#Create heatmap for different population sizes and number of children
    circles = 10   
    runner = CirclesInASquare(circles, plot_sols=False)
    for population_size in [25,50,75]:
        for num_children in [1,2,3,4]:
            runner.fitness_plots.set_subplot(f"Population Size = {population_size}, Number of Children = {num_children}")
            runner.run_evolution_strategies(generations=1000, num_children=num_children, max_age=1000, population_size=population_size,
                                            strategy=Strategy.SINGLE_VARIANCE)
    runner.fitness_plots.show()



def experiment15():
    """
    Shows a heatmap for different `num_children` and `population_size` with fitness as the metric
    """
    circles = 10
    fitness_data = np.zeros((10, 10))  # Create an empty array to store fitness values

    runner = CirclesInASquare(circles, plot_sols=False)

    # Iterate over different `num_children` values
    for i, num_children in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
        # Iterate over different `population_size` values
        for j, population_size in enumerate([20, 30, 40, 50, 60, 70, 80, 90, 100, 110]):
            # Run evolution strategies
            fitness = runner.run_evolution_strategies(generations=1000, num_children=num_children, max_age=1000,
                                            population_size=population_size, strategy=Strategy.SINGLE_VARIANCE)

            # Store fitness value in the data array
            fitness_data[i, j] = circles_in_a_square(fitness)

    # Create a heatmap using the fitness data
    fig, ax = plt.subplots()
    im = ax.imshow(fitness_data, cmap='hot', interpolation='nearest')

    # Set x-axis and y-axis labels
    ax.set_xticks(np.arange(len([20, 30, 40, 50, 60, 70, 80, 90, 100, 110])))
    ax.set_yticks(np.arange(len([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))
    ax.set_xticklabels([20, 30, 40, 50, 60, 70, 80, 90, 100, 110])
    ax.set_yticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # Rotate the x-axis tick labels and set label positions
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    for i in range(len([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])):
        for j in range(len([20, 30, 40, 50, 60, 70, 80, 90, 100, 110])):
            text = ax.text(j, i, round(fitness_data[i, j], 5), ha="center", va="center", color="w")

    # Set title and colorbar
    ax.set_title("Fitness Heatmap")
    plt.colorbar(im)

    plt.show()








def experiment14():
    """
    `mutation_rate` on various n_circles`
    """
    circles = 2
    runner = CirclesInASquare(circles, plot_sols=False, save_sols=False, number_of_runs=10)
    for n_circles in [11, 17]:
        runner.n_circles = n_circles
        runner.fitness_plots.set_subplot(f"Number Of Circles = {str(n_circles)}")
        for mutation_rate in [0, 0.1, 0.3, 1]:
            runner.fitness_plots.set_line(f"mutation_rate = {str(mutation_rate)}")
            runner.run_evolution_strategies(generations=100, num_children=2, max_age=1000000, population_size=75,
                                            strategy=Strategy.SINGLE_VARIANCE,
                                            forces_config=ForcesConfig(force_strength=0.01, mutation_rate=1),
                                            use_warm_start=True, mutation_rate=mutation_rate)
    runner.fitness_plots.show()


def experiment15():
    """
    Tuning `force_epsilon`
    Conclusion: doesn't really matter
    """
    circles = 10
    runner = CirclesInASquare(circles, plot_sols=False, save_sols=False, number_of_runs=10)
    for n_circles in [10]:
        runner.n_circles = n_circles
        runner.fitness_plots.set_subplot(f"Number Of Circles = {str(n_circles)}")
        for force_epsilon in [0.001, 0.00001, 0.0000001, 0.000000001]:
            runner.fitness_plots.set_line(f"force_epsilon = {str(force_epsilon)}")
            runner.run_evolution_strategies(generations=100, num_children=2, max_age=1000000, population_size=75,
                                            strategy=Strategy.SINGLE_VARIANCE,
                                            forces_config=ForcesConfig(force_strength=0.01, mutation_rate=1,
                                                                       force_epsilon=force_epsilon),
                                            use_warm_start=True, mutation_rate=0.1)
    runner.fitness_plots.show()


def experiment16():
    """
    [Single force scales] vs [Multiple force scales] vs [No forces]
    """
    circles = 10
    runner = CirclesInASquare(circles, plot_sols=False, save_sols=True, number_of_runs=20)
    for n_circles in [10]:
        runner.n_circles = n_circles
        runner.fitness_plots.set_subplot(f"Number Of Circles = {str(n_circles)}")
        for name, strategy in [("Single Force Scale", ForcesConfig.Strategy.SINGLE_FORCE_SCALES),
                               ("Multiple Force Scales", ForcesConfig.Strategy.MULTIPLE_FORCE_SCALES),
                               ("Single Variance Mutation (No Forces)", None)]:
            runner.fitness_plots.set_line(name)
            if strategy is None:
                runner.run_evolution_strategies(generations=100000, num_children=3, max_age=100000, population_size=75,
                                                strategy=Strategy.SINGLE_VARIANCE,
                                                use_warm_start=True, mutation_rate=0.9)
            else:
                runner.run_evolution_strategies(generations=100000, num_children=3, max_age=100000, population_size=75,
                                                strategy=Strategy.SINGLE_VARIANCE,
                                                forces_config=ForcesConfig(force_strength=0.01, mutation_rate=0.9,
                                                                           strategy=strategy),
                                                use_warm_start=True, mutation_rate=0.1)
    runner.fitness_plots.show()


def plot_warm_start_solution():
    """
    Helper function used to test warm start functions.
    """
    runner = CirclesInASquare(12)
    population_size = 1
    individual = runner.get_warm_start(population_size)[0]
    x0, x1 = np.reshape(individual, (-1, 2)).transpose()
    plt.scatter(x0, x1, color='black')
    plt.show()


if __name__ == "__main__":
    # NOTE: locally create an empty "results" folder in the root of the repo
<<<<<<< HEAD
    experiment15()
    #fitness_plots_from_backup(100)
    # main()
=======
    # experiment6()
    plot_warm_start_solution()
>>>>>>> 4e02a44515d3c3ec71b9e5345ee967d0daa0c593
