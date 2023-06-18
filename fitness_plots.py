import pickle
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
from evopy import ProgressReport
from enum import Enum

class TrackableVariable(Enum):
    GENERATIONS = "Generations",
    EVALUATIONS = "Evalutions",
    BEST_FITNESS = "Fitness",
    AVG_FITNESS = "Avg Fitness",
    STD_FITNESS = "Std Fitness",
    TIME_ELAPSED = "Time Elapsed",
    NUMBER_OF_CIRCLES = "Circles"

class FitnessPlots:
    """
    Util class to quickly set up different types of experiments
    """
    BACKUP_FILENAME = "fitness_plots.backup"

    def __init__(self, number_of_runs):
        self.current_index = 0
        self.current_run = 0
        self.number_of_runs = number_of_runs
        self.hline = dict()

        self.current_subplot = "Default"
        self.current_line = "Default"
        self.xs = {self.current_subplot: {
            self.current_line: np.zeros((1, self.number_of_runs))
        }}
        self.ys = {self.current_subplot: {
            self.current_line: np.zeros((1, self.number_of_runs))
        }}
        self.xvars = {self.current_subplot: TrackableVariable.GENERATIONS}
        self.yvars = {self.current_subplot: TrackableVariable.BEST_FITNESS}

        self.ys_mean = dict()
        self.ys_std = dict()
        self.xs_mean = dict()
        self.xs_std = dict()

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def from_backup():
        with open(FitnessPlots.BACKUP_FILENAME, "rb") as f:
            fitness_plots = pickle.load(f)
        return fitness_plots

    def get_current_state(self):
        return f"{self.current_subplot}@{self.current_line}@Run{self.current_run}"

    def print_current_state(self):
        print(30*"-")
        print(f"Run {self.current_run+1}/{self.number_of_runs}:")
        print(self.current_subplot)
        print(self.current_line)

    def is_tracking_circles(self):
        return (self.xvars[self.current_subplot] == TrackableVariable.NUMBER_OF_CIRCLES or
                self.yvars[self.current_subplot] == TrackableVariable.NUMBER_OF_CIRCLES)

    def set_run(self, run, force_reset_index=True):
        if not self.is_tracking_circles():
            self.current_index = 0
        self.current_run = run
        self.print_current_state()

    def set_subplot(self, name,
                    x_variable: TrackableVariable = TrackableVariable.EVALUATIONS,
                    y_variable: TrackableVariable = TrackableVariable.BEST_FITNESS,
                    hline=None):
        """
        Set the subplot
        :param name: name of the subplots
        :param x_variable: the x variable to track for this subplot
        :param y_variable: the y variable to track for this subplot
        :param hline: (optional) horizontal value for this subplot
        """
        if not name in self.ys.keys():
            self.xs[name] = {self.current_line: np.zeros((1, self.number_of_runs))}
            self.ys[name] = {self.current_line: np.zeros((1, self.number_of_runs))}
        self.xvars[name] = x_variable
        self.yvars[name] = y_variable
        self.hline[name] = hline
        self.current_subplot = name

    def set_line(self, name):
        """
        Set the line to add points to
        :param name: name of the line
        """
        self.current_index = 0
        if not name in self.ys[self.current_subplot].keys():
            self.xs[self.current_subplot][name] = np.zeros((1, self.number_of_runs))
            self.ys[self.current_subplot][name] = np.zeros((1, self.number_of_runs))
        self.current_line = name

    def add(self, report: ProgressReport):
        """
        Adds a new data point to the current line of the current subplot
        :param report: statistics of the point that needs to be set
        """

        if self.is_tracking_circles() and report.is_final_report == False:
            return

        for lst, var in [(self.xs, self.xvars), (self.ys, self.yvars)]:
            if var[self.current_subplot] == TrackableVariable.GENERATIONS:
                value = report.generation
            elif var[self.current_subplot] == TrackableVariable.EVALUATIONS:
                value = report.evaluations
            elif var[self.current_subplot] == TrackableVariable.BEST_FITNESS:
                value = report.best_fitness
            elif var[self.current_subplot] == TrackableVariable.AVG_FITNESS:
                value = report.avg_fitness
            elif var[self.current_subplot] == TrackableVariable.STD_FITNESS:
                value = report.std_fitness
            elif var[self.current_subplot] == TrackableVariable.TIME_ELAPSED:
                value = report.time_elapsed
            elif var[self.current_subplot] == TrackableVariable.NUMBER_OF_CIRCLES:
                value = len(report.best_individual.genotype)//2
            else:
                raise Exception("Invalid TrackableVariable")

            while lst[self.current_subplot][self.current_line].shape[0] <= self.current_index:
                lst[self.current_subplot][self.current_line] = np.append(lst[self.current_subplot][self.current_line], np.zeros((1, self.number_of_runs)), axis=0)
            lst[self.current_subplot][self.current_line][self.current_index, self.current_run] = value
        self.current_index += 1

    def clean(self):
        """
        Deletes empty lines and subplots
        """
        for subplot_name in set(self.ys.keys()):
            for line_name in set(self.ys[subplot_name].keys()):
                if self.ys[subplot_name][line_name].shape[0] <= 1:
                    del self.xs[subplot_name][line_name]
                    del self.ys[subplot_name][line_name]
            if len(self.ys[subplot_name]) == 0:
                del self.xs[subplot_name]
                del self.ys[subplot_name]
                del self.xvars[subplot_name]
                del self.yvars[subplot_name]

        if self.is_tracking_circles():
            for subplot_name in set(self.ys.keys()):
                for line_name in set(self.ys[subplot_name].keys()):

                    def ThomasForLoopFabriek(x):
                        res = []
                        for i in range(x.shape[1]):
                            res.append([])
                            for j in range(x.shape[0] // x.shape[1]):
                                res[i].append(x[i + j * x.shape[1], i])
                        return np.array(res).transpose()

                    self.xs[subplot_name][line_name] = ThomasForLoopFabriek(self.xs[subplot_name][line_name])
                    self.ys[subplot_name][line_name] = ThomasForLoopFabriek(self.ys[subplot_name][line_name])

                    # indices = np.where(self.xs[subplot_name][line_name].flatten() != 0)
                    #
                    # self.xs[subplot_name][line_name] = self.xs[subplot_name][line_name].flatten()[indices].reshape((-1, self.number_of_runs))
                    # self.ys[subplot_name][line_name] = self.ys[subplot_name][line_name].flatten()[indices].reshape((-1, self.number_of_runs))

    def calculate_mean_and_std(self):
        """
        calculates the mean and std of self.xs and self.ys
        """
        for subplot_name in set(self.ys.keys()):
            self.xs_mean[subplot_name] = dict()
            self.xs_std[subplot_name] = dict()
            self.ys_mean[subplot_name] = dict()
            self.ys_std[subplot_name] = dict()
            for line_name in set(self.ys[subplot_name].keys()):
                self.xs_mean[subplot_name][line_name] = np.mean(self.xs[subplot_name][line_name], axis=1)
                self.xs_std[subplot_name][line_name] = np.std(self.xs[subplot_name][line_name], axis=1)
                self.ys_mean[subplot_name][line_name] = np.mean(self.ys[subplot_name][line_name], axis=1)
                self.ys_std[subplot_name][line_name] = np.std(self.ys[subplot_name][line_name], axis=1)
                if np.any(self.xs_std[subplot_name][line_name] > 0):
                    print(f"WARNING: Deviation on x-axis detected for {subplot_name} and {line_name}")

    def set_number_of_error_bars(self, number_of_error_bars):
        """
        Thin out the places where the std is greater than 0
        """
        for subplot_name in set(self.ys_std.keys()):
            for line_name in set(self.ys_std[subplot_name].keys()):
                total_bars = self.ys_std[subplot_name][line_name].shape[0]
                interval = max(1, total_bars//number_of_error_bars)
                indices = np.where([n%interval != 0 for n in range(total_bars)])
                self.xs_std[subplot_name][line_name][indices] = 0
                self.ys_std[subplot_name][line_name][indices] = 0

    def show(self, filename=BACKUP_FILENAME, number_of_error_bars=float("inf"), shared_axis=False):
        """
        Shows all subplots in a square layout
        """
        self.clean()
        self.calculate_mean_and_std()
        self.set_number_of_error_bars(number_of_error_bars)
        rows = int(len(self.ys) ** (1 / 2))
        columns = ceil(len(self.ys) / rows)
        fig, ax = plt.subplots(rows, columns)
        rainbow = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] #[next(subax._get_lines.prop_cycler)['color'] for _ in range(20)]
        rainbow_index = 0
        colors = dict()
        y_min = float("inf")
        y_max = -float("inf")
        for i, subplot_name in enumerate(self.ys.keys()):
            if rows == 1 and columns == 1:
                subax = ax
            elif rows == 1:
                subax = ax[i]
            else:
                subax = ax[i%rows, i//rows]
            subax.set_xlabel(self.xvars[subplot_name].value[0])
            subax.set_ylabel(self.yvars[subplot_name].value[0])
            subax.set_title(subplot_name)
            if not self.hline[subplot_name] is None:
                subax.axhline(y=self.hline[subplot_name], color='black', linestyle='--', label="Optimum")
            for line_name in self.ys[subplot_name].keys():
                if not line_name in colors:
                    colors[line_name] = rainbow[rainbow_index]
                    rainbow_index = (rainbow_index+1)%len(rainbow)
                if len(self.ys[subplot_name][line_name]) > 0:
                    subax.plot(self.xs_mean[subplot_name][line_name], self.ys_mean[subplot_name][line_name],
                             label=line_name, color=colors[line_name])
                    subax.fill_between(x=self.xs_mean[subplot_name][line_name],
                                     y1=self.ys_mean[subplot_name][line_name]-self.ys_std[subplot_name][line_name],
                                     y2=self.ys_mean[subplot_name][line_name]+self.ys_std[subplot_name][line_name],
                                     facecolor=colors[line_name], alpha=0.3)
                    # Old version: errorbars
                    # subax.errorbar(self.xs_mean[subplot_name][line_name], self.ys_mean[subplot_name][line_name],
                    #                xerr = self.xs_std[subplot_name][line_name],
                    #                yerr = self.ys_std[subplot_name][line_name],
                    #                label=line_name, color=colors[line_name])
            subax.legend() #legend per subplot
            y_min = min(y_min, self.ys[subplot_name][line_name].min())
            y_max = max(y_max, self.ys[subplot_name][line_name].max(), self.hline[subplot_name]*1.001)
        #plt.legend() #1 legend for all subplots
        if shared_axis:
            plt.setp(ax, ylim=[y_min, y_max])
        plt.show()
        self.save(filename)

    def print_best_results(self):
        for subplot_name in self.ys.keys():
            for line_name in self.ys[subplot_name].keys():
                print("---------------")
                print("Best fitness report:")
                print(subplot_name)
                print(line_name)
                print(self.ys[subplot_name][line_name].max())