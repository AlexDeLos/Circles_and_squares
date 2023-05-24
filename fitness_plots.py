from math import ceil
import matplotlib.pyplot as plt
from evopy import ProgressReport
from enum import Enum
class TrackableVariable(Enum):
    GENERATIONS = 1,
    EVALUATIONS = 2
    BEST_FITNESS = 3,
    AVG_FITNESS = 4,
    STD_FITNESS = 5,
    TIME_ELAPSED = 6

class FitnessPlots:
    """
    Util class to quickly set up different types of experiments
    """

    def __init__(self):
        self.current_subplot = "Default"
        self.current_line = "Default"
        self.xs = {self.current_subplot: {
            self.current_line: []
        }}
        self.ys = {self.current_subplot: {
            self.current_line: []
        }}
        self.xvars = {self.current_subplot: TrackableVariable.GENERATIONS}
        self.yvars = {self.current_subplot: TrackableVariable.BEST_FITNESS}

    def set_subplot(self, name,
                    x_variable: TrackableVariable = TrackableVariable.GENERATIONS,
                    y_variable: TrackableVariable = TrackableVariable.BEST_FITNESS):
        """
        Set the subplot
        :param name: name of the subplot
        :param x_variable: the x variable to track for this subplot
        :param y_variable: the y variable to track for this subplot
        """
        if not name in self.ys.keys():
            self.xs[name] = {self.current_line: []}
            self.ys[name] = {self.current_line: []}
        self.xvars[name] = x_variable
        self.yvars[name] = y_variable
        self.current_subplot = name

    def set_line(self, name):
        """
        Set the line to add points to
        :param name: name of the line
        """
        if not name in self.ys[self.current_subplot].keys():
            self.xs[self.current_subplot][name] = []
            self.ys[self.current_subplot][name] = []
        self.current_line = name

    def add(self, report: ProgressReport):
        """
        Adds a new data point to the current line of the current subplot
        :param report: statistics of the point that needs to be set
        """
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
            else:
                raise Exception("Invalid TrackableVariable")
            lst[self.current_subplot][self.current_line].append(value)

    def clean(self):
        """
        Deletes empty lines and subplots
        """
        for subplot_name in set(self.ys.keys()):
            for line_name in set(self.ys[subplot_name].keys()):
                if len(self.ys[subplot_name][line_name]) == 0:
                    del self.xs[subplot_name][line_name]
                    del self.ys[subplot_name][line_name]
            if len(self.ys[subplot_name]) == 0:
                del self.xs[subplot_name]
                del self.ys[subplot_name]
                del self.xvars[subplot_name]
                del self.yvars[subplot_name]
    def show(self):
        """
        Shows all subplots in a square layout
        """
        self.clean()
        rows = int(len(self.ys) ** (1 / 2))
        columns = ceil(len(self.ys) / rows)
        fig, ax = plt.subplots(rows, columns)
        rainbow = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] #[next(subax._get_lines.prop_cycler)['color'] for _ in range(20)]
        rainbow_index = 0
        colors = dict()
        for i, subplot_name in enumerate(self.ys.keys()):
            if rows == 1 and columns == 1:
                subax = ax
            elif rows == 1:
                subax = ax[i]
            else:
                subax = ax[i%rows, i//rows]
            subax.set_xlabel("Generation")
            subax.set_ylabel("Fitness")
            subax.set_title(subplot_name)
            for line_name in self.ys[subplot_name].keys():
                if not line_name in colors:
                    colors[line_name] = rainbow[rainbow_index]
                    rainbow_index = (rainbow_index+1)%len(rainbow)
                if len(self.ys[subplot_name][line_name]) > 0:
                    subax.plot(self.xs[subplot_name][line_name], self.ys[subplot_name][line_name], label=line_name, color=colors[line_name])
                    subax.legend() #legend per subplot
        #plt.legend() #1 legend for all subplots
        plt.show()