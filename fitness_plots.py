from math import ceil
import matplotlib.pyplot as plt

class FitnessPlots:
    """
    A class that creates plots of the fitness over the generations
    """
    def __init__(self):
        self.current_subplot = "Default"
        self.current_line = "Default"
        self.generations = {self.current_subplot: {
            self.current_line: []
        }}
        self.fitnessess = {self.current_subplot: {
            self.current_line: []
        }}
        self.min_fitness = float("inf")
        self.max_fitness = -float("inf")

    def set_subplot(self, name):
        """
        Set the subplot to add points to
        :param name: name of the subplot
        """
        if not name in self.fitnessess.keys():
            self.generations[name] = {self.current_line: []}
            self.fitnessess[name] = {self.current_line: []}
        self.current_subplot = name

    def set_line(self, name):
        """
        Set the line to add points to
        :param name: name of the line
        """
        if not name in self.fitnessess[self.current_subplot].keys():
            self.generations[self.current_subplot][name] = []
            self.fitnessess[self.current_subplot][name] = []
        self.current_line = name

    def add(self, generation, fitness):
        """
        Adds a new point to the current line of the current subplot
        :param generation: current generation
        :param fitness: best fitness of this generation
        """
        self.generations[self.current_subplot][self.current_line].append(generation)
        self.fitnessess[self.current_subplot][self.current_line].append(fitness)

        self.max_fitness = max(self.max_fitness, fitness)
        self.min_fitness = min(self.min_fitness, fitness)

    def clean(self):
        """
        deletes empty lines and subplots
        """
        for subplot_name in set(self.fitnessess.keys()):
            for line_name in set(self.fitnessess[subplot_name].keys()):
                if len(self.fitnessess[subplot_name][line_name]) == 0:
                    del self.generations[subplot_name][line_name]
                    del self.fitnessess[subplot_name][line_name]
            if len(self.fitnessess[subplot_name]) == 0:
                del self.generations[subplot_name]
                del self.fitnessess[subplot_name]
    def show(self):
        self.clean()
        rows = int(len(self.fitnessess)**(1/2))
        columns = ceil(len(self.fitnessess)/rows)
        fig, ax = plt.subplots(rows, columns)
        for i, subplot_name in enumerate(self.fitnessess.keys()):
            if rows == 1 and columns == 1:
                subax = ax
            elif rows == 1:
                subax = ax[i]
            else:
                subax = ax[i%rows, i//rows]
            subax.set_xlabel("Generation")
            subax.set_ylabel("Fitness")
            subax.set_title(subplot_name)
            for line_name in self.fitnessess[subplot_name].keys():
                if len(self.fitnessess[subplot_name][line_name]) > 0:
                    subax.plot(self.generations[subplot_name][line_name], self.fitnessess[subplot_name][line_name], label=line_name)

        plt.setp(ax,ylim=[self.min_fitness, self.max_fitness])
        plt.show()