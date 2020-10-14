import numpy as np


class GA:
    """Simple Genetic Algorithm"""

    def __init__(
        self,
        population,
        selection,
        crossover,
        mutation,
        # fun_fitness=lambda x: x ** 2
        fun_fitness=lambda x: np.arctan(-x) + np.pi,
    ):
        """
        fun_fitness: fitness based on objective values. minimize the objective by default
        """
        self.population = population
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.fun_fitness = fun_fitness

    def run(self, fun_evaluation, gen=50):
        """
        solve the problem based on Simple GA process
        """

        # initialize population
        self.population.initialize()

        # solving process
        for i in range(0, gen):
            # select
            fitness, _ = self.population.fitness(fun_evaluation, self.fun_fitness)
            self.selection.select(self.population, fitness)

            # crossover
            self.crossover.cross(self.population)

            # mutation
            self.mutation.mutate(self.population, np.random.rand())

        # return the best individual
        return self.population.best(fun_evaluation, self.fun_fitness)