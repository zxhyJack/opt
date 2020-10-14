import numpy as np
import copy
from GAComponents import Individual

# ----------------------------------------------------------
# SELECTION
# ----------------------------------------------------------
class Selection:
    """base class for selection"""

    def select(self, population, fitness):
        raise NotImplementedError


class RouletteWheelSelection(Selection):
    """
    select individuals by Roulette Wheel:
    individuals selected with a probability of its fitness
    """

    def select(self, population, fitness):
        selected_individuals = np.random.choice(
            population.individuals, population.size, p=fitness
        )
        # pay attention to deep copy these objects
        population.individuals = np.array(
            [copy.deepcopy(I) for I in selected_individuals]
        )


class Crossover:
    def __init__(self, rate=0.8, alpha=0.5):
        """
        crossover operation:
            rate: propability of crossover.
            alpha: factor for crossing two chroms, [0,1]
        """
        # parameters check is skipped
        self.rate = rate
        self.alpha = alpha

    @staticmethod
    def cross_individuals(individual_a, individual_b, alpha):
        """
        generate two child individuals based on parent individuals:
        new values are calculated at random positions
        alpha: linear ratio to cross two genes, exchange two genes if alpha is 0.0
        """
        # random positions to be crossed
        pos = np.random.rand(individual_a.dimension) <= 0.5

        # cross value
        temp = (individual_b.solution - individual_a.solution) * pos * (1 - alpha)
        new_value_a = individual_a.solution + temp
        new_value_b = individual_b.solution - temp

        # return new individuals
        new_individual_a = Individual(individual_a.ranges)
        new_individual_b = Individual(individual_b.ranges)

        new_individual_a.solution = new_value_a
        new_individual_b.solution = new_value_b

        return new_individual_a, new_individual_b

    def cross(self, population):

        new_individuals = []
        random_population = np.random.permutation(
            population.individuals
        )  # random order
        num = int(population.size / 2.0) + 1

        for individual_a, individual_b in zip(
            population.individuals[0 : num + 1], random_population[0 : num + 1]
        ):
            # crossover
            if np.random.rand() <= self.rate:
                child_individuals = self.cross_individuals(
                    individual_a, individual_b, self.alpha
                )
                new_individuals.extend(child_individuals)
            else:
                new_individuals.append(individual_a)
                new_individuals.append(individual_b)

        population.individuals = np.array(new_individuals[0 : population.size + 1])


class Mutation:
    def __init__(self, rate):
        """
        mutation operation:
        rate: propability of mutation, [0,1]
        """
        self.rate = rate

    def mutate_individual(self, individual, positions, alpha):
        """
        positions: mutating gene positions, list
        alpha: mutatation magnitude
        """
        for pos in positions:
            if np.random.rand() < 0.5:
                individual.solution[pos] -= (
                    individual.solution[pos] - individual.ranges[:, 0][pos]
                ) * alpha
            else:
                individual.solution[pos] += (
                    individual.ranges[:, 1][pos] - individual.solution[pos]
                ) * alpha

        # reset evaluation
        individual.evaluation = None
        individual.fitness = None

    def mutate(self, population, alpha):
        """
        alpha: mutating magnitude
        """
        for individual in population.individuals:
            if np.random.rand() > self.rate:
                continue

            # select random positions to mutate
            num = np.random.randint(individual.dimension) + 1
            pos = np.random.choice(individual.dimension, num, replace=False)
            self.mutate_individual(individual, pos, alpha)
