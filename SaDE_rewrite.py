import uuid
import math
import time
import os
import random
import statistics
from matplotlib import pyplot as plt


class SaDE:
    def __init__(
        self,
        NP,
        dim,
        bounds,
        func_fitness,
        max_iter=200,
        maximize=False,
    ):
        self.NP = NP
        self.dim = dim
        self.bounds = bounds
        self.fitness = func_fitness
        self.max_iter = max_iter
        self.maximize = maximize
        self.population = []
        self.fpopulation = []
        self.fbest_list = []

    def generatePopulation(self):
        for i in range(self.NP):
            lp = []
            for j in range(self.dim):
                lp.append(random.uniform(self.bounds[j][0], self.bounds[j][1]))
            self.population.append(lp)

    def evaluatePopulation(self):
        for i in self.population:
            self.fpopulation.append(self.fitness(i))

    def getBestSolution(self):
        best_fitness = self.fpopulation[0]
        best_solution = [x for x in self.population[0]]
        for i in range(1, len(self.population)):
            if self.maximize == True:
                if self.fpopulation[i] >= best_fitness:
                    best_fitness = float(self.fpopulation[i])
                    best_solution = [x for x in self.population[i]]
            else:
                if self.fpopulation[i] <= best_fitness:
                    best_fitness = float(self.fpopulation[i])
                    best_solution = [x for x in self.population[i]]
        return best_fitness, best_solution

    def rand_1_bin(self, individual, F, CR):
        p1 = individual
        while p1 == individual:
            p1 = random.choice(self.population)
        p2 = individual
        while p2 == individual or p2 == p1:
            p2 = random.choice(self.population)
        p3 = individual
        while p3 == individual or p3 == p1 or p3 == p2:
            p3 = random.choice(self.population)

        cutpoint = random.randint(0, self.dim - 1)
        candidateSolution = []

        for i in range(self.dim):
            if i == cutpoint or random.uniform(0, 1) < CR:
                candidateSolution.append(p3[i] + F * (p1[i] - p2[i]))
            else:
                candidateSolution.append(individual[i])

        return candidateSolution

    def currentToBest_2_bin(self, ind, best, wf, CR):
        p1 = ind
        while p1 == ind:
            p1 = random.choice(self.population)
        p2 = ind
        while p2 == ind or p2 == p1:
            p2 = random.choice(self.population)

        cutpoint = random.randint(0, self.dim - 1)
        candidateSolution = []

        for i in range(self.dim):
            if i == cutpoint or random.uniform(0, 1) < CR:
                candidateSolution.append(
                    ind[i] + wf * (best[i] - ind[i]) + wf * (p1[i] - p2[i])
                )
            else:
                candidateSolution.append(ind[i])

        return candidateSolution

    def boundsRes(self, individual):
        for i in range(len(individual)):
            if individual[i] < self.bounds[i][0]:
                individual[i] = self.bounds[i][0]
            if individual[i] > self.bounds[i][1]:
                individual[i] = self.bounds[i][1]

    def run(self):
        ns1 = 0
        ns2 = 0
        nf1 = 0
        nf2 = 0
        p1 = 0.5
        p2 = 0.5
        learningPeriod = 50
        crPeriod = 5
        crmUpdatePeriod = 25
        start = time.time()
        # start the algorithm
        best = []  # global best positions
        fbest = 0.00  # fitness of global best position
        result = []
        # global best fitness
        if self.maximize == True:
            fbest = 0.00
        else:
            fbest = math.inf

        # initial_generations
        self.generatePopulation()
        self.evaluatePopulation()
        fbest, best = self.getBestSolution()

        # evolution_step
        crm = 0.5  # initialize Crm
        # generates crossover rate values
        crossover_rate = [random.normalvariate(crm, 0.1) for i in range(self.NP)]
        cr_list = []  # save available crossover_rate
        for iter in range(self.max_iter):
            avrFit = 0.00
            strategy = 0
            for i in range(0, self.NP):
                # generate weight factor values
                F = random.normalvariate(0.5, 0.3)
                if random.uniform(0, 1) < p1:
                    candidateSolution = self.rand_1_bin(
                        self.population[i],
                        F,
                        crossover_rate[i],
                    )
                    strategy = 1
                else:
                    candidateSolution = self.currentToBest_2_bin(
                        self.population[i],
                        best,
                        F,
                        crossover_rate[i],
                    )
                    strategy = 2

                self.boundsRes(candidateSolution)
                fcandidateSolution = self.fitness(candidateSolution)

                if self.maximize == False:
                    if fcandidateSolution < self.fpopulation[i]:
                        self.population[i] = candidateSolution
                        self.fpopulation[i] = fcandidateSolution
                        cr_list.append(crossover_rate[i])  # save available CR
                        if strategy == 1:
                            ns1 += 1
                        elif strategy == 2:
                            ns2 += 1
                    else:
                        if strategy == 1:
                            nf1 += 1
                        elif strategy == 2:
                            nf2 += 1
                else:
                    if fcandidateSolution >= self.fpopulation[i]:
                        self.population[i] = candidateSolution
                        self.fpopulation[i] = fcandidateSolution
                        cr_list.append(crossover_rate[i])
                        if strategy == 1:
                            ns1 += 1
                        elif strategy == 2:
                            ns2 += 1
                    else:
                        if strategy == 1:
                            nf1 += 1
                        elif strategy == 2:
                            nf2 += 1
                avrFit += self.fpopulation[i]
            avrFit = avrFit / self.NP
            fbest, best = self.getBestSolution()
            result.append(
                {
                    "iter": iter,
                    "fbest": round(fbest, 3),
                    "avrFit": round(avrFit, 3),
                    "elapTime": round((time.time() - start) * 1000.0, 3),
                }
            )

            if iter % crPeriod == 0 and iter != 0:
                crossover_rate = [
                    random.normalvariate(crm, 0.1) for iter in range(self.NP)
                ]
                if iter % crmUpdatePeriod == 0:
                    crm = sum(cr_list) / len(cr_list)
                    cr_list = []

            if iter % learningPeriod == 0 and iter != 0:
                p1 = (ns1 * (ns2 + nf2)) / (ns2 * (ns1 + nf1) + ns1 * (ns2 + nf2))
                p2 = 1 - p1
                ns1 = 0
                ns2 = 0
                nf1 = 0
                nf2 = 0
        return result

if __name__ == "__main__":
    # fitness_function
    def func_fitness(X):
        result = 0.00
        for dim in X:
            result += (dim - 1) ** 2 - 10 * math.cos(2 * math.pi * (dim - 1))
        return 10 * len(X) + result

    dim = 4
    bounds = [(-5.12, 5.12)] * dim
    p = SaDE(
        NP=20,
        dim=4,
        bounds=bounds,
        func_fitness=func_fitness,
        max_iter=200,
    )
    p.run()
    for i in p.run(): print(i)
