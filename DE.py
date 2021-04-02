"""
Differential Evolution without numpy
"""
import random


class DE:
    def __init__(
        self, NP, D, max_iter, x_low_bound, x_up_bound, F, CR, calculate_fitness
    ):
        self.NP = NP
        self.D = D
        self.max_iter = max_iter
        self.CR = CR
        self.F = F
        self.x_low_bound = x_low_bound
        self.x_up_bound = x_up_bound
        self.cal_fitness = calculate_fitness
        self.X = [[0 for i in range(self.D)] for j in range(self.NP)]
        self.V = [[0 for i in range(self.D)] for j in range(self.NP)]
        self.U = [[0 for i in range(self.D)] for j in range(self.NP)]
        self.global_solution = [0 for i in range(self.D)]
        self.global_fitness = 0
        self.initial_population()

    def initial_population(self):
        for i in range(self.NP):
            for j in range(self.D):
                self.X[i][j] = (
                    self.x_low_bound
                    + (self.x_up_bound - self.x_low_bound) * random.random()
                )
        self.global_solution = self.X[0]
        self.global_fitness = self.cal_fitness(self.global_solution)

    def mutation(self):
        for i in range(self.NP):
            # r1 r2 r3 random and different
            r1 = random.randrange(self.NP)
            while r1 == i:
                r1 = random.randrange(self.NP)
            r2 = random.randrange(self.NP)
            while r2 == i or r2 == r1:
                r2 = random.randrange(self.NP)
            r3 = random.randrange(self.NP)
            while r3 == i or r3 == r1 or r3 == r2:
                r3 = random.randrange(self.NP)

            for j in range(self.D):
                self.V[i][j] = self.X[r1][j] + self.F * (self.X[r2][j] - self.X[r3][j])
                # limit of bound
                self.V[i][j] = (
                    self.x_low_bound
                    if self.V[i][j] < self.x_low_bound
                    else self.V[i][j]
                )
                self.V[i][j] = (
                    self.x_up_bound if self.V[i][j] > self.x_up_bound else self.V[i][j]
                )

    def crossover(self):
        for i in range(self.NP):
            for j in range(self.D):
                self.U[i][j] = (
                    self.V[i][j]
                    if random.random() <= self.CR or j == random.randrange(self.D)
                    else self.X[i][j]
                )
                # limit of bound
                self.U[i][j] = (
                    self.x_low_bound
                    if self.U[i][j] < self.x_low_bound
                    else self.U[i][j]
                )
                self.U[i][j] = (
                    self.x_up_bound if self.U[i][j] > self.x_up_bound else self.U[i][j]
                )

    def selection(self):
        for i in range(self.NP):
            x_fitness = self.cal_fitness(self.X[i])
            u_fitness = self.cal_fitness(self.U[i])
            if u_fitness < x_fitness:
                for j in range(self.D):
                    self.X[i][j] = self.U[i][j]

            x_fitness = self.cal_fitness(self.X[i])
            if x_fitness < self.global_fitness:
                self.global_fitness = x_fitness
                self.global_solution = self.X[i]

    def run(self):
        for i in range(self.max_iter):
            self.mutation()
            self.crossover()
            self.selection()


if __name__ == "__main__":
    calculate_fitness = (
        lambda x: x[0] ** 2
        + x[1] ** 2
        + x[2] ** 2
        + x[3] ** 2
        + x[4] ** 2
        + x[5] ** 2
        + x[6] ** 2
        + x[7] ** 2
        + x[8] ** 2
        + x[9] ** 2
    )
    de = DE(
        NP=50,
        D=10,
        max_iter=300,
        x_low_bound=-10,
        x_up_bound=10,
        F=0.5,
        CR=0.5,
        calculate_fitness=calculate_fitness,
    )
    de.run()
    print(de.global_solution)
    print(de.global_fitness)
