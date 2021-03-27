import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 10  # DNA length
POP_SIZE = 50  # population size
CROSS_RATE = 0.5  # mating probability (DNA crossover)
MUTATION_RATE = 0.003  # mutation probability
N_GENERATIONS = 50
X_BOUND = [0, 100]  # x upper and lower bounds


# def F(x):
#     return (
#         np.sin(10 * x) * x + np.cos(2 * x) * x
#     )  # to find the maximum of this function

F = (
    lambda x: x ** 2
    # + x[1] ** 2
    # + x[2] ** 2
    # + x[3] ** 2
    # + x[4] ** 2
    # + x[5] ** 2
    # + x[6] ** 2
    # + x[7] ** 2
    # + x[8] ** 2
    # + x[9] ** 2
)

# find non-zero fitness for selection
def get_fitness(pred):
    return pred + 1e-3 - np.min(pred)


# convert binary DNA to decimal and normalize it to a range(0, 5)
def translateDNA(pop):
    return (
        pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * X_BOUND[1]
    )


def select(pop, fitness):  # nature selection wrt pop's fitness
    index = np.random.choice(
        np.arange(POP_SIZE),
        size=POP_SIZE,
        replace=True,
        p=fitness / fitness.sum(),
    )
    return pop[index]


def crossover(parent, pop):  # mating process (genes crossover)
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)  # select another individual from pop
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)  # choose crossover points
        parent[cross_points] = pop[i_, cross_points]  # mating and produce one child
    return parent


def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child


pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))  # initialize the pop DNA

plt.ion()  # something about plotting
x = np.linspace(*X_BOUND, num=200)    # *X_BOUND为解构列表X_BOUND
plt.plot(x, F(x))

for _ in range(N_GENERATIONS):
    F_values = F(translateDNA(pop))  # compute function value by extracting DNA

    # something about plotting
    if "sca" in globals():
        sca.remove()
    sca = plt.scatter(translateDNA(pop), F_values, s=200, lw=0, c="red", alpha=0.5)
    plt.pause(0.05)

    # GA part (evolution)
    fitness = -get_fitness(F_values)
    print("Most fitted DNA: ", pop[np.argmin(fitness), :])
    pop = select(pop, fitness)
    pop_copy = pop.copy()
    for parent in pop:
        child = crossover(parent, pop_copy)
        child = mutate(child)
        parent[:] = child  # parent is replaced by its child

plt.ioff()
plt.show()