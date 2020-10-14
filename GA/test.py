import numpy as np
from GAComponents import Individual, Population
from GAOperators import RouletteWheelSelection, Crossover, Mutation
from GA import GA

# schaffer-N4
# sol: x=[0,1.25313], min=0.292579
schaffer_n4 = (
    lambda x: 0.5
    + (np.cos(np.sin(abs(x[0] ** 2 - x[1] ** 2))) ** 2 - 0.5)
    / (1.0 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2
)

fun = (
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

I = Individual([(0, 10)] * 10)
P = Population(I, 10)
S = RouletteWheelSelection()
C = Crossover(0.9, 0.75)
M = Mutation(0.2)
g = GA(P, S, C, M)

print(g.run(fun, 10).evaluation)

# res = []
# for i in range(10):
#     res.append(g.run(schaffer_n4, 50).evaluation)

# val = schaffer_n4([0, 1.25313])
# val_ga = sum(res) / len(res)

# print("the minimum: {0}".format(val))
# print("the GA minimum: {0}".format(val_ga))
# print("error: {:<3f} %".format((val_ga / val - 1.0) * 100))
