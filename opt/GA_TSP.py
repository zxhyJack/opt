import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt


def func_transformer(func):

    prefered_function_format = """
    def demo_func(x):
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2
    """

    is_vector = getattr(func, "is_vector", False)
    if is_vector:
        return func
    else:
        if func.__code__.co_argcount == 1:

            def func_transformed(X):
                return np.array([func(x) for x in X])

            return func_transformed
        elif func.__code__.co_argcount > 1:

            def func_transformed(X):
                return np.array([func(*tuple(x)) for x in X])

            return func_transformed

    raise ValueError(
        """
    object function error,
    function should be like this:
    """
        + prefered_function_format
    )


def reverse(individual):
    """
    Reverse n1 to n2
    Also called `2-Opt`: removes two random edges, reconnecting them so they cross
    Karan Bhatia, "Genetic Algorithms and the Traveling Salesman Problem", 1994
    https://pdfs.semanticscholar.org/c5dd/3d8e97202f07f2e337a791c3bf81cd0bbb13.pdf
    """
    n1, n2 = np.random.randint(0, individual.shape[0] - 1, 2)
    if n1 >= n2:
        n1, n2 = n2, n1 + 1
    individual[n1:n2] = individual[n1:n2][::-1]
    return individual


class GA_TSP:
    """
    Do genetic algorithm to solve the TSP (Travelling Salesman Problem)
    Parameters
    ----------------
    func : function
        The func you want to do optimal.
        It inputs a candidate solution(a routine), and return the costs of the routine.
    size_pop : int
        Size of population
    max_iter : int
        Max of iter
    prob_mut : float between 0 and 1
        Probability of mutation
    Attributes
    ----------------------
    Lind : array_like
         The num of genes corresponding to every variable of func（segments）
    generation_best_X : array_like. Size is max_iter.
        Best X of every generation
    generation_best_ranking : array_like. Size if max_iter.
        Best ranking of every generation
    Examples
    -------------
    Firstly, your data (the distance matrix). Here I generate the data randomly as a demo:
    ```py
    num_points = 8
    points_coordinate = np.random.rand(num_points, 2)  # generate coordinate of points
    distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
    print('distance_matrix is: \n', distance_matrix)
    def cal_total_distance(routine):
        num_points, = routine.shape
        return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])
    ```
    Do GA
    ```py
    from sko.GA import GA_TSP
    ga_tsp = GA_TSP(func=cal_total_distance, n_dim=8, pop=50, max_iter=200, Pm=0.001)
    best_points, best_distance = ga_tsp.run()
    ```
    """

    def __init__(self, func, n_dim, size_pop=50, max_iter=200, prob_mut=0.001):
        self.func = func_transformer(func)
        self.size_pop = size_pop  # size of population
        self.max_iter = max_iter
        self.prob_mut = prob_mut  # probability of mutation
        self.n_dim = n_dim

        self.Chrom = None
        self.X = None  # shape = (size_pop, n_dim)
        self.Y_raw = None  # shape = (size_pop,) , value is f(x)
        self.Y = None  # shape = (size_pop,) , value is f(x) + penalty for constraint
        self.FitV = None  # shape = (size_pop,)

        # self.FitV_history = []
        self.generation_best_X = []
        self.generation_best_Y = []

        self.all_history_Y = []
        self.all_history_FitV = []

        self.best_x, self.best_y = None, None

        self.has_constraint = False
        self.len_chrom = self.n_dim
        self.crtbp()

    def crtbp(self):
        # create the population
        tmp = np.random.rand(self.size_pop, self.len_chrom)
        self.Chrom = tmp.argsort(axis=1)
        return self.Chrom

    def chrom2x(self, Chrom):
        return Chrom

    def x2y(self):
        self.Y_raw = self.func(self.X)
        if not self.has_constraint:
            self.Y = self.Y_raw
        else:
            # constraint
            penalty_eq = np.array(
                [np.sum(np.abs([c_i(x) for c_i in self.constraint_eq])) for x in self.X]
            )
            penalty_ueq = np.array(
                [
                    np.sum(np.abs([max(0, c_i(x)) for c_i in self.constraint_ueq]))
                    for x in self.X
                ]
            )
            self.Y = self.Y_raw + 1e5 * penalty_eq + 1e5 * penalty_ueq
        return self.Y

    def ranking(self):
        # GA select the biggest one, but we want to minimize func, so we put a negative here
        self.FitV = -self.Y

    def selection_tournament_faster(self, tourn_size=3):
        """
        Select the best individual among *tournsize* randomly chosen
        Same with `selection_tournament` but much faster using numpy
        individuals,
        :param self:
        :param tourn_size:
        :return:
        """
        aspirants_idx = np.random.randint(
            self.size_pop, size=(self.size_pop, tourn_size)
        )
        aspirants_values = self.FitV[aspirants_idx]
        winner = aspirants_values.argmax(axis=1)  # winner index in every team
        sel_index = [aspirants_idx[i, j] for i, j in enumerate(winner)]
        self.Chrom = self.Chrom[sel_index, :]
        return self.Chrom

    def crossover_pmx(self):
        """
        Executes a partially matched crossover (PMX) on Chrom.
        For more details see [Goldberg1985]_.

        :param self:
        :return:

        .. [Goldberg1985] Goldberg and Lingel, "Alleles, loci, and the traveling
        salesman problem", 1985.
        """
        Chrom, size_pop, len_chrom = self.Chrom, self.size_pop, self.len_chrom
        for i in range(0, size_pop, 2):
            Chrom1, Chrom2 = self.Chrom[i], self.Chrom[i + 1]
            cxpoint1, cxpoint2 = np.random.randint(0, self.len_chrom - 1, 2)
            if cxpoint1 >= cxpoint2:
                cxpoint1, cxpoint2 = cxpoint2, cxpoint1 + 1
            # crossover at the point cxpoint1 to cxpoint2
            pos1_recorder = {value: idx for idx, value in enumerate(Chrom1)}
            pos2_recorder = {value: idx for idx, value in enumerate(Chrom2)}
            for j in range(cxpoint1, cxpoint2):
                value1, value2 = Chrom1[j], Chrom2[j]
                pos1, pos2 = pos1_recorder[value2], pos2_recorder[value1]
                Chrom1[j], Chrom1[pos1] = Chrom1[pos1], Chrom1[j]
                Chrom2[j], Chrom2[pos2] = Chrom2[pos2], Chrom2[j]
                pos1_recorder[value1], pos1_recorder[value2] = pos1, j
                pos2_recorder[value1], pos2_recorder[value2] = j, pos2

            self.Chrom[i], self.Chrom[i + 1] = Chrom1, Chrom2
        return self.Chrom

    def mutation_reverse(self):
        """
        Reverse
        :param self:
        :return:
        """
        for i in range(self.size_pop):
            if np.random.rand() < self.prob_mut:
                self.Chrom[i] = reverse(self.Chrom[i])
        return self.Chrom

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for i in range(self.max_iter):
            Chrom_old = self.Chrom.copy()
            self.X = self.chrom2x(self.Chrom)
            self.Y = self.x2y()
            self.ranking()
            self.selection_tournament_faster()
            self.crossover_pmx()
            self.mutation_reverse()

            # put parent and offspring together and select the best size_pop number of population
            self.Chrom = np.concatenate([Chrom_old, self.Chrom], axis=0)
            self.X = self.chrom2x(self.Chrom)
            self.Y = self.x2y()
            self.ranking()
            selected_idx = np.argsort(self.Y)[: self.size_pop]
            self.Chrom = self.Chrom[selected_idx, :]

            # record the best ones
            generation_best_index = self.FitV.argmax()
            self.generation_best_X.append(self.X[generation_best_index, :].copy())
            self.generation_best_Y.append(self.Y[generation_best_index])
            self.all_history_Y.append(self.Y.copy())
            self.all_history_FitV.append(self.FitV.copy())

        global_best_index = np.array(self.generation_best_Y).argmin()
        self.best_x = self.generation_best_X[global_best_index]
        self.best_y = self.func(np.array([self.best_x]))
        return self.best_x, self.best_y


if __name__ == "__main__":
    # num_points = 50
    # points_coordinate = np.random.rand(num_points, 2)  # generate coordinate of points

    num_points = 101
    points_coordinate = np.array(  #  data set
        [
            [40.00, 50.00],
            [45.00, 68.00],
            [45.00, 70.00],
            [42.00, 66.00],
            [42.00, 68.00],
            [42.00, 65.00],
            [40.00, 69.00],
            [40.00, 66.00],
            [38.00, 68.00],
            [38.00, 70.00],
            [35.00, 66.00],
            [35.00, 69.00],
            [25.00, 85.00],
            [22.00, 75.00],
            [22.00, 85.00],
            [20.00, 80.00],
            [20.00, 85.00],
            [18.00, 75.00],
            [15.00, 75.00],
            [15.00, 80.00],
            [30.00, 50.00],
            [30.00, 52.00],
            [28.00, 52.00],
            [28.00, 55.00],
            [25.00, 50.00],
            [25.00, 52.00],
            [25.00, 55.00],
            [23.00, 52.00],
            [23.00, 55.00],
            [20.00, 50.00],
            [20.00, 55.00],
            [10.00, 35.00],
            [10.00, 40.00],
            [8.00, 40.00],
            [8.00, 45.00],
            [5.00, 35.00],
            [5.00, 45.00],
            [2.00, 40.00],
            [0.00, 40.00],
            [0.00, 45.00],
            [35.00, 30.00],
            [35.00, 32.00],
            [33.00, 32.00],
            [33.00, 35.00],
            [32.00, 30.00],
            [30.00, 30.00],
            [30.00, 32.00],
            [30.00, 35.00],
            [28.00, 30.00],
            [28.00, 35.00],
            [26.00, 32.00],
            [25.00, 30.00],
            [25.00, 35.00],
            [44.00, 5.00],
            [42.00, 10.00],
            [42.00, 15.00],
            [40.00, 5.00],
            [40.00, 15.00],
            [38.00, 5.00],
            [38.00, 15.00],
            [35.00, 5.00],
            [50.00, 30.00],
            [50.00, 35.00],
            [50.00, 40.00],
            [48.00, 30.00],
            [48.00, 40.00],
            [47.00, 35.00],
            [47.00, 40.00],
            [45.00, 30.00],
            [45.00, 35.00],
            [95.00, 30.00],
            [95.00, 35.00],
            [53.00, 30.00],
            [92.00, 30.00],
            [53.00, 35.00],
            [45.00, 65.00],
            [90.00, 35.00],
            [88.00, 30.00],
            [88.00, 35.00],
            [87.00, 30.00],
            [85.00, 25.00],
            [85.00, 35.00],
            [75.00, 55.00],
            [72.00, 55.00],
            [70.00, 58.00],
            [68.00, 60.00],
            [66.00, 55.00],
            [65.00, 55.00],
            [65.00, 60.00],
            [63.00, 58.00],
            [60.00, 55.00],
            [60.00, 60.00],
            [67.00, 85.00],
            [65.00, 85.00],
            [65.00, 82.00],
            [62.00, 80.00],
            [60.00, 80.00],
            [60.00, 85.00],
            [58.00, 75.00],
            [55.00, 80.00],
            [55.00, 85.00],
        ]
    )

    distance_matrix = spatial.distance.cdist(
        points_coordinate, points_coordinate, metric="euclidean"
    )

    def cal_total_distance(routine):
        """The objective function. input routine, return total distance.
        cal_total_distance(np.arange(num_points))
        """
        (num_points,) = routine.shape
        return sum(
            [
                distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]]
                for i in range(num_points)
            ]
        )

    ga_tsp = GA_TSP(
        func=cal_total_distance,
        n_dim=num_points,
        size_pop=50,
        max_iter=4000,
        prob_mut=1,
    )
    
    best_points, best_distance = ga_tsp.run()

    print("best_points", best_points)
    print("best_distance", best_distance)

    fig, ax = plt.subplots(1)
    best_points_ = np.concatenate([best_points, [best_points[0]]])
    best_points_coordinate = points_coordinate[best_points_, :]
    ax.plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], "o-r")
    # ax[1].plot(ga_tsp.generation_best_Y)
    plt.show()
