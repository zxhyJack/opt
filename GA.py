import numpy as np

# import pandas as pd
# import matplotlib.pyplot as plt


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


class GA:
    """genetic algorithm

    Parameters
    ----------------
    func : function
        The func you want to do optimal
    n_dim : int
        number of variables of func
    lb : array_like
        The lower bound of every variables of func
    ub : array_like
        The upper bound of every variables of func
    constraint_eq : tuple
        equal constraint
    constraint_ueq : tuple
        unequal constraint
    precision : array_like
        The precision of every variables of func
    size_pop : int
        Size of population
    max_iter : int
        Max of iter
    prob_mut : float between 0 and 1
        Probability of mutation
    Attributes
    ----------------------
    Lind : array_like
         The num of genes of every variable of func（segments）
    generation_best_X : array_like. Size is max_iter.
        Best X of every generation
    generation_best_ranking : array_like. Size if max_iter.
        Best ranking of every generation
    """

    def __init__(
        self,
        func,
        n_dim,
        size_pop=50,
        max_iter=200,
        prob_mut=0.001,
        lb=-1,
        ub=1,
        constraint_eq=tuple(),
        constraint_ueq=tuple(),
        precision=1e-7,
    ):
        self.func = func_transformer(func)
        self.n_dim = n_dim
        self.size_pop = size_pop  # size of population
        self.max_iter = max_iter
        self.prob_mut = prob_mut  # probability of mutation
        self.lb = np.array(lb) * np.ones(self.n_dim)
        self.ub = np.array(ub) * np.ones(self.n_dim)

        # constraint:
        self.has_constraint = len(constraint_eq) > 0 or len(constraint_ueq) > 0
        self.constraint_eq = list(
            constraint_eq
        )  # a list of unequal constraint functions with c[i] <= 0
        self.constraint_ueq = list(
            constraint_ueq
        )  # a list of equal functions with ceq[i] = 0

        self.Chrom = None
        self.X = None  # shape = (size_pop, n_dim)
        self.Y_raw = None  # shape = (size_pop,) , value is f(x)
        self.Y = None  # shape = (size_pop,) , value is f(x) + penalty for constraint
        self.FitV = None  # shape = (size_pop,)

        self.FitV_history = []
        self.generation_best_X = []
        self.generation_best_Y = []

        self.all_history_Y = []
        self.all_history_FitV = []

        self.best_x, self.best_y = None, None

        self.precision = np.array(precision) * np.ones(
            self.n_dim
        )  # works when precision is int, float, list or array

        # Lind is the num of genes of every variable of func（segments）
        Lind_raw = np.log2((self.ub - self.lb) / self.precision + 1)
        self.Lind = np.ceil(Lind_raw).astype(int)

        # if precision is integer:
        # if Lind_raw is integer, which means the number of all possible value is 2**n, no need to modify
        # if Lind_raw is decimal, we need ub_extend to make the number equal to 2**n,
        self.int_mode_ = (self.precision % 1 == 0) & (Lind_raw % 1 != 0)
        self.int_mode = np.any(self.int_mode_)
        if self.int_mode:
            self.ub_extend = np.where(
                self.int_mode_,
                self.lb + (np.exp2(self.Lind) - 1) * self.precision,
                self.ub,
            )

        self.len_chrom = sum(self.Lind)

        self.crtbp()

    def crtbp(self):
        # create the population
        self.Chrom = np.random.randint(
            low=0, high=2, size=(self.size_pop, self.len_chrom)
        )
        return self.Chrom

    def gray2rv(self, gray_code):
        # Gray Code to real value: one piece of a whole chromosome
        # input is a 2-dimensional numpy array of 0 and 1.
        # output is a 1-dimensional numpy array which convert every row of input into a real number.
        _, len_gray_code = gray_code.shape
        b = gray_code.cumsum(axis=1) % 2
        mask = np.logspace(start=1, stop=len_gray_code, base=0.5, num=len_gray_code)
        return (b * mask).sum(axis=1) / mask.sum()

    def chrom2x(self, Chrom):
        cumsum_len_segment = self.Lind.cumsum()
        X = np.zeros(shape=(self.size_pop, self.n_dim))
        for i, j in enumerate(cumsum_len_segment):
            if i == 0:
                Chrom_temp = Chrom[:, : cumsum_len_segment[0]]
            else:
                Chrom_temp = Chrom[:, cumsum_len_segment[i - 1] : cumsum_len_segment[i]]
            X[:, i] = self.gray2rv(Chrom_temp)

        if self.int_mode:
            X = self.lb + (self.ub_extend - self.lb) * X
            X = np.where(X > self.ub, self.ub, X)
            # the ub may not obey precision, which is ok.
            # for example, if precision=2, lb=0, ub=5, then x can be 5
        else:
            X = self.lb + (self.ub - self.lb) * X
        return X

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

    def selection(self, tourn_size=3):
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
        self.Chrom = self.Chrom[sel_index, :]  # ?
        return self.Chrom

    def crossover(self):
        """
        3 times faster than `crossover_2point`, but only use for 0/1 type of Chrom
        :param self:
        :return:
        """
        Chrom, size_pop, len_chrom = self.Chrom, self.size_pop, self.len_chrom
        half_size_pop = int(size_pop / 2)
        Chrom1, Chrom2 = Chrom[:half_size_pop], Chrom[half_size_pop:]
        mask = np.zeros(shape=(half_size_pop, len_chrom), dtype=int)
        for i in range(half_size_pop):
            n1, n2 = np.random.randint(0, self.len_chrom, 2)
            if n1 > n2:
                n1, n2 = n2, n1
            mask[i, n1:n2] = 1
        mask2 = (Chrom1 ^ Chrom2) & mask
        Chrom1 ^= mask2
        Chrom2 ^= mask2
        return self.Chrom

    def mutation(self):
        """
        mutation of 0/1 type chromosome
        faster than `self.Chrom = (mask + self.Chrom) % 2`
        :param self:
        :return:
        """
        #
        mask = np.random.rand(self.size_pop, self.len_chrom) < self.prob_mut
        self.Chrom ^= mask
        return self.Chrom

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for i in range(self.max_iter):
            self.X = self.chrom2x(self.Chrom)
            self.Y = self.x2y()
            self.ranking()
            self.selection()
            self.crossover()
            self.mutation()

            # record the best ones
            generation_best_index = self.FitV.argmax()
            self.generation_best_X.append(self.X[generation_best_index, :])
            self.generation_best_Y.append(self.Y[generation_best_index])
            self.all_history_Y.append(self.Y)
            self.all_history_FitV.append(self.FitV)

        global_best_index = np.array(self.generation_best_Y).argmin()
        self.best_x = self.generation_best_X[global_best_index]
        self.best_y = self.func(np.array([self.best_x]))
        return self.best_x, self.best_y


if __name__ == "__main__":

    func = (
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

    ga = GA(
        func=func,
        n_dim=10,
        size_pop=50,
        max_iter=300,
        lb=[0] * 10,
        ub=[100] * 10,
        precision=1e-7,
    )
    best_x, best_y = ga.run()
    print("best_x:", best_x, "\n", "best_y:", best_y)
# %%
import pandas as pd
import matplotlib.pyplot as plt

Y_history = pd.DataFrame(ga.all_history_Y)
fig, ax = plt.subplots(2, 1)
ax[0].plot(Y_history.index, Y_history.values, ".", color="red")
Y_history.min(axis=1).cummin().plot(kind="line")
plt.show()