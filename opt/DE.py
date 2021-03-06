import numpy as np


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


class DE:
    def __init__(
        self,
        func,
        n_dim,
        F=0.5,
        size_pop=50,
        max_iter=200,
        prob_mut=0.3,
        lb=-1,
        ub=1,
        constraint_eq=tuple(),
        constraint_ueq=tuple(),
    ):
        self.func = func_transformer(func)
        self.n_dim = n_dim
        self.size_pop = size_pop  # size of population
        self.max_iter = max_iter
        self.prob_mut = prob_mut  # probability of mutation
        # constraint:
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

        self.F = F
        self.V, self.U = None, None
        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(
            self.n_dim
        )
        self.crtbp()

    def crtbp(self):
        # create the population
        self.X = np.random.uniform(
            low=self.lb, high=self.ub, size=(self.size_pop, self.n_dim)
        )
        return self.X

    def x2y(self):
            self.Y_raw = self.func(self.X)
            self.Y = self.Y_raw
            return self.Y

    def selection(self):
        """
        greedy selection
        """
        X = self.X.copy()
        f_X = self.x2y().copy()
        self.X = U = self.U
        f_U = self.x2y()

        self.X = np.where((f_X < f_U).reshape(-1, 1), X, U)

    def crossover(self):
        """
        if rand < prob_crossover, use V, else use X
        """
        mask = np.random.rand(self.size_pop, self.n_dim) < self.prob_mut
        self.U = np.where(mask, self.V, self.X)

    def mutation(self):
        """
        V[i]=X[r1]+F(X[r2]-X[r3]),
        where i, r1, r2, r3 are randomly generated
        """
        X = self.X
        # i is not needed,
        # and TODO: r1, r2, r3 should not be equal
        random_idx = np.random.randint(0, self.size_pop, size=(self.size_pop, 3))

        r1, r2, r3 = random_idx[:, 0], random_idx[:, 1], random_idx[:, 2]

        # ??????F????????????????????????????????????????????????????????????
        self.V = X[r1, :] + self.F * (X[r2, :] - X[r3, :])

        # the lower & upper bound still works in mutation
        # mask = np.random.uniform(
        #     low=self.lb, high=self.ub, size=(self.size_pop, self.n_dim)
        # )
        self.V = np.where(self.V < self.lb, self.lb, self.V)
        self.V = np.where(self.V > self.ub, self.ub, self.V)

    def run(self):
        for i in range(self.max_iter):
            self.mutation()
            self.crossover()
            self.selection()

            # record the best ones
            generation_best_index = self.Y.argmin()
            self.generation_best_X.append(self.X[generation_best_index, :].copy())
            self.generation_best_Y.append(self.Y[generation_best_index])
            self.all_history_Y.append(self.Y)

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

    de = DE(
        func=func,
        n_dim=10,
        size_pop=50,
        max_iter=300,
        lb=[0] * 10,
        ub=[100] * 10,
    )
    best_x, best_y = de.run()
    print("best_x:", best_x, "\n", "best_y:", best_y)

# %%
import pandas as pd
import matplotlib.pyplot as plt

Y_history = pd.DataFrame(de.all_history_Y)
fig, ax = plt.subplots(2, 1)
ax[0].plot(Y_history.index, Y_history.values, ".", color="red")
Y_history.min(axis=1).cummin().plot(kind="line")
plt.show()