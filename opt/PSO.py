import numpy as np
import matplotlib.pyplot as plt


def func_transformer(func):
    """
    transform this kind of function:
    ```
    def demo_func(x):
        x1, x2, x3 = x
        return x1 ** 2 + x2 ** 2 + x3 ** 2
    ```
    into this kind of function:
    ```
    def demo_func(x):
        x1, x2, x3 = x[:,0], x[:,1], x[:,2]
        return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2
    ```
    getting vectorial performance if possible
    :param func:
    :return:
    """

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


class PSO:
    """
    Do PSO (Particle swarm optimization) algorithm.

    This algorithm was adapted from the earlier works of J. Kennedy and
    R.C. Eberhart in Particle Swarm Optimization [IJCNN1995]_.

    The position update can be defined as:

    .. math::

       x_{i}(t+1) = x_{i}(t) + v_{i}(t+1)

    Where the position at the current step :math:`t` is updated using
    the computed velocity at :math:`t+1`. Furthermore, the velocity update
    is defined as:

    .. math::

       v_{ij}(t + 1) = w * v_{ij}(t) + c_{p}r_{1j}(t)[y_{ij}(t) − x_{ij}(t)]
                       + c_{g}r_{2j}(t)[\hat{y}_{j}(t) − x_{ij}(t)]

    Here, :math:`cp` and :math:`cg` are the cognitive and social parameters
    respectively. They control the particle's behavior given two choices: (1) to
    follow its *personal best* or (2) follow the swarm's *global best* position.
    Overall, this dictates if the swarm is explorative or exploitative in nature.
    In addition, a parameter :math:`w` controls the inertia of the swarm's
    movement.

    .. [IJCNN1995] J. Kennedy and R.C. Eberhart, "Particle Swarm Optimization,"
    Proceedings of the IEEE International Joint Conference on Neural
    Networks, 1995, pp. 1942-1948.

    Parameters
    --------------------
    func : function
        The func you want to do optimal
    dim : int
        Number of dimension, which is number of parameters of func.
    pop : int
        Size of population, which is the number of Particles. We use 'pop' to keep accordance with GA
    max_iter : int
        Max of iter iterations

    Attributes
    ----------------------
    pbest_x : array_like, shape is (pop,dim)
        best location of every particle in history
    pbest_y : array_like, shape is (pop,1)
        best image of every particle in history
    gbest_x : array_like, shape is (1,dim)
        general best location for all particles in history
    gbest_y : float
        general best image  for all particles in history
    gbest_y_hist : list
        gbest_y of every iteration

    """

    def __init__(
        self, func, dim, pop=40, max_iter=150, lb=None, ub=None, w=0.8, c1=0.5, c2=0.5
    ):
        self.func = func_transformer(func)
        self.w = w  # inertia
        self.cp, self.cg = (
            c1,
            c2,
        )  # parameters to control personal best, global best respectively
        self.pop = pop  # number of particles
        self.dim = (
            dim  # dimension of particles, which is the number of variables of func
        )
        self.max_iter = max_iter  # max iter

        self.has_constraints = not (lb is None and ub is None)
        self.lb = -np.ones(self.dim) if lb is None else np.array(lb)
        self.ub = np.ones(self.dim) if ub is None else np.array(ub)
        assert (
            self.dim == len(self.lb) == len(self.ub)
        ), "dim == len(lb) == len(ub) is not True"
        assert np.all(self.ub > self.lb), "upper-bound must be greater than lower-bound"

        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.dim))
        v_high = self.ub - self.lb
        self.V = np.random.uniform(
            low=-v_high, high=v_high, size=(self.pop, self.dim)
        )  # speed of particles
        self.Y = self.cal_y()  # y = f(x) for all particles
        self.pbest_x = (
            self.X.copy()
        )  # personal best location of every particle in history
        self.pbest_y = self.Y.copy()  # best image of every particle in history
        self.gbest_x = np.zeros((1, self.dim))  # global best location for all particles
        self.gbest_y = np.inf  # global best y for all particles
        self.gbest_y_hist = []  # gbest_y of every iteration
        self.update_gbest()

        # record verbose values
        self.record_mode = False
        self.record_value = {"X": [], "V": [], "Y": []}
        self.best_x, self.best_y = (
            self.gbest_x,
            self.gbest_y,
        )  # history reasons, will be deprecated

    def update_V(self):
        r1 = np.random.rand(self.pop, self.dim)
        r2 = np.random.rand(self.pop, self.dim)
        self.V = (
            self.w * self.V
            + self.cp * r1 * (self.pbest_x - self.X)
            + self.cg * r2 * (self.gbest_x - self.X)
        )

    def update_X(self):
        self.X = self.X + self.V

        if self.has_constraints:
            self.X = np.clip(self.X, self.lb, self.ub)

    def cal_y(self):
        # calculate y for every x in X
        self.Y = self.func(self.X).reshape(-1, 1)
        return self.Y

    def update_pbest(self):
        """
        personal best
        :return:
        """
        self.pbest_x = np.where(self.pbest_y > self.Y, self.X, self.pbest_x)
        self.pbest_y = np.where(self.pbest_y > self.Y, self.Y, self.pbest_y)

    def update_gbest(self):
        """
        global best
        :return:
        """
        if self.gbest_y > self.Y.min():
            self.gbest_x = self.X[self.Y.argmin(), :].copy()
            self.gbest_y = self.Y.min()

    def recorder(self):
        if not self.record_mode:
            return
        self.record_value["X"].append(self.X)
        self.record_value["V"].append(self.V)
        self.record_value["Y"].append(self.Y)

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for iter_num in range(self.max_iter):
            self.update_V()
            self.recorder()
            self.update_X()
            self.cal_y()
            self.update_pbest()
            self.update_gbest()

            self.gbest_y_hist.append(self.gbest_y)
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y

    fit = run


if __name__ == "__main__":

    def demo_func(x):
        return (
            x[0] ** 2
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

    pso = PSO(
        func=demo_func,
        dim=10,
        pop=40,
        max_iter=300,
        lb=[-10] * 10,
        ub=[10] * 10,
        w=0.8,
        c1=0.5,
        c2=0.5,
    )
    pso.run()
    print("best_x is ", pso.gbest_x)
    print("best_y is", pso.gbest_y)

    plt.plot(pso.gbest_y_hist)
    # plt.plot(x, y)
    plt.show()
