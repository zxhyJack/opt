import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class SimulatedAnnealingBase:
    """
    DO SA(Simulated Annealing)

    Parameters
    ----------------
    func : function
        The func you want to do optimal
    n_dim : int
        number of variables of func
    x0 : array, shape is n_dim
        initial solution
    T_max :float
        initial temperature
    T_min : float
        end temperature
    L : int
        num of iteration under every temperature（Long of Chain）

    Attributes
    ----------------------


    Examples
    -------------
    See https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_sa.py
    """

    def __init__(self, func, x0, T_max=100, T_min=1e-7, L=300, max_stay_counter=150, **kwargs):
        assert T_max > T_min > 0, 'T_max > T_min > 0'

        self.func = func
        self.T_max = T_max  # initial temperature
        self.T_min = T_min  # end temperature
        self.L = int(L)  # num of iteration under every temperature（also called Long of Chain）
        # stop if best_y stay unchanged over max_stay_counter times (also called cooldown time)
        self.max_stay_counter = max_stay_counter

        self.n_dims = len(x0)

        self.best_x = np.array(x0)  # initial solution
        self.best_y = self.func(self.best_x)
        self.T = self.T_max
        self.iter_cycle = 0
        self.generation_best_X, self.generation_best_Y = [self.best_x], [self.best_y]
        # history reasons, will be deprecated
        self.best_x_history, self.best_y_history = self.generation_best_X, self.generation_best_Y

    def get_new_x(self, x):
        u = np.random.uniform(-1, 1, size=self.n_dims)
        x_new = x + 20 * np.sign(u) * self.T * ((1 + 1.0 / self.T) ** np.abs(u) - 1.0)
        return x_new

    def cool_down(self):
        self.T = self.T * 0.99

    # def isclose(self, a, b, rel_tol=1e-09, abs_tol=1e-30):
    #     return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    def run(self):
        x_current, y_current = self.best_x, self.best_y
        stay_counter = 0
        # while True:
        for i in range(self.max_stay_counter):
            for i in range(self.L):
                x_new = self.get_new_x(x_current)
                y_new = self.func(x_new)

                # Metropolis
                df = y_new - y_current
                if df < 0 or np.exp(-df / self.T) > np.random.rand():
                    x_current, y_current = x_new, y_new
                    if y_new < self.best_y:
                        self.best_x, self.best_y = x_new, y_new

            self.iter_cycle += 1
            self.cool_down()
            self.generation_best_Y.append(self.best_y)
            self.generation_best_X.append(self.best_x)

            # if best_y stay for max_stay_counter times, stop iteration
            # if self.isclose(self.best_y_history[-1], self.best_y_history[-2]):
            #     stay_counter += 1
            # else:
            #     stay_counter = 0

            if self.T < self.T_min:
                stop_code = 'Cooled to final temperature'
                break
            if stay_counter > self.max_stay_counter:
                stop_code = 'Stay unchanged in the last {stay_counter} iterations'.format(stay_counter=stay_counter)
                break

        return self.best_x, self.best_y

if __name__=="__main__":
    # func = lambda x: x[0] ** 2 + (x[1] - 0.05) ** 2 + x[2] ** 2
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

    sa = SimulatedAnnealingBase(func=func, x0=range(10), T_max=100, T_min=1e-9, L=300, max_stay_counter=3000)
    best_x, best_y = sa.run()
    print('best_x:', best_x)
    print('best_y', best_y)

    # plt.plot(pd.DataFrame(sa.best_y_history).cummin(axis=0))
    plt.plot(sa.best_y_history)
    plt.show()
