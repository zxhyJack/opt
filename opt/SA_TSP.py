import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from SA import SimulatedAnnealingBase


def swap(individual):
    n1, n2 = np.random.randint(0, individual.shape[0] - 1, 2)
    if n1 >= n2:
        n1, n2 = n2, n1 + 1
    individual[n1], individual[n2] = individual[n2], individual[n1]
    return individual


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


def transpose(individual):
    # randomly generate n1 < n2 < n3. Notice: not equal
    n1, n2, n3 = sorted(np.random.randint(0, individual.shape[0] - 2, 3))
    n2 += 1
    n3 += 2
    slice1, slice2, slice3, slice4 = (
        individual[0:n1],
        individual[n1:n2],
        individual[n2 : n3 + 1],
        individual[n3 + 1 :],
    )
    individual = np.concatenate([slice1, slice3, slice2, slice4])
    return individual


class SA_TSP(SimulatedAnnealingBase):
    def cool_down(self):
        self.T = self.T_max / (1 + np.log(1 + self.iter_cycle))

    def get_new_x(self, x):
        x_new = x.copy()
        new_x_strategy = np.random.randint(3)
        if new_x_strategy == 0:
            x_new = swap(x_new)
        elif new_x_strategy == 1:
            x_new = reverse(x_new)
        elif new_x_strategy == 2:
            x_new = transpose(x_new)

        return x_new


if __name__ == "__main__":
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
    num_points = points_coordinate.shape[0]
    distance_matrix = spatial.distance.cdist(
        points_coordinate, points_coordinate, metric="euclidean"
    )
    distance_matrix = distance_matrix * 1000  # 1 degree of lat/lon ~ = 111000m

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

    sa_tsp = SA_TSP(
        func=cal_total_distance,
        x0=range(num_points),
        T_max=1000,
        T_min=1e-9,
        L=100,
    )

    best_points, best_distance = sa_tsp.run()
    print(best_points, best_distance, cal_total_distance(best_points))

    fig, ax = plt.subplots(1, 2)

    best_points_ = np.concatenate([best_points, [best_points[0]]])
    best_points_coordinate = points_coordinate[best_points_, :]
    ax[0].plot(sa_tsp.best_y_history)
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Distance")
    ax[1].plot(
        best_points_coordinate[:, 0],
        best_points_coordinate[:, 1],
        marker="o",
        markerfacecolor="b",
        color="c",
        linestyle="-",
    )
    # ax[1].xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    # ax[1].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    # ax[1].set_xlabel("Longitude")
    # ax[1].set_ylabel("Latitude")
    plt.show()