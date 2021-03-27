import numpy as np
from scipy import spatial
import pandas
import matplotlib.pyplot as plt

# np.seterr(divide="ignore", invalid="ignore")


class ACO:
    def __init__(
        self,
        func,
        n_dim,
        size_pop=10,
        max_iter=20,
        alpha=1,
        beta=2,
        rho=0.1,
        distance_matrix=None,
    ):
        self.func = func
        self.n_dim = n_dim  # 城市数量
        self.size_pop = size_pop  # 蚂蚁数量
        self.max_iter = max_iter  # 迭代次数
        self.alpha = alpha  # 信息素重要程度
        self.beta = beta  # 适应度的重要程度
        self.rho = rho  # 信息素挥发速度

        self.prob_matrix_distance = 1 / (  # eta的值
            distance_matrix + 1e-10 * np.eye(n_dim, n_dim)
        )  # 避免除零错误

        self.Tau = np.ones((n_dim, n_dim))  # 信息素矩阵，每次迭代都会更新
        self.Table = np.zeros((size_pop, n_dim)).astype(np.int)  # 某一代每个蚂蚁的爬行路径
        self.y = None  # 某一代每个蚂蚁的爬行总距离
        self.generation_best_X, self.generation_best_Y = [], []  # 记录各代的最佳情况
        self.x_best_history, self.y_best_history = (
            self.generation_best_X,
            self.generation_best_Y,
        )  # 历史原因，为了保持统一
        self.best_x, self.best_y = None, None

    def run(self):
        for i in range(self.max_iter):  # 对每次迭代
            prob_matrix = (self.Tau ** self.alpha) * (
                self.prob_matrix_distance
            ) ** self.beta  # 转移概率，无须归一化。
            for j in range(self.size_pop):  # 对每个蚂蚁
                self.Table[j, 0] = 0  # start point，其实可以随机，但没什么区别
                for k in range(self.n_dim - 1):  # 蚂蚁到达的每个节点
                    taboo_set = set(self.Table[j, : k + 1])  # 已经经过的点和当前点，不能再次经过
                    allow_list = list(set(range(self.n_dim)) - taboo_set)  # 在这些点中做选择
                    prob = prob_matrix[self.Table[j, k], allow_list]
                    prob = prob / prob.sum()  # 概率归一化
                    # print(prob)
                    next_point = np.random.choice(allow_list, size=1, p=prob)[0]
                    self.Table[j, k + 1] = next_point

            # 计算距离
            y = np.array([self.func(i) for i in self.Table])

            # 顺便记录历史最好情况
            index_best = y.argmin()
            x_best, y_best = self.Table[index_best, :].copy(), y[index_best].copy()
            self.generation_best_X.append(x_best)
            self.generation_best_Y.append(y_best)

            # 计算需要新涂抹的信息素
            Delta_tau = np.zeros((self.n_dim, self.n_dim))
            for j in range(self.size_pop):  # 每个蚂蚁
                for k in range(self.n_dim - 1):  # 每个节点
                    n1, n2 = self.Table[j, k], self.Table[j, k + 1]  # 蚂蚁从n1节点爬到n2节点
                    Delta_tau[n1, n2] += 1 / y[j]  # 涂抹的信息素
                n1, n2 = (
                    self.Table[j, self.n_dim - 1],
                    self.Table[j, 0],
                )  # 蚂蚁从最后一个节点爬回到第一个节点
                Delta_tau[n1, n2] += 1 / y[j]  # 涂抹信息素

            # 信息素挥发+信息素涂抹
            self.Tau = (1 - self.rho) * self.Tau + Delta_tau

        best_generation = np.array(self.generation_best_Y).argmin()
        self.best_x = self.generation_best_X[best_generation]
        self.best_y = self.generation_best_Y[best_generation]
        return self.best_x, self.best_y

    # fit = run


if __name__ == "__main__":
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

    # points_coordinate = np.random.rand(num_points, 2)  # generate coordinate of points
    # print(points_coordinate.shape)

    distance_matrix = spatial.distance.cdist(  # 计算两个点集合中两两点之间的距离
        points_coordinate, points_coordinate, metric="euclidean"
    )

    # print(distance_matrix)

    def cal_total_distance(routine):
        (num_points,) = routine.shape
        return sum(
            [
                distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]]
                for i in range(num_points)
            ]
        )

    aco = ACO(
        func=cal_total_distance,
        n_dim=num_points,
        size_pop=50,
        max_iter=300,
        alpha=1,
        beta=2,
        rho=0.1,
        distance_matrix=distance_matrix,
    )

    best_x, best_y = aco.run()
    print(best_x)
    print(best_y)

    fig, ax = plt.subplots(1)
    best_points_ = np.concatenate([best_x, [best_x[0]]])
    best_points_coordinate = points_coordinate[best_points_, :]
    ax.plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], "o-r")
    # pandas.DataFrame(aco.y_best_history).cummin().plot(ax=ax[1])
    plt.show()