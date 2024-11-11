import random
from copy import deepcopy as cp
class Board:
    """
    这是有关迷宫的基本设置父类
    """
    ## 属性
    # 分别表示棋盘的长和宽
    n, m = None, None
    # 最低容忍误差（迭代终止条件）
    tolerances = 1e-15
    # 惩罚值
    penalty_value = -500
    # 奖励值
    prize_value = 10
    # 该值应该的值域：[0,1]，用于表示非最优策略的概率占比大小，可以视作探索性的大小
    varepsilon = 0.4
    # 每个状态的行动数
    sum_action = 5
    # 参数，这个值越大越远视，越小越近视
    _lambda = 0.9

    # 行动
    xx = [0, 0, 0, -1, 1]
    yy = [0, -1, 1, 0, 0]
    s = "O←→↑↓"


    ## 矩阵
    # 棋盘地图矩阵
    chessboard = [["*****"],
                  ["*##**"],
                  ["**#**"],
                  ["*#X#*"],
                  ["*#***"],]
    # 惩罚值矩阵
    Punishment = None
    # 最优策略矩阵，一共有n*m个状态，这里将其放到二维张量中
    policy = None

    # 状态行动概率矩阵（三维）
    state_action_probabilities = None
    # 状态行动价值矩阵（三维）
    state_action_values = None


    # 状态矩阵
    states = None

    def init_matrix(self):
        """
        矩阵初始化
        :return:None
        """
        self.chessboard = [list(x[0]) for x in self.chessboard]
        self.policy = [[0 for _ in range(len(x))] for x in self.chessboard]
        self.states = [[0 for _ in range(len(x))] for x in self.chessboard]
        self.get_Punishment()

        # 初始化两个三维张量
        init_probabilities = [1 / self.sum_action] * self.sum_action
        self.state_action_probabilities = [[cp(init_probabilities) for _ in range(self.m)] for _ in range(self.n)]

        init_values = [0] * self.sum_action
        self.state_action_values = [[cp(init_values) for _ in range(self.m)] for _ in range(self.n)]

    def get_Punishment(self):
        """
        计算惩罚值矩阵
        :return:None
        """

        self.Punishment = [[None for _ in range(len(x))] for x in self.chessboard]
        # print(self.Punishment)
        for i in range(self.n):
            for j in range(self.m):
                if self.chessboard[i][j] == "*":
                    self.Punishment[i][j] = 0
                elif self.chessboard[i][j] == "#":
                    self.Punishment[i][j] = self.penalty_value
                else:
                    self.Punishment[i][j] = self.prize_value
        return self.Punishment

    @staticmethod
    def debug(lit):
        print("*" * 20)
        for x in lit:
            for y in x:
                print(y, end="\t\t")
            print()

    def get_reward_by_action(self, x, y, i):
        """
         返回某一行动的奖惩值
        :param x: x坐标
        :param y: y坐标
        :param i: 行动
        :return: 奖惩值
        """
        next_x, next_y = self.next_point(x, y, i)

        # 超界惩罚
        if self.is_crossing_boundaries(next_x, next_y):
            return self.penalty_value
        # print(next_x, next_y, self.n, self.m)
        return self.Punishment[next_x][next_y]

    def next_point(self, x, y, i):
        x += self.xx[i]
        y += self.yy[i]
        return x, y

    def get_states(self):
        """
        更新状态
        :return:None
        """
        # 记录改变值
        tolerances_sum = 0
        new_states = [[0 for _ in range(len(x))] for x in self.chessboard]
        for x in range(self.n):
            for y in range(self.m):
                next_x, next_y = self.next_point(x, y, self.policy[x][y])
                new_states[x][y] = self.get_reward_by_action(x, y, self.policy[x][y]) + self._lambda * self.states[next_x][next_y]

                # 累计误差值
                tolerances_sum += new_states[x][y] - self.states[x][y]

        # 更新状态值函数
        self.states = new_states
        return tolerances_sum

    def is_crossing_boundaries(self, x, y):
        """
        判断点是否超界
        :param x: x
        :param y: y
        :return: bool
        """
        return x < 0 or y < 0 or x >= self.n or y >= self.m

    def show(self):

        print("迷宫规模如下所示")
        self.debug(self.chessboard)

        print("状态行动值矩阵如下")
        self.debug(self.state_action_values)
        print("最终决策如下所示")
        for x in range(self.n):
            for y in range(self.m):
                print(self.s[self.policy[x][y]], end="\t")
            print()

    def get_optimal_action(self, x, y):
        """
        用于寻找指定状态的最优行动
        :param x: x
        :param y: y
        :return: action
        """
        max_index = 0
        for i in range(self.sum_action):
            if self.state_action_values[x][y][i] > self.state_action_values[x][y][max_index]:
                max_index = i
        return max_index

    def get_action_by_probability(self, x, y):
        """
        依照概率为状态x，y选择一个行动
        :param x: x
        :param y: y
        :return: action
        """
        action_list = range(self.sum_action)
        a = random.choices(action_list, self.state_action_probabilities[x][y], k=1)[0]
        return a

    def update_state(self, x, y):
        """
        更新状态
        :param x: x
        :param y: y
        :return: None
        """
        ## 状态更新加权求和
        self.states[x][y] = 0

        # 遍历行动
        for i in range(self.sum_action):
            self.states[x][y] += self.state_action_probabilities[x][y][i] * self.state_action_values[x][y][i]

    def update_probability(self, x, y):
        """
        更新指定节点的最大值概率
        :param x:x
        :param y:y
        :return:None
        """
        # 找最大值位置
        max_index = self.get_optimal_action(x, y)

        # 更新最大概率策略
        self.state_action_probabilities[x][y] = [self.varepsilon / self.sum_action] * self.sum_action
        self.state_action_probabilities[x][y][max_index] = 1 - (self.sum_action - 1) * self.varepsilon / self.sum_action

    def update_policy(self):
        """
        以最大概率决定最优策略
        :return: None
        """
        for i in range(self.n):
            for j in range(self.m):
                self.policy[i][j] = self.get_optimal_action(i, j)

    def __init__(self):
        # 初始化迷宫规模
        self.n = len(self.chessboard)
        self.m = len(self.chessboard[0][0])

        # 初始化张量
        self.init_matrix()

# Sarsa算法实现
class Sarsa(Board):
    # 迭代次数
    episode_count = 100000
    # 这个系数表示系统应该给误差的关注度,值域需要在[0,1] 且值要小一点
    alpha = 0.1
    # n_step sarsa 算法的参数，用于定义n_step的长度
    n_step = 5
    # n_step 算法的迭代次数
    n_step_count = 100000

    def collect_experience(self):
        """
        随机生成一个样例
        :return: (s_t，a_t,r_{t+1}, s_{t+1},a_{t+1})
        """
        # 生成起始点
        x = random.randint(0, self.n - 1)
        y = random.randint(0, self.m - 1)

        # 随机生成一个行动
        a = self.get_action_by_probability(x, y)

        # 行动回报
        reward = self.get_reward_by_action(x, y, a)

        # 计算出下一个状态
        next_x, next_y = self.next_point(x, y, a)
        if self.is_crossing_boundaries(next_x, next_y): # 越界了就保持不动
            next_x, next_y = x, y

        # 随机生成一个下一个状态的行动
        next_a = self.get_action_by_probability(next_x, next_y)

        return (x, y), a, reward, (next_x, next_y), next_a



    def run_expected_sarsa(self):
        # 迭代
        for episode in range(self.episode_count):
            # 随机生成一个样例
            (x, y), a, r, (next_x, next_y), _ = self.collect_experience()

            ## 更新测量值
            self.state_action_values[x][y][a] = self.state_action_values[x][y][a] - self.alpha * (self.state_action_values[x][y][a] - (r + self._lambda * self.states[next_x][next_y]))

            ## 更新策略
            self.update_probability(x, y)

            ## 更新状态值
            self.update_state(x, y)

        # 以最大概率决定最优策略
        for i in range(self.n):
            for j in range(self.m):
                self.policy[i][j] = self.get_optimal_action(i, j)
        print("expected sarsa 算法实现")
        self.show()

    def get_n_step_track(self):
        """
        n_step sarsa 的辅助函数
        :return: (x, y), a, E
        """
        # 随机选择初始节点
        x = random.randint(0, self.n - 1)
        y = random.randint(0, self.m - 1)

        action_list = range(self.sum_action)
        a = random.choices(action_list, self.state_action_probabilities[x][y], k=1)[0]

        # 数值初始化
        E = 0
        now_lambda = 1
        x1, y1, a1 = x, y, 0

        for _ in range(self.n_step):
            # 随机生成行动以及行动到达的状态
            a1 = self.get_action_by_probability(x1, y1)
            next_x, next_y = self.next_point(x1, y1, a1)
            r = self.get_reward_by_action(next_x, next_y, a1)

            # 越界则保持不动
            if self.is_crossing_boundaries(next_x, next_y):
                next_x, next_y = x1, y1

            # 累加
            E += now_lambda * r

            # 更新数值以及行动
            now_lambda *= self._lambda
            x1, y1 = next_x, next_y

        # 完成累加
        E += now_lambda * self.state_action_values[x][y][a1]

        return (x, y), a, E

    def run_n_step_sarsa(self):
        """
        n_step_sarsa 算法实现，
        该算法是一种在蒙特卡洛算法和sarsa算法的一种折中
        所以它既不属于在线算法也不属于离线算法
        :return: None
        """
        # 迭代
        for episode in range(self.n_step_count):

            # 随机生成一个样例
            (x, y), a, E = self.get_n_step_track()

            ## 更新测量值
            self.state_action_values[x][y][a] = self.state_action_values[x][y][a] - self.alpha * ( self.state_action_values[x][y][a] - E)

            ## 更新策略
            self.update_probability(x, y)


        # 以最大概率决定最优策略
        for i in range(self.n):
            for j in range(self.m):
                self.policy[i][j] = self.get_optimal_action(i, j)
        print("n_step sarsa 算法实现")
        self.show()


    def run_Sarsa(self):
        # 迭代
        for episode in range(self.episode_count):
            # 随机生成一个样例
            (x, y), a, r, (next_x, next_y), next_a = self.collect_experience()

            ## 更新测量值
            self.state_action_values[x][y][a] = self.state_action_values[x][y][a] - self.alpha * (self.state_action_values[x][y][a] - (r + self._lambda * self.state_action_values[next_x][next_y][next_a]))

            ## 更新策略
            self.update_probability(x, y)

        # 得出最优策略
        self.update_policy()

        print("Sarsa 算法实现")
        self.show()

    def __init__(self):
        super().__init__()

if __name__ == '__main__':
    sarse = Sarsa()
    sarse.run_n_step_sarsa()