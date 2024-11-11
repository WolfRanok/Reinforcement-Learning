import random
from copy import deepcopy as cp
from DQN_network import *


class Board:
    """
    这是有关迷宫的基本设置父类
    """
    ## 属性
    # 分别表示棋盘的长和宽
    n, m = None, None
    # 惩罚值
    penalty_value = -100
    # 奖励值
    prize_value = 10
    # 每个状态的行动数
    sum_action = 5
    # 参数，这个值越大越远视，越小越近视
    _lambda = 0.97
    # 该值应该的值域：[0,1]，用于表示非最优策略的概率占比大小，可以视作探索性的大小
    varepsilon = 0.3
    # 行动
    xx = [0, -1, 1, 0, 0]
    yy = [0, 0, 0, -1, 1]
    s = "O↑↓←→"

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
                new_states[x][y] = self.get_reward_by_action(x, y, self.policy[x][y]) + self._lambda * \
                                   self.states[next_x][next_y]

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

    def show(self, text):
        """
        可视化
        :param text: 文本
        :return:
        """
        # 更新策略
        self.update_policy()
        print(text)
        print("迷宫规模如下所示")
        self.debug(self.chessboard)

        # print("状态行动值矩阵如下")
        # self.debug(self.state_action_values)
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


class DNQ(Board):
    # 迭代次数
    max_steps = 1000
    # 路径长度
    track_length = 500
    # 模型
    main_network = None
    target_network = None
    def get_max_index(self, x, y):
        """
        获取状态s的最优行动
        :param x: x
        :param y: y
        :return: max_index
        """
        max_index = 0
        for i in range(self.sum_action):
            if predict([x, y, i],self.target_network)[0] > predict([x, y, max_index],self.target_network)[0]:
                max_index = i
        return max_index

    def get_action_by_probability(self, x, y):
        """
        依照概率给出行动
        :param x: x
        :param y: y
        :return: action
        """
        action_list = range(self.sum_action)
        probabilities = [self.varepsilon/self.sum_action] * self.sum_action
        max_index = self.get_max_index(x, y)
        probabilities[max_index] = 1 - (self.sum_action - 1) * self.varepsilon/self.sum_action

        a = random.choices(action_list, probabilities, k=1)[0]
        return a

    def get_track(self):
        """
        随机生成一条探索路径
        :return: T：[(s,a,r,s')...]
        """
        # 初始化路径
        T = []
        # 随机创造一个初始点
        x = random.randint(0, self.n - 1)
        y = random.randint(0, self.m - 1)
        for i in range(self.track_length):
            # 依照概率生成一个行动
            a = self.get_action_by_probability(x, y)
            r = self.get_reward_by_action(x, y, a)
            next_x, next_y = self.next_point(x, y, a)

            # 越界
            if self.is_crossing_boundaries(next_x, next_y):
                next_x, next_y = x, y

            # 更新
            T.append([(x, y), a, r, (next_x, next_y)])
            x, y = next_x, next_y

        return T

    def update_policy(self):
        """
        更新整体策略
        :return: None
        """
        for x in range(self.n):
            for y in range(self.m):
                self.policy[x][y] = self.get_max_index(x, y)

    def get_max_value(self, x, y):
        """
        获得状态s的最大行动期望
        :param x: x
        :param y: y
        :return: q_max
        """
        q_max = predict([x, y, 0], self.target_network)[0]
        action_list = range(1, self.sum_action)
        for a in action_list:
            q_max = max(predict([x, y, a], self.target_network)[0], q_max)
        return q_max

    def update_all_state_action_values(self):
        """
        更新所有行动状态
        :return: None
        """
        # 制作输入
        date_input = []
        for x in range(self.n):
            for y in range(self.m):
                for a in range(self.sum_action):
                    date_input.append([x, y, a])
        new_values_list = predict(date_input, self.target_network)

        # 更新
        top = 0
        for x in range(self.n):
            for y in range(self.m):
                for a in range(self.sum_action):
                    self.state_action_values[x][y][a] = new_values_list[top][0]
                    top += 1


    def run_DQN(self):
        """
        Deep Q learning (on policy 版本)算法实现
        :return: None
        """
        # 迭代
        for epoch in range(self.max_steps):
            # 创建路径
            T = self.get_track()

            # 初始化输入输出
            date_input = []
            date_output = []

            for (x, y), a, r, (next_x, next_y) in T:
                # 计算返回值（目标值）
                y_hat = r + self._lambda * self.get_max_value(next_x, next_y)

                # 整理输入输出
                date_input.append([x, y, a])
                date_output.append([y_hat])

            # 训练
            train_network(self.main_network, date_input, date_output)
            if epoch % 10 == 9:
                print(f"已迭代{epoch+1}轮")
                # 更新行动状态值
                self.update_all_state_action_values()
                self.debug(self.state_action_values)
            copy_model(self.target_network, self.main_network)

        # 可视化
        self.show("Deep Q learning (on policy 版本)算法实现")

    def __init__(self):
        super().__init__()

        # 初始化模型，并统一参数
        self.main_network = DeepQNetwork()
        self.target_network = DeepQNetwork()
        copy_model(self.target_network, self.main_network)

if __name__ == '__main__':
    dnq = DNQ()
    dnq.run_DQN()