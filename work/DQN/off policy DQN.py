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
    penalty_value = -500
    # 奖励值
    prize_value = 1000
    # 每个状态的行动数
    sum_action = 5
    # 参数，这个值越大越远视，越小越近视
    _lambda = 0.9
    # 该值应该的值域：[0,1]，用于表示非最优策略的概率占比大小，可以视作探索性的大小
    varepsilon = 0.4
    # 行动
    xx = [0, -1, 1, 0, 0]
    yy = [0, 0, 0, -1, 1]
    s = "O↑↓←→"

    ## 矩阵
    # 棋盘地图矩阵
    chessboard = [["***"],
                  ["*#*"],
                  ["*#X"],]
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

    def show(self):

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



class DQN(Board):
    ## 属性
    # 样本路径长度:self.n * self.m * self.sum_action * 1000
    sample_length = None

    # 进行目标网络更新时的频率，即梅更新多少次main_network再更新target_network
    update_target_network_num = 1000

    # 算法迭代次数，即总的target_network网络的更新次数
    iteration_count = 10

    ## 模型
    main_network = None
    target_network = None

    def get_track(self):
        """
        获取路径样本
        :return: [((s_x,s_y),a,r,(next_s_x,next_s_y)),..]
        """
        # 初始化返回值
        T = []

        # 随机生成一个起始点
        x = random.randint(0, self.n - 1)
        y = random.randint(0, self.m - 1)

        # 开始创建
        for _ in range(self.sample_length):
            # 依照概率获取行动和回报
            a = self.get_action_by_probability(x, y)
            r = self.get_reward_by_action(x, y, a)

            # 下一个点
            next_x, next_y = self.next_point(x, y, a)
            if self.is_crossing_boundaries(next_x, next_y):
                next_x, next_y = x, y

            T.append(((x, y), a, r, (next_x, next_y)))

            x, y = next_x, next_y

        return T

    def get_optimal_action(self, s_list):
        """
        专用于函数的最大期望索引寻找函数
        :param s_list: 所有状态的列表
        :return: max_index
        """
        # 初始化
        n = len(s_list)
        max_index_list = [[0] for _ in range(n)]
        # 得到状态s所有行动a的预测情况
        all_states_actions = []
        for x, y in s_list:
            for a in range(self.sum_action):
                all_states_actions.append([x, y, a])

        # 预测值
        all_value = predict(all_states_actions, self.target_network)

        for i in range(n):
            j = l = i * self.sum_action
            r = l + self.sum_action
            max_index = l
            while j < r:
                if all_value[j][0] > all_value[max_index][0]:
                    max_index = j
                j += 1
            max_index_list[i][0] = max_index - l

        return max_index_list

    def handle(self, t):
        """
        将数据处理成可以训练的形式，即：
        数据输入：[s_x,s_y,a]
        数据输出：[y^hat]
        :param t: 采集到的数据
        :return: None
        """

        # # 初始化其他参数
        # next_s_list = []
        # date_input = []
        # r_list = []
        # for (x, y), a, r, (next_x, next_y) in t:
        #     next_s_list.append([next_x, next_y])
        #     date_input.append([x, y, a])
        #     r_list.append([r])
        #
        # # 预测最大值索引
        # max_index_list = self.get_optimal_action(next_s_list)
        #
        # # 得到q_hat的输入
        # next_date_input = []
        # for (next_x, next_y), [max_index] in zip(next_s_list, max_index_list):
        #     next_date_input.append([next_x, next_y, max_index])
        #
        # # 贝尔曼公式求y，输出
        # q_max_list = predict(next_date_input, self.target_network)
        # date_output = r_list + self._lambda * q_max_list

        # 输入，输出
        date_input, date_output = [],[]
        for (x, y), a, r, (next_x, next_y) in t:
            # 计算出输出值
            max_index = self.get_optimal_action([[next_x, next_y]])[0][0] # 最优的行动
            q_max = predict([next_x, next_y, max_index], self.target_network)[0]
            # 贝尔曼公式求y
            yhat = r + self._lambda * q_max

            # 添加结果
            date_input.append([x, y, a])
            date_output.append([yhat])

        return date_input, date_output


    def randomly_select_data(self, T):
        """
        从数据集T中随机选择若干数据进行选择
        :param T: 数据集
        :return: 可训练数据输入、输出
        """
        t = random.sample(T, self.update_target_network_num)
        return self.handle(t)


    def update_policy(self):
        """
        专属于函数的模型策略更新方法
        :return: None
        """
        # 遍历状态
        all_states = [ [x, y] for y in range(self.m) for x in range(self.n)]
        max_index_list = self.get_optimal_action(all_states)
        top = 0
        for x in range(self.n):
            for y in range(self.m):
                self.policy[x][y] = max_index_list[top][0]
                top += 1

    def run_deep_q_learning(self):
        """
        Deep Q learning 算法实现（off policy版本）
        :return: None
        """
        # 数据采样(数据只用一次)
        T = self.get_track()

        self.debug(self.chessboard)

        for each in range(self.iteration_count):

            # 随机采样
            date_input, date_output = self.randomly_select_data(T)
            print(date_input[:5], date_output[:5])
            # return
            # 训练main_network
            train_network(self.main_network, date_input, date_output)

            copy_model(self.target_network, self.main_network)


        # 更新策略
        self.update_policy()
        print("Deep Q Learning 算法实现")
        self.show()

    def __init__(self):
        super().__init__()

        # 确保每一个状态的每一个行动都可以遍历到
        self.sample_length = self.n * self.m * self.sum_action * 1000

        # 初始化模型
        self.main_network = DeepQNetwork()
        self.target_network = DeepQNetwork()
        copy_model(self.target_network, self.main_network)

if __name__ == '__main__':
    dqn = DQN()
    dqn.run_deep_q_learning()