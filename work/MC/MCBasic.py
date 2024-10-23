import random
"""
该脚本用基础蒙特卡洛算法，解迷宫寻址问题
"""


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
    penalty_value = -10

    # 行动
    xx = [0, 0, 0, -1, 1]
    yy = [0, -1, 1, 0, 0]
    s = "O←→↑↓"

    # 参数
    _lambda = 0.9

    ## 矩阵
    # 棋盘地图矩阵
    chessboard = [["*#***"],
                  ["*#*#*"],
                  ["*#x#*"],
                  ["*###*"],
                  ["*****"],]
    # 惩罚值矩阵
    Punishment = None
    # 策略矩阵，一共有n*m个状态，这里将其放到二维张量中
    policy = None
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
                    self.Punishment[i][j] = 1
        return self.Punishment

    @staticmethod
    def debug(lit):
        print("*" * 20)
        for x in lit:
            for y in x:
                print(y, end="\t\t")
            print()

    def action(self, x, y, i):
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
                new_states[x][y] = self.action(x, y, self.policy[x][y]) + self._lambda * self.states[next_x][next_y]

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
        for x in range(self.n):
            for y in range(self.m):
                print(self.s[self.policy[x][y]], end="\t")
            print()


## 基础蒙特卡洛算法实现
class MCBasic(Board):
    ## 属性
    # 最大迭代次数
    max_iteration = 50

    # 超界返回
    crossing_boundaries = -10000000

    def get_reward(self, x, y, i):
        """
        迭代寻址计算在点(x,y)下该i策略下的能得到的奖励
        :param x: x
        :param y: y
        :param i: 策略
        :return: int
        """

        # 初始化
        sum_reward = 0
        now_lambda = 1

        # 迭代max_iteration次
        for _ in range(self.max_iteration):
            ## 贝尔曼公式解reward
            next_x, next_y = self.next_point(x, y, i)

            # 越界返回
            if self.is_crossing_boundaries(next_x, next_y):
                return self.crossing_boundaries

            # 当前行动的reward
            reward = self.action(x, y, i)
            # 累计reward,更新参数
            sum_reward += now_lambda * (reward + self.states[next_x][next_y] * self._lambda)
            i = self.policy[next_x][next_y] # 更新决策
            now_lambda *= self._lambda
            x, y = next_x, next_y

        return sum_reward

    def MC_policy(self):
        """
        蒙特卡洛核心思想实现
        使用统计的方法来计算出该选择的价值
        :return: None
        """

        # 初始化新策略变量
        new_policy = [[0 for _ in range(self.m)] for _ in range(self.n)]

        # 遍历每一个点的找到该点的最优策略
        for x in range(self.n):
            for y in range(self.m):
                # 初始化参数
                max_value = -10000000
                optimal_action = 0

                # 遍历5种策略
                for i in range(5):
                    # 计算当前点的行动i的回报
                    now_value = self.get_reward(x, y, i)
                    if now_value > max_value:
                        max_value = now_value
                        optimal_action = i

                # 当前点的策略更新
                new_policy[x][y] = optimal_action
        # 策略更新
        self.policy = new_policy

    def run(self):
        self.MC_policy()
        count = 0
        while self.get_states() > self.tolerances:
            # 更新策略
            self.MC_policy()

            count += 1
        print("基础蒙特卡洛算法实现\n迷宫布局如下所示")
        self.debug(self.chessboard)
        print("*" * 20)
        print(f"经过{count}次迭代，结束算法，最终决策如下")
        self.show()


    def __init__(self):
        # 计算地图的长度和宽度
        self.n, self.m = len(self.chessboard), len(self.chessboard[0][0])
        # 初始化矩阵
        self.init_matrix()


## 优化后的蒙特卡洛算法
# 优化1
# 每次的路径上的策略都会一一使用到，不会出现浪费
#
# 优化2
# 每一个状态下的行动不唯一，概率大小由行动的回报决定

class MCE_Greedy(Board):
    ## 属性
    # 该值应该的值域：[0,1]，用于表示非最优策略的概率占比大小
    varepsilon = 0.4
    # 每个状态的行动数
    sum_actions = 5
    # 定义迭代的次数
    iteration_count = 10000
    # 决策轨迹的长度
    track_length = 100

    ## 矩阵
    # 策略概率矩阵矩阵（三维张量）
    states_action_probabilities = None
    # 每个策略的回报矩阵（三维张量）
    states_action_values = None

    def init_MCE_Greedy__matrix(self):
        """
        自定义的矩阵初始化，除了常规的初始化之外还有对概率从初始化
        :return: None
        """
        self.init_matrix()

        # 这里定义第一个策略概率最大
        action_probabilities = [1-(self.sum_actions-1)*self.varepsilon/self.sum_actions, self.varepsilon/self.sum_actions, self.varepsilon/self.sum_actions, self.varepsilon/self.sum_actions, self.varepsilon/self.sum_actions]
        self.states_action_probabilities = [[action_probabilities for _ in range(self.m)] for _ in range(self.n)]

        # 初始化回报
        action_values = [0, 0, 0, 0, 0]
        self.states_action_values = [[action_values for _ in range(self.m)] for _ in range(self.n)]

    def get_action(self, x, y):
        """
        在指定的状态中按概率随机选择一个行动
        :param x: x
        :param y: y
        :return: action
        """
        # 这里列出所有的行动
        action_list = list(range(self.sum_actions))

        # 按概率随机选择
        action = random.choices(action_list, self.states_action_probabilities[x][y], k=1)[0]
        return action

    def get_start_point(self):
        """
        随机创建一个起点
        :return: (x,y)
        """
        start_point = random.choice(range(self.n * self.m))

        return start_point // self.m, start_point % self.m

    def get_track(self):
        """
        按照概率生成一条轨道，轨道内容包括（状态、行动、reward）
        :return: [(s,a,r)...]
        """
        # 轨迹
        T = []

        # 创建起点
        x, y = self.get_start_point()

        for _ in range(self.track_length):
            # 随机选择行动，获得reward
            action = self.get_action(x, y)
            reward = self.action(x, y, action)

            # 累加
            T.append(((x, y), action, reward))

            next_x, next_y = self.next_point(x, y, action)
            if not self.is_crossing_boundaries(next_x, next_y):     # 没越界，更新位置，否则不动
                x = next_x
                y = next_y

        # debug功能函数
        def debug_track(T):
            for (x,y), action, reward in T:
                print(f"s: ({x},{y}), a:{self.s[action]}, r:{reward}")
        # # 测试
        # debug_track(T)
        return T



    def run(self):
        """
        优化后的蒙特卡洛算法实现
        :return: None
        """

        # 该算法需要迭代iteration_count次
        for i in range(self.iteration_count):
            # 获取决策轨迹
            T = self.get_track()
            g = 0
            Hash = set()    # 定义空集合

            j = 0
            for (x, y), action, reward in reversed(T):  # 倒序遍历
                g = g * self._lambda + reward
                j += 1
                if j < 50:  # 迭代长度不够的不进行更新
                    continue

                if ((x, y), action) not in Hash:
                    Hash.add(((x, y), action))
                    self.states_action_values[x][y][action] = g # 只更新行动的v值，不需要更新状态值

                    # 找到最优行动决策
                    max_index = 0
                    for j in range(self.sum_actions):
                        if self.states_action_values[x][y][j] > self.states_action_values[x][y][max_index]:
                            max_index = j

                    # 更新概率
                    self.states_action_probabilities[x][y] = [self.varepsilon / self.sum_actions] * 5
                    self.states_action_probabilities[x][y][max_index] = 1 - (self.sum_actions - 1) * self.varepsilon / self.sum_actions

        # 这里定义每个状态的最优概率值作为最优决策
        for x in range(self.n):
            for y in range(self.m):

                # 找最优决策
                i = 0
                for j in range(self.sum_actions):
                    if self.states_action_probabilities[x][y][j] > self.states_action_probabilities[x][y][i]:
                        i = j

                self.policy[x][y] = i

        print("优化版蒙特卡洛算法实现，以下为棋盘结构")
        self.debug(self.chessboard)
        print("*"*20)
        print(f"迭代{self.iteration_count}次后的最优决策如下")
        self.show()

    def __init__(self):
        # 计算迷宫的长和宽
        self.n, self.m = len(self.chessboard), len(self.chessboard[0][0])

        # 初始化矩阵
        self.init_MCE_Greedy__matrix()
        # self.debug(self.states_action_probabilities)

if __name__ == '__main__':
    mce = MCE_Greedy()
    mce.run()
