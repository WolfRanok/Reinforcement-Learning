"""
以下代码用最优贝尔曼求解最优路径
"""

class Behrman():
    ## 属性
    # 分别表示棋盘的长和宽
    n, m = None, None
    # 最低容忍误差（迭代终止条件）
    tolerances = 1e-15


    # 行动
    xx = [0, 0, 0, -1, 1]
    yy = [0, -1, 1, 0, 0]
    s = "O←→↑↓"

    # 参数
    _lambda = 0.9

    ## 矩阵
    # 棋盘地图矩阵
    chessboard = [["*#*#*****"],
                  ["*#*#*#*#*"],
                  ["*#*#x#*#*"],
                  ["*#*###*#*"],
                  ["*#*****#*"],
                  ["*#######*"],
                  ["*********"]]
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

    def action(self, x, y, i):
        """
         返回某一行动的奖惩值
        :param x: x坐标
        :param y: y坐标
        :param i: 行动
        :return: 奖惩值
        """
        next_x, next_y = self.next_point(x, y, i)
        return self.Punishment[next_x][next_y]

    def next_point(self, x, y, i):
        x += self.xx[i]
        y += self.yy[i]
        return x, y

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
                    self.Punishment[i][j] = -10
                else:
                    self.Punishment[i][j] = 1
        # print(self.debug(self.Punishment))
        return self.Punishment

    def get_policy(self):
        """
        更新策略，即找到每一个点的最优行动
        :return: None
        """
        # 遍历所有状态
        for x in range(self.n):
            for y in range(self.m):
                # 遍历行动
                state_max_idx = 0
                state_max_value = -10000000

                # 遍历5种行动
                for i in range(5):
                    next_x, next_y = self.next_point(x, y, i)
                    if next_x >= self.n or next_x < 0 or next_y >= self.m or next_y < 0:
                        continue

                    # 当前行动可以得到的状态值
                    state_value = self.action(x, y, i) + self._lambda * self.states[next_x][next_y]

                    if state_value > state_max_value:
                        state_max_value = state_value
                        state_max_idx = i
                # 更新最优行动
                self.policy[x][y] = state_max_idx

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
    @staticmethod
    def debug(lit):
        for x in lit:
            for y in x:
                print(y, end="\t\t")
            print()

    def show(self):
        for x in range(self.n):
            for y in range(self.m):
                print(self.s[self.policy[x][y]], end="\t")
            print()

    def run_value_iteration(self):
        """
        值迭代，算法执行
        状态和策略一起迭代优化
        :return:None
        """
        count = 0
        error = 1

        while error > self.tolerances:
            # 更新策略
            self.get_policy()

            # 更新状态并记录误差
            error = self.get_states()
            count += 1
        print("最优贝尔曼算法(值迭代)实现\n迷宫如下")
        self.debug(behrman.chessboard)
        print(f'一共迭代了{count}得到结果,最终策略如下')
        self.show()

    def run_policy_iteration(self):
        """
        策略迭代，算法执行
        先迭代完状态，再计算最优策略
        :return:None
        """
        count = 0

        print("最优贝尔曼算法(策略迭代)实现\n迷宫如下")
        self.debug(behrman.chessboard)
        print("*"*16)

        while True:
            count += 1
            count_now = 0

            # Policy evaluation
            while self.get_states() > self.tolerances or count_now + count == 1:    # count_now + count 这一步是为了保证第一次可以执行
                count_now += 1

            if count_now == 0:
                print(f"策略迭代算法在{count-1}次迭代后收敛")
                break
            else:
                print(f"第{count}次迭代，状态在{count_now}次内部迭代之后收敛")

            # 更新策略
            self.get_policy()

        print("最终策略如下")
        self.show()

    def run_truncated_policy_iteration(self, truncated_count=10):
        """
        截断式迭代算法，算法实现
        介于值迭代和策略迭代之间从一种算法，在做状态迭代时按照指定的迭代次数。
        :param truncated_count: 指定迭代次数，默认为10
        :return: None
        """
        count = 0
        flag = True

        while flag:
            # 指定做truncated_count次的状态优化
            for i in range(truncated_count):
                # 达到停止条件
                if i == 1 and self.get_states() < self.tolerances and count > 0:
                    flag = False
                    break
                else:
                    self.get_states()
            # 更新策略
            self.get_policy()
            count += 1

        print("最优贝尔曼算法(截断式策略迭代)实现\n迷宫如下")
        self.debug(behrman.chessboard)
        print("*" * 16)
        print(f"算法在迭代{count}次后收敛")
        print("最终策略如下")
        self.show()

    def __init__(self):
        # 计算长宽
        self.n = len(self.chessboard)
        self.m = len(self.chessboard[0][0])
        print(self.n, self.m)

        # 矩阵初始化
        self.init_matrix()


if __name__ == '__main__':
    behrman = Behrman()
    behrman.run_truncated_policy_iteration()
