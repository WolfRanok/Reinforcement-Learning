"""
以下代码用最优贝尔曼求解最优路径
"""

class Behrman():
    ## 属性
    # 分别表示棋盘的长和宽
    n, m = None, None
    # 最低容忍误差（迭代终止条件）
    tolerances = 0.000001

    # 行动
    xx = [0, 0, 0, -1, 1]
    yy = [0, -1, 1, 0, 0]
    s = "O←→↑↓"

    # 参数
    _lambda = 0.5

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
        :return:
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
        # self.debug(self.states)
        # print("-----")
        # self.debug(new_states)
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

    def run(self):
        """
        算法执行
        :return:None
        """
        count = 0
        self.get_policy()
        while self.get_states() > self.tolerances:
            # 更新策略
            self.get_policy()
            count += 1
        print("最优贝尔曼算法实现\n迷宫如下")
        self.debug(behrman.chessboard)
        print(f'一共迭代了{count}得到结果,最终策略如下')
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
    behrman.run()
