"""
该脚本用于实现Q-learning 算法
该算法可以是on-policy 也可以是off-policy 算法
已下提供两种算法的实现方式
"""
import random
from Sarsa import Board
from copy import deepcopy as cp

class QLearningAgent(Board):
    ## 属性
    # 算法迭代次数
    QLearningAgent_count = 10000
    # 这个系数表示系统应该给误差的关注度,值域需要在[0,1] 且值要小一点
    alpha = 0.1
    # 序列长度
    epsilon_length = 50

    def get_on_policy_example(self):
        """
        配合on_policy Q-learning算法的辅助函数
        :return: s_t, a_t, E(r_{t+1}+q*)
        """
        # 随机产生一个点
        x = random.randint(0, self.n - 1)
        y = random.randint(0, self.m - 1)
        a = self.get_action_by_probability(x, y)
        r = self.get_reward_by_action(x, y, a)

        next_x, next_y = self.next_point(x, y, a)
        if self.is_crossing_boundaries(next_x, next_y): # 越界的处理
            next_x, next_y = x, y

        max_index = self.get_optimal_action(next_x, next_y)

        E = r + self._lambda * self.state_action_values[next_x][next_y][max_index]

        return (x, y), a, E


    def run_on_policy_Q_learning(self):
        """
        on_policy_Q_learning 算法实现
        :return: None
        """
        # 迭代
        for i in range(self.QLearningAgent_count):
            # 获得实例
            (x, y), a, q = self.get_on_policy_example()

            ## 更新策略值
            self.state_action_values[x][y][a] = self.state_action_values[x][y][a] - self.alpha * (self.state_action_values[x][y][a] - q)

            ## 更新策略
            self.update_probability(x, y)

        # 得到最优策略
        self.update_policy()

        print("on_policy_Q_learning算法实现")
        self.show()

    def get_off_policy_example(self):
        """
        配合off_policy Q-learning算法的辅助函数
        :return:{[s_0, a_0, E_0(r_1 ,s1)],...}
        """
        # 初始化序列
        T = []

        # 随机生成起始点
        x = random.randint(0, self.n - 1)
        y = random.randint(0, self.m - 1)

        for _ in range(self.epsilon_length):
            a = self.get_action_by_probability(x, y)

            next_x, next_y = self.next_point(x, y, a)
            # 越界处理
            if self.is_crossing_boundaries(next_x, next_y):
                next_x, next_y = x, y

            # 回报值
            r = self.get_reward_by_action(x, y, a)

            # 最优行动
            max_index = self.get_optimal_action(next_x, next_y)

            # 获取期望
            E = r + self._lambda * self.state_action_values[next_x][next_y][max_index]

            # 更新
            T.append(((x, y), a, E))
            x, y = next_x, next_y

        return T



    def run_off_policy_Q_learning(self):
        """
        off_policy_Q_learning算法实现
        :return: None
        """
        # 迭代
        for _ in range(self.QLearningAgent_count):
            # 获取样例
            T = self.get_off_policy_example()
            new_state_action_values = cp(self.state_action_values)
            for (x, y), a, E in T:
                ## 更新策略值
                new_state_action_values[x][y][a] = self.state_action_values[x][y][a] - self.alpha * (self.state_action_values[x][y][a] - E)
            # 一轮更新结束
            self.state_action_values = new_state_action_values

        # 更新最终策略
        self.update_policy()
        print("off_policy Q-learning算法实现如下")
        self.show()

    def __init__(self):
        super().__init__()

if __name__ == '__main__':
    q = QLearningAgent()
    q.run_off_policy_Q_learning()
