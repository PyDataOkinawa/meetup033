import numpy as np
from policy import Greedy


class Agent():
    def __init__(self):
        pass

    def get_action(self, state):
        raise NotImplementedError()

    def update(self, **kwargs):
        raise NotImplementedError()


class Sarsa(Agent):
    def __init__(self, table_shape, policy=None, lr=0.01, gamma=0.99):
        self.q_table = np.zeros(table_shape)
        self.policy = Greedy() if policy is None else policy
        self.lr = lr
        self.gamma = gamma

    def get_action(self, state):
        return int(self.policy.select(self.q_table[state]))

    def update(self, **kwargs):
        state = kwargs.pop('state', 0) 
        action = kwargs.pop('action', 0)
        next_state = kwargs.pop('next_state', 0)
        reward = kwargs.pop('reward', 0)

        # kwargsが残っている場合はいらないものがあるはずなのでエラー
        if kwargs: raise ValueError()

        next_action = np.argmax(self.q_table[next_state])

        l = (1 - self.lr) * self.q_table[state, action]
        r = self.lr * (reward + self.gamma * self.q_table[next_state, next_action])
        self.q_table[state][action] = l + r
