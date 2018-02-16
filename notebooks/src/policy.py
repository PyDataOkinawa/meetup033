import numpy as np

class Policy():
    def select(self, actions):
        raise NotImplementedError()


class Greedy(Policy):
    def select(self, actions):
        return np.argmax(actions)


class EpsGreedy(Policy):
    def __init__(self, eps=0.1):
        self.eps = eps

    def select(self, actions):
        # ランダムで出した値よりepsのほうが大きいなら
        # ランダムに腕を選択する
        if np.random.uniform() < self.eps:
            return np.random.randint(0, len(actions))

        # greedy
        return np.argmax(actions)
