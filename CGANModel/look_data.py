"""
@Time ： 2021/5/25 11:09
@Auth ： Lewis
@File ：look_data.py
@IDE ：PyCharm
"""
from utils import *

s_, s, a = load_pong()


def get_reward(state):
    # state 是4张的处理过的01图片
    obs = np.transpose(state, (2, 0, 1))
    R = 0
    for idx in range(4):
        obs_ = obs[idx]
        for i in range(84):
            for j in range(8):
                if obs_[i][j] == 1:
                    R += 1
        for i in range(84):
            for j in range(76, 84):
                if obs_[i][j] == 1:
                    R -= 1
    if R > 0:
        R = 1
    elif R < 0:
        R = -1
    return R

print(len(s_))
num = 0
for i in s_:
    if get_reward(i) != 0:
        num += 1

print(num / len(s_))
