# -*- coding: utf-8 -*-
# @time       : 2019-10-08 14:32
# @author     : ssdcxy
# @email      : 18379190862@163.com
# @file       : HMM.py
# @description: 隐马尔可夫模型

# 隐藏状态为cold,hot; 观察状态为1，2，3
# 转移概率
a = [[0, 0.2, 0.8, 0],
     [0, 0.6, 0.4, 0],
     [0, 0.3, 0.7, 0],
     [0, 1, 1, 0]]
# 观察似然度
b = [[0, 0.5, 0.2, 0],
     [0, 0.4, 0.4, 0],
     [0, 0.1, 0.4, 0]]
# 观察序列
o = [3, 1, 3]
# 状态序列
q = ['[start]', 'cold', 'hot']


# 似然度问题
def forward(T, N):
    """
    前向算法
    :param T: 观察序列长度
    :param N: 状态数
    :return: 观察序列似然度
    """
    # 初始化
    fd = [[0 for j in range(T)]
          for i in range(N+2)]
    for s in range(1, N+1):
        fd[s][0] = a[0][s] * b[o[0]-1][s]
    # 递归
    for t in range(1, T):
        for s in range(1, N+1):
            for s1 in range(1, N+1):
                fd[s][t] += fd[s1][t-1] * a[s1][s] * b[o[t]-1][s]
    # 结束
    for s in range(1, N+1):
        fd[0][T-1] += fd[s][T-1] * a[N+1][s]
    return fd[0][T-1]


# 解码问题
def viterb(T, N):
    # 初始化
    vb = [[0 for j in range(T)]
          for i in range(N+2)]
    bp = [[0 for j in range(T)]
          for i in range(N+2)]
    for s in range(1, N+1):
        vb[s][0] = a[0][s] * b[o[0]-1][s]
    # 递归
    for t in range(1, T):
        for s in range(1, N+1):
            max = vb[1][t-1] * a[1][s] * b[o[t]-1][s]
            index = 1
            bp[s][t] = 1
            for s1 in range(1, N+1):
                if vb[s1][t-1] * a[s1][s] * b[o[t]-1][s] > max:
                    max = vb[s1][t-1] * a[s1][s] * b[o[t]-1][s]
                    index = s1
            vb[s][t] = max
            bp[s][t] = index
    max = vb[1][T-1] * a[0][1]
    index = 1
    # 结束
    for s in range(2, N+1):
        max = vb[s][T-1] * a[N+1][s]
        index = s
    vb[0][T-1] = max
    bp[0][T-1] = index
    # 反向追踪
    lst = []
    index = bp[0][T-1]
    lst.append(q[index])
    for t in range(T, 0, -1):
        index = bp[index][t-1]
        lst.append(q[index])
    return lst


if __name__ == "__main__":
    forward(3, 2)
    print(viterb(3, 2))
    print()
