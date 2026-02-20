import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd

'''
1. Регрессия
Построить регрессию используя аналитическое решение через псевдообратную матрицу Мура-Пенроза. Данные задать следующим образом:

N=1000
x = np.linspace(0, 1, N)
z = 20*np.sin(2*np.pi * 3 * x) + 100*np.exp(x)
error = 10 * np.random.randn(N)
t = z + error

В качестве базисных функций использовать полиномы до степени M.

- В одном графическом окне построить график функции z(x) в виде непрерывной кривой, t(x) в виде точек и решения задачи регрессии в виде непрерывной кривой для M = 1
- В одном графическом окне построить график функции z(x) в виде непрерывной кривой, t(x) в виде точек и решения задачи регрессии в виде непрерывной кривой для M = 8
- В одном графическом окне построить график функции z(x) в виде непрерывной кривой, t(x) в виде точек и решения задачи регрессии в виде непрерывной кривой для M = 100
- Построить график зависимости ошибки E(w) от степени полинома M. M меняется от 1 до 100.
'''

'''
Y = X * w
w = F+ * Y

F+ = inv(transpose(F) * F) * transpose(F) # Псевдо обратная матрица
E(w)=sum((ti-yi)^2) / N
error = np.sum((t - Y) ** 2) / N
    
'''

def demo():
    X = np.array([1,3,9])
    Y = np.array([2,4,5])
    M = 3

    # Матрица плана
    F = [
        [1, 1, 1],
        [1, 3, 9],
        [1, 9, 81],
    ]
    # Примеры из обучающей выборки
    #t = F*w
    F_plus = np.linalg.inv(X.T*X)*X.T
    w = F_plus * Y
    N = 1000
    x = np.linspace(0, 1, N)
    z = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)

def calculateY(x,M,t):
    F = []
    for i in range(0, M + 1):
        F.append(np.pow(x, i))
    F = np.array(F).T

    F_plus = np.linalg.inv(F.T @ F) @ F.T
    w = F_plus @ t
    Y = F @ w
    return Y
def HW1():
    print("start")
    N = 1000 # кол-во элементов в обучающей выборке
    x = np.linspace(0, 1, N)
    z = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)
    error = 10 * np.random.randn(N)
    t = z + error

    # M - макс степень полинома
    fig, axes = plt.subplots(2, 2)

    # 1 В одном графическом окне построить график функции z(x) в виде непрерывной кривой, t(x) в виде точек и решения задачи регрессии в виде непрерывной кривой для M = 1
    M = 1
    Y1 = calculateY(x,M,t)

    axes[0, 0].plot(x, z, 'g-', linewidth=2, label='z(x)')
    axes[0, 0].scatter(x, t, c='blue', s=10, alpha=0.5, label='t(x)')
    axes[0, 0].plot(x, Y1, 'r--', linewidth=2, label=f'M=1')
    axes[0, 0].set_xlabel('Значения x')
    axes[0, 0].set_ylabel('Значения функций')
    axes[0, 0].set_title(f'M = 1')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 2 В одном графическом окне построить график функции z(x) в виде непрерывной кривой, t(x) в виде точек и решения задачи регрессии в виде непрерывной кривой для M = 8
    M = 8
    Y2 = calculateY(x,M,t)

    axes[0, 1].plot(x, z, 'g-', linewidth=2, label='z(x)')
    axes[0, 1].scatter(x, t, c='blue', s=10, alpha=0.5, label='t(x)')
    axes[0, 1].plot(x, Y2, 'r--', linewidth=2, label=f'M=8')
    axes[0, 1].set_xlabel('Значения x')
    axes[0, 1].set_ylabel('Значения функций')
    axes[0, 1].set_title(f'M = 8')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 3 В одном графическом окне построить график функции z(x) в виде непрерывной кривой, t(x) в виде точек и решения задачи регрессии в виде непрерывной кривой для M = 100
    M = 100
    Y3 = calculateY(x,M,t)

    axes[1, 0].plot(x, z, 'g-', linewidth=2, label='z(x)')
    axes[1, 0].scatter(x, t, c='blue', s=10, alpha=0.5, label='t(x)')
    axes[1, 0].plot(x, Y3, 'r--', linewidth=2, label=f'M=100')
    axes[1, 0].set_xlabel('Значения x')
    axes[1, 0].set_ylabel('Значения функций')
    axes[1, 0].set_title(f'M = 100')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 4 Построить график зависимости ошибки E(w) от степени полинома M. M меняется от 1 до 100.
    # sum((ti-yi)^2) / N # средне квадратичная ошибка (Y)
    # 1..M # Степени полинома (X)

    E = []
    m = [i for i in range(1,101)]
    for M in m:
        Y = calculateY(x, M, t)
        E.append(np.sum((t - Y) ** 2) / N)

    axes[1, 1].plot(m, E, 'r--', linewidth=2, label=f'E(w)')
    axes[1, 1].set_xlabel('Степень полинома M')
    axes[1, 1].set_ylabel('Ошибка')
    axes[1, 1].set_title('Зависимость ошибки от M')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    plt.tight_layout()
    plt.show(block=True)
    print("done")

if __name__ == "__main__":
    HW1()
    # demo()