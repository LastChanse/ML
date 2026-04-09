from sklearn.datasets import load_boston
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

'''
4. Простые классификаторы и метрики.
* Построить данные, моделирующие измерение роста футболистов и баскетболистов (по 1000 на класс).
* Рост футболистов задать как нормальное распределение со средним mu_0 и стандартным отклонением sigma_0.
* Рост баскетболистов задать как нормальное распределение со средним mu_1 и стандартным отклонением sigma_1.
* Задать бинарный классификатор на основе порога (некоторое число).
* С помощью классификатора произвести классификацию спортсменов по их росту.
* Реализовать функции, которые по результатам классификации вычисляют метрики:
* TP, TN, FP, FN, Accuracy, Precision, Recall, F1-score, ошибки 1-го и 2-го рода (alpha, beta).

+ Постоить график ROC кривой для указанного классификатора:
+ Изменять порог от 0 до T
+ Найти площадь под построенной кривой (AUC) и вывести её на консоль
+ Найти значение порога, при котором достигается максимальное значение Accuracy и подсчитать для него основные метрики:
TP, TN, FP, FN, Accuracy, Precision, Recall, F1-score, ошибки 1-го и 2-го рода (alpha, beta). 
'''

def calculateAll(t,y,idx):
    y_pred = np.array(y)
    t_true = np.array([t[idx[i]] for i in range(len(t))])
    TP = np.sum((y_pred == 1) & (t_true == 1))
    TN = np.sum((y_pred == 0) & (t_true == 0))
    FP = np.sum((y_pred == 1) & (t_true == 0))
    FN = np.sum((y_pred == 0) & (t_true == 1))
    Accuracy = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN > 0 else 0
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1_score = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0
    alpha = FP / (FP + TN) if (FP + TN) > 0 else 0
    beta = FN / (FN + TP) if (FN + TP) > 0 else 0
    return TP, TN, FP, FN, Accuracy, Precision, Recall, F1_score, alpha, beta

def printAll(TP, TN, FP, FN, Accuracy, Precision, Recall, F1_score, alpha, beta):
    print(f"{TP=}\n{TN=}\n{FP=}\n{FN=}\n{Accuracy=}\n{Precision=}\n{Recall=}\n{F1_score=}\n{alpha=}\n{beta=}\n")

# Основная функция
def AUC(RecallMas, alphaMas):
    data = sorted(zip(alphaMas, RecallMas))
    data = sorted(zip(alphaMas, RecallMas))
    x, y = zip(*data)

    auc = 0.0
    for i in range(len(x) - 1):
        delta_x = x[i + 1] - x[i]
        avg_y = (y[i] + y[i + 1]) / 2
        auc += delta_x * avg_y
    return auc


def HW4():
    N = 1000
    # Футболисты
    mu_0 = 180   # Среднее
    sigma_0 = 10 # Стандартное отклонение
    rost_fut = np.random.normal(mu_0, sigma_0, N)
    # Баскетболисты
    mu_1 = 190   # Среднее
    sigma_1 = 14 # Стандартное отклонение
    rost_bas = np.random.normal(mu_1, sigma_1, N)

    # 0 футболист, 1 баскетболист
    def classificator(rost, T):
        return 0 if rost < T else 1

    X = np.append(rost_fut, rost_bas)
    t = [0]*len(rost_fut)+[1]*len(rost_bas)
    print(len(X), len(t))
    # перемешивание
    idx = np.random.permutation(N*2)
    T = int(max(X))
    TP, TN, FP, FN, Accuracy, Precision, Recall, F1_score, alpha, beta = None,None,None,None,None,None,None,None,None,None
    AccuracyBest = 0
    RecallMas = []
    alphaMas = []
    TMas = []
    for Tval in range(int(min(X)),T):
        y = []
        for i in idx:
            y.append(classificator(X[i],Tval))
        _TP, _TN, _FP, _FN, _Accuracy, _Precision, _Recall, _F1_score, _alpha, _beta = calculateAll(t,y,idx)
        RecallMas.append(_Recall)
        alphaMas.append(_alpha)
        TMas.append(Tval)
        printAll(TP, TN, FP, FN, Accuracy, Precision, Recall, F1_score, alpha, beta);

        if _Accuracy > AccuracyBest:
            AccuracyBest = _Accuracy
            TP, TN, FP, FN, Accuracy, Precision, Recall, F1_score, alpha, beta = _TP, _TN, _FP, _FN, _Accuracy, _Precision, _Recall, _F1_score, _alpha, _beta

    print("\n=== ЛУЧШАЯ МОДЕЛЬ ===")
    printAll(TP, TN, FP, FN, Accuracy, Precision, Recall, F1_score, alpha, beta);

    print("Площадь AUC:", AUC(RecallMas,alphaMas))

    # график ROC
    plt.plot(alphaMas,RecallMas)
    plt.title("ROC")
    plt.show()


if __name__ == "__main__":
    HW4()