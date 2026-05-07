from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import load_digits
import warnings
warnings.filterwarnings("ignore")
from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# Критерии качества #####################################################################
def entropy_criterion(y: np.ndarray) -> float:
    """
    Энтропия Шеннона
    H(S) = - Σ p_k * log2(p_k + ε)
    """
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-12))


def gini_criterion(y: np.ndarray) -> float:
    """
    БОНУС 1: Неопределённость Джини
    H(S_i) = 1 - Σ (N_i^K / N_i)^2
    """
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)


def misclassification_criterion(y: np.ndarray) -> float:
    """
    БОНУС 2: Ошибка классификации (misclassification error).
    H(S_i) = 1 - max_K (N_i^K / N_i)
    """
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.max(probs)


# Словарь критериев
CRITERION_DICT = {
    'entropy': entropy_criterion,
    'gini': gini_criterion,
    'misclassification': misclassification_criterion
}

def information_gain(y: np.ndarray, left_mask: np.ndarray, right_mask: np.ndarray,
                     criterion_func) -> float:
    """
    Прирост информации при разбиении родительского узла.
    IG = H(parent) - (|S_left|/|S|)*H(S_left) - (|S_right|/|S|)*H(S_right)
    Критерий H передаётся через criterion_func.
    """
    parent_impurity = criterion_func(y)
    n = len(y)
    n_left = np.sum(left_mask)
    n_right = np.sum(right_mask)

    # Если одно из подмножеств пустое, разбиение бессмысленно
    if n_left == 0 or n_right == 0:
        return 0.0

    # Примеси дочерних узлов
    child_impurity = (n_left / n) * criterion_func(y[left_mask]) + \
                     (n_right / n) * criterion_func(y[right_mask])
    return parent_impurity - child_impurity

# Бинарное дерево решений #####################################################################

class Node:
    """
    Узел дерева решений.
    feature  - индекс признака для разбиения (для осевых разбиений)
    weights  - веса признаков для косых гиперплоскостей (бонус 5)
    phi      - функция нелинейного преобразования (бонус 6)
    threshold - порог
    left, right - поддеревья
    value, class_probs, impurity, num_samples - информация в узле
    """
    def __init__(self, feature=None, weights=None, phi=None, threshold=None,
                 left=None, right=None, value=None, class_probs=None,
                 impurity=None, num_samples=None):
        self.feature = feature
        self.weights = weights
        self.phi = phi
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.class_probs = class_probs
        self.impurity = impurity
        self.num_samples = num_samples

# Классификатор Decision Tree #####################################################################

class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """
    Классификатор на основе бинарного дерева решений.
    Параметры:
        max_depth        - максимальная глубина дерева
        min_impurity     - минимальная примесь для остановки
        min_samples      - минимальное число объектов в узле
        criterion        - 'entropy', 'gini' или 'misclassification'
        split_type       - 'axis_parallel' (по осям),
                           'oblique' (косые, бонус 5),
                           'nonlinear' (нелинейные, бонус 6)
        n_search         - количество случайных поисков для ограниченного перебора (бонус 4)
    """

    def __init__(self, max_depth=10, min_impurity=0.01, min_samples=5,
                 criterion='entropy', split_type='axis_parallel',
                 n_search=50):
        self.max_depth = max_depth
        self.min_impurity = min_impurity
        self.min_samples = min_samples
        self.criterion = criterion
        self.split_type = split_type
        self.n_search = n_search
        self.root = None
        self.n_classes = None
        self._criterion_func = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Обучение дерева на данных X, y."""
        self.n_classes = len(np.unique(y))
        self._criterion_func = CRITERION_DICT[self.criterion]
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """Рекурсивное построение дерева."""
        n_samples = len(y)
        current_impurity = self._criterion_func(y)
        counts = np.bincount(y, minlength=self.n_classes)
        probs = counts / n_samples
        majority_class = np.argmax(probs)

        if (depth >= self.max_depth or
            n_samples < self.min_samples or
            current_impurity < self.min_impurity):
            return Node(value=majority_class, class_probs=probs,
                        impurity=current_impurity, num_samples=n_samples)

        best_split = self._find_best_split(X, y)

        if best_split is None:
            return Node(value=majority_class, class_probs=probs,
                        impurity=current_impurity, num_samples=n_samples)

        left_mask, right_mask, node = best_split

        # Проверка, что маски не пустые
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return Node(value=majority_class, class_probs=probs,
                        impurity=current_impurity, num_samples=n_samples)

        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        node.impurity = current_impurity
        node.num_samples = n_samples

        return node

    def _find_best_split(self, X: np.ndarray, y: np.ndarray):
        """
        Поиск наилучшего разбиения.
        Поддерживает:
          - axis_parallel: полный перебор по осям
          - limited_axis: ограниченный перебор (бонус 4)
          - oblique: косые гиперплоскости (бонус 5)
          - nonlinear: нелинейное разделение (бонус 6)
        """
        if self.split_type == 'axis_parallel':
            return self._find_best_split_axis(X, y, limited=False)
        elif self.split_type == 'limited_axis':
            return self._find_best_split_axis(X, y, limited=True)
        elif self.split_type == 'oblique':
            return self._find_best_split_oblique(X, y)
        elif self.split_type == 'nonlinear':
            return self._find_best_split_nonlinear(X, y)
        else:
            raise ValueError(f"Неизвестный split_type: {self.split_type}")

    def _find_best_split_axis(self, X: np.ndarray, y: np.ndarray,
                              limited: bool = False):
        """
        БОНУС 4 (при limited=True): ограниченный перебор параметров.
        Вместо всех порогов перебирает n_search случайных.
        При limited=False - полный перебор (базовая версия).
        """
        best_ig = -np.inf
        best_split_info = None
        n_features = X.shape[1]

        # Если ограниченный перебор — выбирает случайное подмножество признаков
        if limited:
            # Берём min(n_features, n_search) случайных признаков
            n_feat_to_check = min(n_features, self.n_search)
            features_to_check = np.random.choice(n_features, n_feat_to_check, replace=False)
        else:
            features_to_check = range(n_features)

        for feat in features_to_check:
            uniq = np.unique(X[:, feat])
            if len(uniq) <= 1:
                continue

            # Пороги — середины между соседними уникальными значениями
            thresholds = (uniq[:-1] + uniq[1:]) / 2.0

            if limited:
                # Ограниченный перебор: n_search случайных порогов
                n_thresh = min(len(thresholds), self.n_search)
                thresholds = np.random.choice(thresholds, n_thresh, replace=False)

            for th in thresholds:
                left_mask = X[:, feat] <= th
                right_mask = ~left_mask
                ig = information_gain(y, left_mask, right_mask, self._criterion_func)
                if ig > best_ig:
                    best_ig = ig
                    node = Node(feature=feat, threshold=th)
                    best_split_info = (left_mask, right_mask, node)

        return best_split_info

    def _find_best_split_oblique(self, X: np.ndarray, y: np.ndarray):
        """
        БОНУС 5: Разделение гиперплоскостями, не параллельными осям координат.
        Используем случайный поиск:
          w·x + b = 0
        Перебирает n_search случайных направлений w.
        """
        best_ig = -np.inf
        best_split_info = None
        n_samples, n_features = X.shape

        for _ in range(self.n_search):
            # Генерируем случайное направление
            w = np.random.randn(n_features)
            w = w / (np.linalg.norm(w) + 1e-12)  # нормируем

            # Проекции всех точек на это направление
            projections = X @ w

            # Ищем порог — середины между уникальными проекциями
            uniq = np.unique(projections)
            if len(uniq) <= 1:
                continue

            thresholds = (uniq[:-1] + uniq[1:]) / 2.0

            # Ограниченный перебор порогов
            n_thresh = min(len(thresholds), self.n_search)
            thresh_to_check = np.random.choice(thresholds, n_thresh, replace=False)

            for th in thresh_to_check:
                left_mask = projections <= th
                right_mask = ~left_mask
                ig = information_gain(y, left_mask, right_mask, self._criterion_func)
                if ig > best_ig:
                    best_ig = ig
                    node = Node(weights=w, threshold=th)
                    best_split_info = (left_mask, right_mask, node)

        return best_split_info

    def _find_best_split_nonlinear(self, X: np.ndarray, y: np.ndarray):
        """
        БОНУС 6: Нелинейное разделение через базисные функции phi.
        Используем phi(x) = [x_1, x_2, ..., x_d, x_1^2, x_2^2, ..., x_d^2,
                              x_1*x_2, x_1*x_3, ...]
        (полиномиальные признаки степени 2 + исходные)
        Затем применяем осевое разделение в расширенном пространстве.
        """
        n_samples, n_features = X.shape

        # Строим расширенное пространство признаков
        # phi(x) = [исходные признаки, квадраты, попарные произведения]
        X_squared = X ** 2
        # Попарные произведения (без повторений)
        cross_terms = []
        for i in range(n_features):
            for j in range(i+1, n_features):
                cross_terms.append(X[:, i] * X[:, j])

        if len(cross_terms) > 0:
            X_cross = np.column_stack(cross_terms)
            X_augmented = np.column_stack([X, X_squared, X_cross])
        else:
            X_augmented = np.column_stack([X, X_squared])

        # Сохраняем функцию phi для предсказаний
        def phi_function(x):
            x_sq = x ** 2
            terms = [x, x_sq]
            for i in range(len(x)):
                for j in range(i+1, len(x)):
                    terms.append(np.array([x[i] * x[j]]))
            return np.concatenate(terms)

        # Ищем разбиение в расширенном пространстве (осевое, но ограниченный перебор)
        best_ig = -np.inf
        best_split_info = None
        n_aug_features = X_augmented.shape[1]

        # Ограниченный перебор признаков
        n_feat_to_check = min(n_aug_features, self.n_search)
        features_to_check = np.random.choice(n_aug_features, n_feat_to_check, replace=False)

        for feat in features_to_check:
            uniq = np.unique(X_augmented[:, feat])
            if len(uniq) <= 1:
                continue

            thresholds = (uniq[:-1] + uniq[1:]) / 2.0
            n_thresh = min(len(thresholds), self.n_search)
            thresholds = np.random.choice(thresholds, n_thresh, replace=False)

            for th in thresholds:
                left_mask = X_augmented[:, feat] <= th
                right_mask = ~left_mask
                ig = information_gain(y, left_mask, right_mask, self._criterion_func)
                if ig > best_ig:
                    best_ig = ig
                    # Сохраняем feature как индекс в расширенном пространстве
                    node = Node(feature=feat, threshold=th, phi=phi_function)
                    best_split_info = (left_mask, right_mask, node)

        return best_split_info

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание меток классов."""
        return np.array([self._predict_one(x, self.root) for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Вероятностные предсказания."""
        return np.array([self._predict_proba_one(x, self.root) for x in X])

    def _transform_sample(self, x: np.ndarray, node: Node) -> float:
        """
        Преобразует объект x в скаляр согласно типу разбиения в узле.
        """
        if node.feature is not None and node.phi is None:
            # Осевое разбиение (в т.ч. в расширенном пространстве для nonlinear)
            return x[node.feature]
        elif node.weights is not None:
            # Косое разбиение (бонус 5)
            return x @ node.weights
        elif node.phi is not None:
            # Нелинейное разбиение (бонус 6)
            x_aug = node.phi(x)
            return x_aug[node.feature]
        return 0.0

    def _predict_one(self, x: np.ndarray, node: Node) -> int:
        """Рекурсивный проход по дереву до листа."""
        # Если лист (есть class_probs и нет детей или есть value)
        if node.left is None and node.right is None:
            return node.value

        projection = self._transform_sample(x, node)

        if projection <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def _predict_proba_one(self, x: np.ndarray, node: Node) -> np.ndarray:
        """Рекурсивное извлечение распределения классов из листа."""
        if node.left is None and node.right is None:
            return node.class_probs

        projection = self._transform_sample(x, node)

        if projection <= node.threshold:
            return self._predict_proba_one(x, node.left)
        else:
            return self._predict_proba_one(x, node.right)

def hyperparameter_validation(X_train, y_train, X_test, y_test):
    """
    БОНУС 3: Рандомизированная валидация гиперпараметров.
    Перебор:
      - max_depth
      - критерий (entropy, gini, misclassification)
      - min_impurity
      - min_samples
    Возвращфает лучшие параметры и accuracy.
    """
    # Сетка параметров
    param_distributions = {
        'max_depth': np.arange(3, 16),           # 3..15
        'criterion': ['entropy', 'gini', 'misclassification'],
        'min_impurity': np.logspace(-3, -0.5, 10),  # 0.001 .. 0.3
        'min_samples': np.arange(2, 21),          # 2..20
    }

    n_iter = 100  # количество случайных комбинаций

    best_params = None
    best_score = -np.inf

    print("БОНУС 3: Валидация гиперпараметров")
    print(f"Количество случайных комбинаций: {n_iter}")

    for iteration in range(n_iter):
        # Случайно выбирает параметрым
        params = {
            'max_depth': np.random.choice(param_distributions['max_depth']),
            'criterion': np.random.choice(param_distributions['criterion']),
            'min_impurity': np.random.choice(param_distributions['min_impurity']),
            'min_samples': np.random.choice(param_distributions['min_samples']),
        }

        # Обучение модели
        tree = DecisionTreeClassifier(
            max_depth=params['max_depth'],
            min_impurity=params['min_impurity'],
            min_samples=params['min_samples'],
            criterion=params['criterion'],
            split_type='axis_parallel'
        )
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)
        score = accuracy_score(y_test, y_pred)

        if score > best_score:
            best_score = score
            best_params = params.copy()

        if (iteration + 1) % 20 == 0:
            print(f"  Итерация {iteration+1}/{n_iter}, лучшая точность пока: {best_score:.4f}")

    print(f"\nЛучшие параметры: {best_params}")
    print(f"Лучшая точность на валидации: {best_score:.4f}")

    return best_params, best_score

def HW6():
    # Загрузка датасета digits
    digits = load_digits()
    X, y = digits.data, digits.target

    # Разбиение на обучающую и тестовую выборки (70%/30%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\nРазмер обучающей выборки: {X_train.shape[0]}")
    print(f"Размер тестовой выборки: {X_test.shape[0]}")
    print(f"Количество классов: {len(np.unique(y))}")
    print(f"Количество признаков: {X.shape[1]}")

    # Часть 1: Базовая модель (полный перебор, энтропия, осевые разбиения)
    print("Часть 1: Базовая модель (энтропия, полный перебор)")

    tree_base = DecisionTreeClassifier(
        max_depth=10, min_impurity=0.01, min_samples=5,
        criterion='entropy', split_type='axis_parallel'
    )
    tree_base.fit(X_train, y_train)

    y_train_pred = tree_base.predict(X_train)
    y_test_pred = tree_base.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"Точность на обучении: {train_acc:.4f}")
    print(f"Точность на тесте: {test_acc:.4f}")

    # Часть 2: Сравнение критериев (бонусы 1, 2) + ограниченный перебор (бонус 4)
    print("Часть 2: Сравнение критериев и ограниченный перебор (бонусы 1,2,4)")

    for crit in ['entropy', 'gini', 'misclassification']:
        tree_crit = DecisionTreeClassifier(
            max_depth=10, min_impurity=0.01, min_samples=5,
            criterion=crit, split_type='limited_axis', n_search=30
        )
        tree_crit.fit(X_train, y_train)
        y_pred = tree_crit.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"  Критерий: {crit:20s} | Точность: {acc:.4f}")

    # Часть 3: Косые гиперплоскости (бонус 5)
    print("Часть 3: Косые гиперплоскости (бонус 5)")

    tree_oblique = DecisionTreeClassifier(
        max_depth=8, min_impurity=0.05, min_samples=10,
        criterion='entropy', split_type='oblique', n_search=30
    )
    tree_oblique.fit(X_train, y_train)
    y_pred_obl = tree_oblique.predict(X_test)
    acc_obl = accuracy_score(y_test, y_pred_obl)
    print(f"Точность с косыми гиперплоскостями: {acc_obl:.4f}")

    # Часть 4: Нелинейное разделение (бонус 6)
    print("Часть 4: Нелинейное разделение (бонус 6)")

    tree_nonlin = DecisionTreeClassifier(
        max_depth=6, min_impurity=0.05, min_samples=10,
        criterion='entropy', split_type='nonlinear', n_search=20
    )
    tree_nonlin.fit(X_train, y_train)
    y_pred_nl = tree_nonlin.predict(X_test)
    acc_nl = accuracy_score(y_test, y_pred_nl)
    print(f"Точность с нелинейным разделением: {acc_nl:.4f}")

    # Часть 5: Валидация гиперпараметров (бонус 3)
    best_params, best_val_score = hyperparameter_validation(
        X_train, y_train, X_test, y_test
    )

    print("Обучение итоговой модели с лучшими параметрами")

    tree_best = DecisionTreeClassifier(
        max_depth=best_params['max_depth'],
        min_impurity=best_params['min_impurity'],
        min_samples=best_params['min_samples'],
        criterion=best_params['criterion'],
        split_type='axis_parallel'
    )
    tree_best.fit(X_train, y_train)

    y_train_best = tree_best.predict(X_train)
    y_test_best = tree_best.predict(X_test)

    train_acc_best = accuracy_score(y_train, y_train_best)
    test_acc_best = accuracy_score(y_test, y_test_best)

    print(f"Точность на обучении (лучшие параметры): {train_acc_best:.4f}")
    print(f"Точность на тесте (лучшие параметры): {test_acc_best:.4f}")

    # Часть 6: Построение всех графиков
    print("Построение графиков...")


    # График 1: Confusion matrix для базовой модели (train)
    fig1, ax1 = plt.subplots(figsize=(8, 7))
    fig1.canvas.manager.set_window_title('Confusion Matrix - Train')
    ConfusionMatrixDisplay.from_predictions(
        y_train, y_train_pred, ax=ax1, cmap='Blues', colorbar=False
    )
    ax1.set_title('Confusion Matrix (Train) - Базовая модель')
    plt.tight_layout()
    plt.show(block=False)


    # График 2: Confusion matrix для базовой модели (test)
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    fig2.canvas.manager.set_window_title('Confusion Matrix - Test')
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_test_pred, ax=ax2, cmap='Blues', colorbar=False
    )
    ax2.set_title('Confusion Matrix (Test) - Базовая модель')
    plt.tight_layout()
    plt.show(block=False)


    # График 3: Confusion matrix для лучшей модели (test)
    fig3, ax3 = plt.subplots(figsize=(8, 7))
    fig3.canvas.manager.set_window_title('Confusion Matrix - Best Model')
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_test_best, ax=ax3, cmap='Greens', colorbar=False
    )
    ax3.set_title(f'Confusion Matrix (Test) - Лучшая модель\n'
                  f'criterion={best_params["criterion"]}, '
                  f'max_depth={best_params["max_depth"]}')
    plt.tight_layout()
    plt.show(block=False)


    # График 4: Гистограмма уверенностей (базовая модель)
    test_proba = tree_base.predict_proba(X_test)
    confidences = np.max(test_proba, axis=1)

    correct_mask = (y_test_pred == y_test)
    incorrect_mask = ~correct_mask

    conf_correct = confidences[correct_mask]
    conf_incorrect = confidences[incorrect_mask]

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    fig4.canvas.manager.set_window_title('Гистограмма уверенностей')
    bins = np.linspace(0, 1, 21)
    ax4.hist(conf_correct, bins, alpha=0.7, label=f'Правильные ({len(conf_correct)})',
             color='green', edgecolor='black')
    ax4.hist(conf_incorrect, bins, alpha=0.7, label=f'Ошибочные ({len(conf_incorrect)})',
             color='red', edgecolor='black')
    ax4.set_title('Распределение уверенностей (базовая модель)')
    ax4.set_xlabel('Уверенность')
    ax4.set_ylabel('Количество объектов')
    ax4.legend()
    ax4.grid(alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)


    # График 5: Сравнение критериев
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    fig5.canvas.manager.set_window_title('Сравнение критериев')
    criteria_names = ['Entropy', 'Gini', 'Misclassification']
    criteria_accs = []
    for crit in ['entropy', 'gini', 'misclassification']:
        t = DecisionTreeClassifier(max_depth=10, min_impurity=0.01, min_samples=5,
                                   criterion=crit, split_type='axis_parallel')
        t.fit(X_train, y_train)
        criteria_accs.append(accuracy_score(y_test, t.predict(X_test)))

    bars = ax5.bar(criteria_names, criteria_accs, color=['blue', 'orange', 'red'], edgecolor='black')
    ax5.set_ylabel('Accuracy')
    ax5.set_title('Сравнение критериев качества разделения')
    for bar, acc in zip(bars, criteria_accs):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{acc:.4f}', ha='center', fontsize=11)
    ax5.set_ylim(0, 1)
    plt.tight_layout()
    plt.show(block=False)


    # График 6: Сравнение типов разбиений
    fig6, ax6 = plt.subplots(figsize=(10, 5))
    fig6.canvas.manager.set_window_title('Сравнение типов разбиений')

    split_accs = [test_acc, acc_obl, acc_nl]
    split_names = [
        f'Осевое\n(axis_parallel)\nAcc={test_acc:.3f}',
        f'Косое\n(oblique)\nAcc={acc_obl:.3f}',
        f'Нелинейное\n(nonlinear)\nAcc={acc_nl:.3f}'
    ]
    colors = ['steelblue', 'darkorange', 'forestgreen']
    bars6 = ax6.bar(split_names, split_accs, color=colors, edgecolor='black')
    ax6.set_ylabel('Accuracy')
    ax6.set_title('Сравнение типов разделяющих функций')
    for bar, acc in zip(bars6, split_accs):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{acc:.4f}', ha='center', fontsize=11)
    ax6.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.show(block=False)

    print("Все графики открыты. Закройте окна для завершения.")

    # Держим окна открытыми, пока пользователь не закроет
    plt.show(block=True)


if __name__ == "__main__":
    HW6()