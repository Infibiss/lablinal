import math
import random
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List, Tuple
from sklearn.datasets import load_iris, load_wine
import os
import numpy as np

from matrix import Matrix
from bisection import Bisection


def gauss_solver(A: 'Matrix', B: 'Matrix', eps: 'float' = 1e-15, return_nullspace: bool = True) -> List['Matrix']:
    """
    Вход:
        A: матрица коэффициентов (n×n). Используется класс Matrix из предыдущей лабораторной работы
        B: вектор правых частей (n×1)
    Выход:
        list[Matrix]: список базисных векторов решения системы
    Raises:
        ValueError: если система несовместна или размеры матриц не соответствуют
    """
    # Проверка размеров
    if A.num_rows != A.num_columns:
        raise ValueError("Матрица A должна быть квадратной")
    if A.num_rows != B.num_rows:
        raise ValueError("Размеры A и B не согласуются для решения СЛАУ")

    n = A.num_rows  # Число уравнений
    m = A.num_columns  # Число неизвестных

    # Создаем копию расширенной матрицы [A|B]
    C = Matrix(num_rows=n, num_cols=m + 1, accuracy=A.accuracy)

    for r in range(1, n + 1):
        for c in range(1, m + 1):
            C[r, c] = A[r, c]

        C[r, m + 1] = B[r, 1]

    # Прямой ход с частичным выбором главного элемента
    TOL = eps
    rankA = 0
    row_swap_sign = 1

    for col in range(1, m + 1):
        # Поиск максимального элемента в столбце ниже текущей строки
        pivot_row = None
        pivot_val = 0.0

        for r in range(rankA + 1, n + 1):
            val = abs(C[r, col])

            if val > pivot_val:
                pivot_val = val
                pivot_row = r

        if pivot_row is None or pivot_val < TOL:
            continue

        # Перестановка строк, если нужно
        if pivot_row != rankA + 1:
            row_swap_sign *= -1
            for cc in range(1, m + 2):
                C[rankA + 1, cc], C[pivot_row, cc] = C[pivot_row, cc], C[rankA + 1, cc]

        # Нормировка ведущей строки
        pivot_elt = C[rankA + 1, col]
        if abs(pivot_elt) < TOL:
            continue
        for cc in range(col, m + 2):
            C[rankA + 1, cc] /= pivot_elt

        # Обнуление элементов ниже
        for r in range(rankA + 2, n + 1):
            factor = C[r, col]
            if abs(factor) > TOL:
                for cc in range(col, m + 2):
                    C[r, cc] -= factor * C[rankA + 1, cc]

        rankA += 1

        if rankA == n:
            break

    # Проверка совместности
    for r in range(rankA + 1, n + 1):
        zeroA = all(abs(C[r, c]) < 1e-10 for c in range(1, m + 1))
        if zeroA and any(abs(C[r, c]) > 1e-10 for c in range(m + 1, m + 2)):
            raise ValueError("Система несовместна")

    # Обратный ход
    for r in range(rankA, 0, -1):
        lead_col = next((c for c in range(1, m + 1) if abs(C[r, c]) > 1e-10), None)
        if lead_col is None:
            continue
        for rr in range(r - 1, 0, -1):
            factor = C[rr, lead_col]
            if abs(factor) > 1e-10:
                for cc in range(lead_col, m + 2):
                    C[rr, cc] -= factor * C[r, cc]

    # Нахождение базисных и свободных переменных
    pivot_positions = {}
    used_cols = set()
    rowid = 1

    for c in range(1, m + 1):
        if rowid <= rankA and abs(C[rowid, c]) > 1e-10:
            pivot_positions[rowid] = c
            used_cols.add(c)
            rowid += 1

    free_cols = [c for c in range(1, m + 1) if c not in used_cols]
    if not free_cols and all(abs(B[i + 1, 1]) < eps for i in range(m)):
        free_cols.append(m)

    # Формирование решений
    solutions = []

    # Частное решение (все свободные = 0)
    x_part = [0.0] * m

    for r, colp in pivot_positions.items():
        x_part[colp - 1] = C[r, m + 1]

    base_part = Matrix(num_rows=m, num_cols=1, accuracy=A.accuracy)
    for i in range(m):
        base_part[i + 1, 1] = x_part[i]

    # (a) частное решение оставляем только если B не ноль
    if any(abs(B[i + 1, 1]) > eps for i in range(m)):
        solutions.append(base_part)

    # Базисные векторы ядра (по числу свободных переменных)
    for sc in free_cols:
        param_vec = [0.0] * m
        param_vec[sc - 1] = 1.0
        for r, colp in pivot_positions.items():
            sum_in_row = sum(C[r, c] * param_vec[c - 1] for c in range(1, m + 1) if c != colp)
            param_vec[colp - 1] = -sum_in_row
        # Добавляем только ненулевые векторы
        if any(abs(x) > eps for x in param_vec):
            dir_vec = Matrix(num_rows=m, num_cols=1, accuracy=A.accuracy)
            for i in range(m):
                dir_vec[i + 1, 1] = param_vec[i]
            solutions.append(dir_vec)

    # если ничего так и не добавили (крайний случай) → единичный вектор
    if not solutions and return_nullspace:
        v = Matrix(num_rows=m, num_cols=1, accuracy=A.accuracy)
        v[1, 1] = 1.0
        solutions.append(v)

    return solutions


def center_data(X: 'Matrix') -> 'Matrix':
    """
    Вход: матрица данных X (n×m)
    Выход: центрированная матрица X_centered (n×m)
    """
    n = X.num_rows
    m = X.num_columns
    Xc = Matrix(num_rows=n, num_cols=m, accuracy=X.accuracy)

    for col in range(1, m + 1):
        # Считаем среднее по столбцу
        s = 0.0
        for row in range(1, n + 1):
            s += X[row, col]
        mean_c = s / n
        # Вычитаем из столбца найденное среднее
        for row in range(1, n + 1):
            Xc[row, col] = X[row, col] - mean_c

    return Xc


def covariance_matrix(X_centered: 'Matrix') -> 'Matrix':
    """
    Вход: центрированная матрица X_centered (n×m)
    Выход: матрица ковариаций C (m×m)
    """
    n = X_centered.num_rows
    m = X_centered.num_columns

    # C = (1 / (n - 1)) * (X^T * X)
    scale = 1.0 / (n - 1) if n > 1 else 1.0
    return scale * (X_centered.transpose() * X_centered)


def find_eigenvalues(C: 'Matrix', tol: float = 1e-15) -> List[float]:
    """
    Вход:
        C: матрица ковариаций (m×m)
        tol: допустимая погрешность
    Выход: список вещественных собственных значений, найденных через Bisection
    """
    if C.num_rows != C.num_columns:
        raise ValueError("Матрица не квадратная")

    n = C.num_rows

    # Так как матрица ковариаций симметрична, то ее собственные значения неотрицательны
    left_bound = -1e-6
    right_bound = C.get_trace()  # По теореме Виета их сумма не превосходит следа матрицы

    # det(C - lam * E)
    def det_func(lam: float) -> float:
        M = Matrix(num_rows=n, num_cols=n, accuracy=C.accuracy, arr=C.list())  # Копия
        for i in range(1, n + 1):
            M[i, i] = M[i, i] - lam
        return M.get_determinant()

    # Ищем n корней. Разобьем на отрезки и будем смотреть смену знака, так как тогда внутри корень
    intervals = []
    k = 500  # Коэффициент разбиения отрезка
    step = (right_bound - left_bound) / float(n * k + 1)  # Например поделим так

    x1 = left_bound
    prev_val = det_func(x1)
    for i in range(1, n * k + 2):
        x2 = left_bound + i * step

        val = det_func(x2)
        if prev_val * val < 0:  # Нашли смену знака
            intervals.append((x1, x2))
            if len(intervals) == n:  # Считаем что у нас n корней
                break

        x1 = x2
        prev_val = val

    # Бисекция на каждом интервале
    eigenvalues = []
    for (a, b) in intervals:
        lam_approx = Bisection(det_func, a, b, tol).result
        eigenvalues.append(lam_approx)

    # Сортируем
    eigenvalues.sort(reverse=True)
    return eigenvalues

def find_eigenvectors(C: 'Matrix', eigenvalues: List[float]) -> List['Matrix']:
    """
    Вход:
        C: матрица ковариаций (m×m)
        eigenvalues: список собственных значений
    Выход: список собственных векторов (каждый вектор - объект Matrix)
    """
    vectors = []
    m = C.num_rows
    for lam in eigenvalues:
        # Используем точность повыше
        Cl = Matrix(num_rows=m, num_cols=m, accuracy=15, arr=C.list())
        for i in range(1, m + 1):
            Cl[i, i] = Cl[i, i] - lam

        # Используем метод Гаусса
        B0 = Matrix(num_rows=m, num_cols=1, accuracy=15)
        solutions = gauss_solver(Cl, B0, eps=1e-8)

        # берем первый ненулевой вектор и нормируем
        v = next(vec for vec in solutions if math.sqrt(sum(vec[i + 1, 1] ** 2 for i in range(m))) > 1e-8)

        # нормировка
        norm = math.sqrt(sum(v[i + 1, 1] ** 2 for i in range(m)))
        for i in range(1, m + 1):
            v[i, 1] /= norm
        vectors.append(v)

    return vectors


def explained_variance_ratio(eigenvalues: List[float], k: int) -> float:
    """
    Вход:
        eigenvalues: список собственных значений
        k: число компонент
    Выход: доля объяснённой дисперсии
    """
    s_all = sum(eigenvalues)
    sorted_vals = sorted(eigenvalues, reverse=True)  # Отсортируем по убыванию
    s_k = sum(sorted_vals[:k])  # Берем k главных компонент

    return s_k / s_all if abs(s_all) > 1e-10 else 1.0


def pca(X: 'Matrix', k: list) -> Tuple['Matrix', float]:
    """
    Вход:
        X: матрица данных (n×m)
        k: число главных компонент
    Выход:
        X_proj: проекция данных (n×k)
        ratio: доля объяснённой дисперсии
    """
    # 1) Центрируем
    Xc = center_data(X)

    # 2) Считаем матрицу ковариаций
    C = covariance_matrix(Xc)

    # 3) Находим собственные значения и собственные векторы
    eigvals = find_eigenvalues(C)
    eigvecs = find_eigenvectors(C, eigvals)

    # Упорядочим их по убыванию
    pairs = list(zip(eigvals, eigvecs))
    pairs.sort(key=lambda x: x[0], reverse=True)
    # Если k не выбрано сделаем автоподбор
    if k is None:
        k = auto_select_k(eigvals)
    top_k = pairs[:k]

    # Формируем матрицу W из k собственных векторов (m x k)
    W = Matrix(num_rows=C.num_rows, num_cols=k, accuracy=C.accuracy)
    for j, (ev, evec_list) in enumerate(top_k, start=1):
        # evec_list — это один вектор-столбец (или несколько)
        # Возьмём первый
        for i in range(1, C.num_rows + 1):
            W[i, j] = evec_list[i, 1]

    # 4) Проекция Xc (n x m) на W (m x k) => (n x k)
    X_proj = Xc * W

    # Доля объясненной дисперсии
    ratio = explained_variance_ratio(eigvals, k)
    return (X_proj, ratio)


def plot_pca_projection(X_proj: 'Matrix', labels: List[int] | None = None) -> Figure:
    fig, ax = plt.subplots()
    xs = [X_proj[i, 1] for i in range(1, X_proj.num_rows + 1)]
    ys = [X_proj[i, 2] for i in range(1, X_proj.num_rows + 1)]

    if labels is None:
        ax.scatter(xs, ys, s=30)
    else:
        classes = sorted(set(labels))
        palette = plt.cm.get_cmap('tab10', len(classes))
        for cl in classes:
            pts = [(x, y) for x, y, lab in zip(xs, ys, labels) if lab == cl]
            ax.scatter(*zip(*pts), label=f"class{cl}", color=palette(cl))
        ax.legend()

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA projection")
    return fig


def reconstruction_error(X: 'Matrix', X_recon: 'Matrix') -> float:
    """
    Вход:
        X: исходные данные (n×m)
        X_recon: восстановленные данные (n×m)
    Выход: среднеквадратическая ошибка MSE
    """
    if X.num_rows != X_recon.num_rows or X.num_columns != X.num_columns:
        raise ValueError("Размеры не совпадают")

    n = X.num_rows
    m = X.num_columns
    sse = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diff = X[i, j] - X_recon[i, j]
            sse += diff * diff
    mse = sse / (n * m)
    return mse


def auto_select_k(eigenvalues: List[float], threshold: float = 0.95) -> int:
    """
    Вход:
        eigenvalues: список собственных значений
        threshold: порог объяснённой дисперсии
    Выход: оптимальное число главных компонент k
    """
    for k in range(1, len(eigenvalues) + 1):
        val = explained_variance_ratio(eigenvalues, k)
        if val > threshold:
            return k
    return len(eigenvalues)


def handle_missing_values(X: 'Matrix') -> 'Matrix':
    """
    Вход: матрица данных X (n×m) с возможными NaN
    Выход: матрица данных X_filled (n×m) без NaN
    """
    n = X.num_rows
    m = X.num_columns
    Xc = Matrix(num_rows=n, num_cols=m, accuracy=X.accuracy)

    for col in range(1, m + 1):
        # Считаем среднее по столбцу
        s = 0.0
        for row in range(1, n + 1):
            elem = X[row, col]
            if elem is not None:
                s += elem
        mean_c = s / n
        # Заменяем NaN на среднее по столбцу
        for row in range(1, n + 1):
            if Xc[row, col] is None:
                Xc[row, col] = mean_c
    return Xc


def add_noise_and_compare(X: 'Matrix', noise_level: float = 0.1):
    """
    Вход:
        X: матрица данных (n×m)
        noise_level: уровень шума (доля от стандартного отклонения)
    Выход: результаты PCA до и после добавления шума.
        В этом задании можете проявить творческие способности, поэтому выходные данные не типизированы.
    """
    n = X.num_rows
    m = X.num_columns
    Xc = Matrix(num_rows=n, num_cols=m, accuracy=X.accuracy, arr=X.list())

    # Найдем std по всем элементам (или по столбцам)
    s = 0.0
    cnt = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            val = X[i, j]
            s += val * val
            cnt += 1
    std = (s / cnt) ** 0.5

    # Добавим шум - noise_level * std
    X_noisy = Matrix(num_rows=n, num_cols=m, accuracy=X.accuracy, arr=X.list())
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            noise = random.gauss(0, 1) * noise_level * std
            X_noisy[i, j] = X[i, j] + noise

    # Применяем PCA к Xc
    k = min(m, 2)
    X_proj, ratio = pca(Xc, k)

    # Применяем pca к X_noisy
    X_proj_noisy, ratio_noisy = pca(X_noisy, k)

    # Результаты в виде словаря
    if X_proj.num_rows == X_proj_noisy.num_rows and X_proj.num_columns == X_proj_noisy.num_columns:
        e = 0.0
        total = X_proj.num_rows * X_proj.num_columns
        for r in range(1, X_proj.num_rows + 1):
            for c in range(1, X_proj.num_columns + 1):
                diff = X_proj[r, c] - X_proj_noisy[r, c]
                e += diff * diff
        mse = e / total
    else:
        mse = None

    return {
        "ratio": ratio,
        "ratio_noisy": ratio_noisy,
        "mse_proj": mse,
    }


def apply_pca_to_dataset(dataset_name: str, k: int) -> Tuple['Matrix', float]:
    """
    Вход:
        dataset_name: название датасета
        k: число главных компонент
    Выход: кортеж (проекция данных, качество модели)
    """
    # Для примера используем load_iris, load_wine и т.д.
    df = None
    if dataset_name.lower() == "iris":
        df = load_iris()
    elif dataset_name.lower() == "wine":
        df = load_wine()
    else:
        raise ValueError("Неизвестный датасет")

    # Превращаем ds.data в Matrix
    arr = df.data
    n, m = arr.shape
    X = Matrix(num_rows=n, num_cols=m, accuracy=15, arr=arr.tolist())

    # Возвращаем результат PCA
    X_proj, ratio = pca(X, k)

    # Посмотрим результат и сохраним
    fig = plot_pca_projection(X_proj, labels=df.target.tolist())
    fig.tight_layout()
    save_path = os.path.abspath(f"{dataset_name}_projection.png")
    fig.savefig(save_path)
    print(f"Сохранено изображение: {save_path}")

    return X_proj, ratio


if __name__ == '__main__':
    # Проверка add_noise_and_compare
    X = Matrix(num_rows=4, num_cols=2, arr=[[1, 2], [4, 3], [5, 6], [8, 7]])
    result = add_noise_and_compare(X, noise_level=0.2)

    print("Результаты add_noise_and_compare:", result)
