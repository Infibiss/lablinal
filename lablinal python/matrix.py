class Matrix:
    def __init__(self, num_rows: int = 0, num_cols: int = 0, accuracy: int = 20, arr:list[list[float]] = None):
        """
        Класс хранит разреженную матрицу в формате (values, column_indices, row_sizes), где:
        - values: все ненулевые значения по строкам
        - column_indices: список индексов столбцов для соответствующих значений
        - row_sizes: префиксные суммы (row_sizes[i+1] - row_sizes[i] даёт число ненулевых элементов в i-й строке)
        """
        self.num_rows = num_rows
        self.num_columns = num_cols
        self.accuracy = accuracy

        # В этих списках будет храниться разреженное представление
        self.values = []
        self.column_indices = []
        self.row_sizes = [0]

        if not arr:
            # Если массив не передан, просто заполним row_sizes нулями
            for _ in range(self.num_rows):
                self.row_sizes.append(self.row_sizes[-1])
        else:
            # Если массив передан, добавим элементы
            for row in range(self.num_rows):
                # Каждая строка в row_sizes начинается с предыдущего значения
                self.row_sizes.append(self.row_sizes[-1])
                for column in range(self.num_columns):
                    self._add_element(arr[row][column], column)


    def _add_element(self, element, column) -> None:
        """
        Вспомогательный метод добавления ненулевого элемента в конец структуры (в текущую строку) с округлением
        """
        if element is not None: # Придется хранить NaN для их будущей замены средним
            element = round(float(element), self.accuracy)
        if element != 0: # Придется хранить NaN для их будущей замены средним
            self.values.append(element)
            self.column_indices.append(column)
            self.row_sizes[-1] += 1


    def list(self) -> list[list[float]]:
        """
        Преобразование разреженного представления в двумерный список
        """
        return [[self[row + 1, col + 1] for col in range(self.num_columns)] for row in range(self.num_rows)]


    def check_idx(self, row: int, column: int):
        """
        Проверка корректности индексов
        """
        if row < 0 or row >= self.num_rows or column < 0 or column >= self.num_columns:
            raise IndexError("Выход за границы матрицы")


    def __getitem__(self, item: tuple[int, int]) -> float:
        """
        Получение элемента матрицы по (row, column)
        """
        row, column = item[0] - 1, item[1] - 1
        self.check_idx(row, column)

        # Определяем промежуток ненулевых элементов, отвечающих за эту строку
        start = self.row_sizes[row]
        end = self.row_sizes[row + 1]
        col_part = self.column_indices[start:end]

        # Если column есть в col_part, значит элемент ненулевой, иначе 0
        if column in col_part:
            local_index = col_part.index(column)
            return self.values[start + local_index]
        else:
            return round(0.0, self.accuracy)

    def __setitem__(self, item: tuple[int, int], value: float) -> None:
        """
        Установка элемента в матрице по (row, column)
        Если значение становится 0, оно удаляется из структуры
        """
        row, column = item[0] - 1, item[1] - 1
        value = round(value, self.accuracy)
        self.check_idx(row, column)

        start = self.row_sizes[row]
        end = self.row_sizes[row + 1]
        col_part = self.column_indices[start:end]

        # Если устанавливаем 0
        if value == 0:
            if column in col_part:
                # Найдём индекс в values/column_indices
                local_index = start + col_part.index(column)
                # Сдвигаем все элементы в values и column_indices на 1 влево
                for idx in range(local_index, self.row_sizes[-1] - 1):
                    self.values[idx] = self.values[idx + 1]
                    self.column_indices[idx] = self.column_indices[idx + 1]
                # Уменьшаем все row_sizes, начиная со строки row+1
                for r in range(row + 1, self.num_rows + 1):
                    self.row_sizes[r] -= 1
                # Удаляем последний элемент
                self.values.pop()
                self.column_indices.pop()
            else:
                # Значение и так 0, то ничего делать не надо
                pass
        # Устанавливаем значение != 0
        else:
            if not self.values:
                # Если матрица совсем пустая
                self.values.append(value)
                self.column_indices.append(column)
                for r in range(row + 1, self.num_rows + 1):
                    self.row_sizes[r] += 1
            elif column in col_part:
                # Элемент уже есть, просто меняем его значение
                local_index = start + col_part.index(column)
                self.values[local_index] = value
            else:
                # Элемента нет, придётся вставлять
                # Для начала расширим на 1 элемент в конце
                self.values.append(self.values[-1])
                self.column_indices.append(self.column_indices[-1])

                # Сдвигаем все элементы после позиции end-1 вправо
                for idx in range(self.row_sizes[-1] - 1, end, -1):
                    self.values[idx] = self.values[idx - 1]
                    self.column_indices[idx] = self.column_indices[idx - 1]

                # Увеличиваем все row_sizes, начиная со строки row+1
                for r in range(row + 1, self.num_rows + 1):
                    self.row_sizes[r] += 1

                # Ставим нужный столбец и значение
                self.column_indices[end] = column
                self.values[end] = value

    def _matrices_sum(self, other: 'Matrix') -> 'Matrix':
        """
        Вспомогательная функция для сложения двух матриц
        """
        result = Matrix()
        result.accuracy = self.accuracy
        result.num_rows = self.num_rows
        result.num_columns = self.num_columns

        for row in range(1, self.num_rows + 1):
            result.row_sizes.append(result.row_sizes[-1])
            first_ptr = self.row_sizes[row - 1]
            second_ptr = other.row_sizes[row - 1]

            while first_ptr < self.row_sizes[row] or second_ptr < other.row_sizes[row]:
                # Случай, когда индексы совпадают
                if (first_ptr < self.row_sizes[row] and second_ptr < other.row_sizes[row] and
                        self.column_indices[first_ptr] == other.column_indices[second_ptr]):
                    val = self.values[first_ptr] + other.values[second_ptr]
                    col = self.column_indices[first_ptr]
                    result._add_element(val, col)
                    first_ptr += 1
                    second_ptr += 1
                # Случай, когда пробегаем элементы первой матрицы
                elif (first_ptr < self.row_sizes[row] and (second_ptr >= other.row_sizes[row] or
                       self.column_indices[first_ptr] < other.column_indices[second_ptr])):
                    val = self.values[first_ptr]
                    col = self.column_indices[first_ptr]
                    result._add_element(val, col)
                    first_ptr += 1
                # Случай, когда пробегаем элементы второй матрицы
                else:
                    val = other.values[second_ptr]
                    col = other.column_indices[second_ptr]
                    result._add_element(val, col)
                    second_ptr += 1

        return result

    def __add__(self, other) -> 'Matrix':
        """
        Матричное сложение
        """
        if isinstance(other, Matrix):
            if self.num_rows != other.num_rows or self.num_columns != other.num_columns:
                raise Exception('Размеры матриц не совпадают')
            return self._matrices_sum(other)
        else:
            raise TypeError("Сложение возможно только с другой матрицей")


    def __radd__(self, other) -> 'Matrix':
        return self.__add__(other)


    def transpose(self) -> 'Matrix':
        """
        Транспонирование матрицы
        """
        # Для каждой колонки будущей транспонированной матрицы собираем индексы строк куда перейдет и значения
        column_row = [[] for _ in range(self.num_columns)]
        column_value = [[] for _ in range(self.num_columns)]

        for row in range(self.num_rows):
            start = self.row_sizes[row]
            end = self.row_sizes[row + 1]
            for idx in range(start, end):
                col = self.column_indices[idx]
                val = self.values[idx]
                # В транспонированной матрице это пойдет в строку col
                column_row[col].append(row)
                column_value[col].append(val)

        # Создаем результирующую матрицу
        result = Matrix()
        result.accuracy = self.accuracy
        result.num_rows = self.num_columns
        result.num_columns = self.num_rows

        for col in range(self.num_columns):
            result.row_sizes.append(result.row_sizes[-1])
            for pos in range(len(column_row[col])):
                result._add_element(column_value[col][pos], column_row[col][pos])

        return result


    def _matrix_and_matrix_mul(self, other: 'Matrix') -> 'Matrix':
        """
        Умножение двух матриц
        Используется прием: A*B=A*(B^T)^T, но для экономии транспонируем B один раз и потом просто скалярно перемножаем строки A и строки B^T
        """
        other_T = other.transpose()

        result = Matrix()
        result.accuracy = self.accuracy
        result.num_rows = self.num_rows
        result.num_columns = other.num_columns

        for row in range(self.num_rows):
            result.row_sizes.append(result.row_sizes[-1])
            startA, endA = self.row_sizes[row], self.row_sizes[row + 1]

            for rowB_T in range(other_T.num_rows):
                startB, endB = other_T.row_sizes[rowB_T], other_T.row_sizes[rowB_T + 1]
                element = 0.0
                ptrA = startA
                ptrB = startB

                # Пока оба указателя не вышли за пределы
                while ptrA < endA and ptrB < endB:
                    colA = self.column_indices[ptrA]
                    colB = other_T.column_indices[ptrB]
                    if colA == colB:
                        element += self.values[ptrA] * other_T.values[ptrB]
                        ptrA += 1
                        ptrB += 1
                    elif colA < colB:
                        ptrA += 1
                    else:
                        ptrB += 1

                result._add_element(element, rowB_T)

        return result


    def _matrix_and_scalar_mul(self, scalar: float) -> 'Matrix':
        """
        Умножение матрицы на число
        """
        result = Matrix()
        result.accuracy = self.accuracy
        result.num_rows = self.num_rows
        result.num_columns = self.num_columns

        for row in range(self.num_rows):
            result.row_sizes.append(result.row_sizes[-1])
            start = self.row_sizes[row]
            end = self.row_sizes[row + 1]
            for idx in range(start, end):
                val = self.values[idx]
                col = self.column_indices[idx]
                result._add_element(val * scalar, col)

        return result


    def __mul__(self, other) -> 'Matrix':
        """
        Матричное умножение
        """
        if isinstance(other, Matrix):
            if self.num_columns != other.num_rows:
                raise Exception('Неподходящие размеры матриц')
            return self._matrix_and_matrix_mul(other)
        elif isinstance(other, (int, float)):
            return self._matrix_and_scalar_mul(other)
        else:
            raise TypeError("Неверный тип данных")


    def __rmul__(self, other) -> 'Matrix':
        return self.__mul__(other)


    def _calculate_trace(self) -> float:
        """
        Подсчёт следа
        """
        trace = 0.0
        for row in range(self.num_rows):
            trace += self[row + 1, row + 1]
        return trace


    def get_trace(self) -> float:
        """
        Возвращает след матрицы (если она квадратная)
        """
        if self.num_rows != self.num_columns:
            raise Exception('Матрица не квадратная')
        return self._calculate_trace()


    def _calculate_determinant(self) -> float:
        """
        Вычисление определителя с помощью Гауссова исключения
        Определитель квадратной матрицы можно найти через LU-разложение (верхнетреугольная матрица) с частичным выбором главного элемента
        Ход алгоритма:
        1) Применяем пошаговое исключение:
           - Ищем строку с максимальным по модулю элементом в текущем столбце (pivot)
           - Если pivot=0, то det=0
           - Меняем строки местами при необходимости (учитываем знак перестановки)
           - Нормируем ведущую строку и вычитаем её из нижних строк, зануляя элементы под ведущим элементом
        2) Произведение диагональных элементов (с учётом перестановок строк) дает det
        """
        determinant = 1.0
        for i in range(1, self.num_rows + 1):
            # Ищем строку с максимальным элементом в i-м столбце для pivot
            pivot = i
            for k in range(i + 1, self.num_rows + 1):
                if abs(self[k, i]) > abs(self[pivot, i]):
                    pivot = k

            # Если ведущий элемент 0, то det=0
            if self[pivot, i] == 0:
                return 1e-10

            # Если надо, то меняем строки, учитывая знак перестановки
            if pivot != i:
                determinant *= -1
                for j in range(1, self.num_rows + 1):
                    self[i, j], self[pivot, j] = self[pivot, j], self[i, j]

            # Диагональный элемент
            current = self[i, i]
            determinant *= current

            # Сначала нормируем i-ю строку
            for j in range(i, self.num_rows + 1):
                self[i, j] /= current

            # Вычитаем из нижних строк
            for k in range(i + 1, self.num_rows + 1):
                factor = self[k, i]
                for j in range(i, self.num_rows + 1):
                    self[k, j] -= factor * self[i, j]

        determinant = determinant
        return determinant


    def get_determinant(self) -> float:
        """
        Возвращает определитель матрицы (если она квадратная), используя метод Гаусса с выбором главного элемента
        """
        if self.num_rows != self.num_columns:
            raise Exception('Матрица не квадратная')
        return self._calculate_determinant()


    def determinant_and_invertibility(self) -> None:
        """
        Выводит определитель и можно ли найти обратную (вырождена ли матрица)
        """
        det = self.get_determinant()
        print(det, "Да" if det != 0 else "Нет")

