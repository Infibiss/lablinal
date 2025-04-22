#include <iostream>
#include <vector>
#include <gtest/gtest.h>

using namespace std;

class Matrix {
private:
    int n, m; // Размеры матрицы
    vector<double> val; // Ненулевые элементы
    vector<int> col; // Индексы столбцов для ненулевых элементов
    vector<int> row; // Сумма ненулевых элементов по строкам и предыдущим

public:
    // Конструктор с инициализацией размеров
    Matrix(int r, int c) : n(r), m(c) {
        row.resize(n + 2, 0); // Начальное заполнение row нулями (0 элемент не используется, а 1 элемент фиктивный)
    }

    // Метод для добавления элемента в матрицу
    void add(int i, int j, double elem) {
        if (i <= 0 || i > n || j <= 0 || j > m) {
            throw out_of_range("Индекс выходит за пределы матрицы");
        }

        if (elem == 0) // Пропускаем нулевые элементы
            return;

        // Добавляем элемент в val и col
        val.push_back(elem);
        col.push_back(j);

        // Увеличиваем row[i + 1] и последующие элементы
        for (int k = i + 1; k <= n + 1; k++) {
            row[k]++;
        }
    }

    // Метод для подсчета следа матрицы
    double trace() {
        if (n != m)
            throw logic_error("Не квадратная матрица");
        double trace = 0;
        for (int i = 1; i <= n; i++) {
            trace += (*this)[i][i];
        }
        return trace;
    }

    // Дополнительный класс для перегрузки []
    class RowProxy {
    private:
        const Matrix &matrix;
        int rowIndex;
    public:
        RowProxy(const Matrix &matrix, int rowIndex) : matrix(matrix), rowIndex(rowIndex) {}

        // Доступ к элементу в строке через индекс столбца
        double operator[](int colIndex) const {
            if (colIndex <= 0 || colIndex > matrix.m) {
                throw out_of_range("Индекс столбца выходит за пределы матрицы");
            }

            // Ищем элемент в строке
            for (int k = matrix.row[rowIndex]; k < matrix.row[rowIndex + 1]; k++) {
                if (matrix.col[k] == colIndex) {
                    return matrix.val[k];
                }
            }
            return 0;
        }
    };

    // Перегрузка оператора [] для доступа к строке
    RowProxy operator[](int rowIndex) const {
        if (rowIndex <= 0 || rowIndex > n) {
            throw out_of_range("Индекс строки выходит за пределы матрицы");
        }
        return RowProxy(*this, rowIndex);
    }

    Matrix operator+(Matrix &other) const {
        if (this->n != other.n || this->m != other.m)
            throw logic_error("Разные размеры");

        Matrix res(n, m);
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                res.add(i, j, (*this)[i][j] + other[i][j]);
            }
        }
        return res;
    }

    Matrix operator*(double scalar) const {
        Matrix res(n, m);
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                res.add(i, j, (*this)[i][j] * scalar);
            }
        }
        return res;
    }

    Matrix operator*(Matrix &other) const {
        if (this->m != other.n)
            throw logic_error("Неподходящие размеры");

        Matrix res(this->n, other.m);
        for (int row = 1; row <= n; row++) {
            for (int col = 1; col <= other.m; col++) {
                double sum = 0;
                for (int k = 1; k <= m; k++) {
                    sum += (*this)[row][k] * other[k][col];
                }
                res.add(row, col, sum);
            }
        }
        return res;
    }

    bool operator==(Matrix &other) const {
        if (this->n != other.n || this->m != other.m)
            return false;
        if (this->val != other.val)
            return false;
        if (this->col != other.col)
            return false;
        if (this->row != other.row)
            return false;
        return true;
    }

    // Рекурсивная функция для вычисления определителя
    double determinant() const {
        if (n != m)
            throw logic_error("Не квадратная матрица");
        return determinantRecursive(*this, n);
    }

    // Вспомогательная функция для рекурсивного вычисления определителя
    double determinantRecursive(const Matrix &matrix, int size) const {
        if (size == 1) { // Для матрицы 1x1
            return matrix[1][1];
        }
        if (size == 2) { // Для матрицы 2x2
            return matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1];
        }

        double det = 0;
        for (int col = 1; col <= size; col++) {
            // Создаем подматрицу (минор)
            Matrix minorMatrix(size - 1, size - 1);
            for (int i = 2; i <= size; i++) {
                for (int j = 1; j <= size; j++) {
                    if (j == col) continue; // Пропускаем столбец col
                    minorMatrix.add(i - 1, j < col ? j : j - 1, matrix[i][j]);
                }
            }

            // Рекурсивно вычисляем определитель минора
            det += (col % 2 == 1 ? 1 : -1) * matrix[1][col] * determinantRecursive(minorMatrix, size - 1);
        }

        return det;
    }

    // Метод для вывода матрицы в CSR-формате
    void print() const {
        cout << "val: ";
        for (double v : val) cout << v << " ";
        cout << "\ncol: ";
        for (int c : col) cout << c << " ";
        cout << "\nrow: ";
        for (int r : row) cout << r << " ";
        cout << endl;
    }
};

int main(int argc, char *argv[]) {
    // int n, m;
    // cin >> n >> m;
    //
    // Matrix matrix(n, m);
    //
    // double elem;
    // for (int i = 1; i <= n; i++) {
    //     for (int j = 1; j <= m; j++) {
    //         cin >> elem;
    //         matrix.add(i, j, elem);
    //     }
    // }

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

Matrix add_matrix(int n, int m, vector<vector<double>> &inp) {
    Matrix matrix(n, m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            matrix.add(i + 1, j + 1, inp[i][j]);
        }
    }
    return matrix;
}

TEST(MatrixTest, Test0) {
    int n = 2, m = 2;
    vector<vector<double>> inp = {
        {1, 2},
        {3, 4}
    };
    Matrix matrix = add_matrix(n, m, inp);
    EXPECT_TRUE(5 == matrix.trace());
    EXPECT_TRUE(-2 == matrix.determinant());
}

TEST(MatrixTest, Test1) {
    int n = 4, m = 4;
    vector<vector<double>> inp = {
        {16, 2, 3, 16},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {16, 14, 15, 16}
    };
    Matrix matrix = add_matrix(n, m, inp);
    EXPECT_TRUE(49 == matrix.trace());
    EXPECT_TRUE(144 == matrix.determinant());
}

TEST(MatrixTest, Test2) {
    int n = 3, m = 4;
    vector<vector<double>> inp = {
        {16, 2, 3, 16},
        {5, 6, 7, 8},
        {9, 10, 11, 12}
    };
    Matrix matrix = add_matrix(n, m, inp);
    EXPECT_ANY_THROW(matrix.trace());
    EXPECT_ANY_THROW(matrix.determinant());
}

TEST(MatrixTest, Test3) {
    int n = 2, m = 2;
    vector<vector<double>> inp = {
        {1, 2},
        {3, 4}
    };
    Matrix matrix = add_matrix(n, m, inp);
    vector<vector<double>> inp_ans = {
        {2, 4},
        {6, 8}
    };
    Matrix ans = add_matrix(n, m, inp_ans);
    EXPECT_TRUE(matrix + matrix == ans);
}

TEST(MatrixTest, Test4) {
    int n = 2, m = 2;
    vector<vector<double>> inp = {
        {1, 2},
        {3, 4}
    };
    Matrix matrix = add_matrix(n, m, inp);
    vector<vector<double>> inp_ans = {
        {3, 6},
        {9, 12}
    };
    Matrix ans = add_matrix(n, m, inp_ans);
    EXPECT_TRUE(matrix * 3 == ans);
}

TEST(MatrixTest, Test5) {
    int n = 2, m = 2;
    vector<vector<double>> inp = {
        {1, 2},
        {3, 4}
    };
    Matrix matrix = add_matrix(n, m, inp);
    vector<vector<double>> inp_ans = {
        {7, 10},
        {15, 22}
    };
    Matrix ans = add_matrix(n, m, inp_ans);
    EXPECT_EQ(matrix * matrix == ans);
}

TEST(MatrixTest, Test6) {
    int n = 2, m = 2;
    vector<vector<double>> inp = {
        {1.1, 2.12},
        {3.45, 4.21}
    };
    Matrix matrix = add_matrix(n, m, inp);
    EXPECT_NEAR(-2.683, matrix.determinant(), 1e-5);
}