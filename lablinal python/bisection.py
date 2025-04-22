class Bisection:
    def __init__(self, f, a, b, eps):
        self.f = f
        self.a = a
        self.b = b
        self.eps = eps
        self.result = a

        self.solve()

    def solve(self):
        if self.f(self.a) * self.f(self.b) > 0:
            raise ValueError('Неверный отрезок')

        a, b = self.a, self.b
        while b - a > self.eps:
            mid = (a + b) / 2

            if self.f(a) * self.f(mid) < 0:
                b = mid
            else:
                a = mid

        self.result = (a + b) / 2
