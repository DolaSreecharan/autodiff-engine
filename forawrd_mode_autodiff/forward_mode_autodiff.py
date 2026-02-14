import math

class duel:
    def __init__(self, val, dervative=0):
        self.val = val
        self.dervative = dervative

    def __add__(self, other):
        other = other if isinstance(other, duel) else duel(other)
        return duel(self.val + other.val, self.dervative + other.dervative)

    def __mul__(self, other):
        other = other if isinstance(other, duel) else duel(other)
        return duel(
            self.val * other.val,
            self.dervative * other.val + self.val * other.dervative
        )

    def __truediv__(self, other):
        other = other if isinstance(other, duel) else duel(other)
        return duel(
            self.val / other.val,
            (other.val * self.dervative - self.val * other.dervative) / (other.val) ** 2
        )

    def __sub__(self, other):
        other = other if isinstance(other, duel) else duel(other)
        return duel(self.val - other.val, self.dervative - other.dervative)

    def __pow__(self, other):
        other = other if isinstance(other, duel) else duel(other)
        val = self.val ** other.val
        der = val * (
            other.dervative * math.log(self.val) +
            other.val * (self.dervative / self.val)
        )
        return duel(val, der)

    def __neg__(self):
        return duel(-self.val, -self.dervative)

    def __radd__(self, other):
        other = duel(other)
        return duel(self.val + other.val, self.dervative + other.dervative)

    def __rmul__(self, other):
        other = duel(other)
        return duel(
            self.val * other.val,
            self.dervative * other.val + self.val * other.dervative
        )

    def __rtruediv__(self, other):
        other = duel(other)
        return duel(
            other.val / self.val,
            (self.val * other.dervative - other.val * self.dervative) / (self.val) ** 2
        )

    def __rsub__(self, other):
        other = duel(other)
        return duel(other.val - self.val, other.dervative - self.dervative)

    def __rpow__(self, other):
        other = duel(other)
        val = other.val ** self.val
        der = val * (
            self.dervative * math.log(other.val) +
            self.val * (other.dervative / other.val)
        )
        return duel(val, der)

    def __repr__(self):
        return f'val = {self.val}, der = {self.dervative}'


def sin(v):
    return duel(math.sin(v.val), math.cos(v.val) * v.dervative)

def cos(v):
    return duel(math.cos(v.val), -math.sin(v.val) * v.dervative)

def exp(v):
    return duel(math.exp(v.val), math.exp(v.val) * v.dervative)

def log(v):
    return duel(math.log(v.val), (1 / v.val) * v.dervative)

def tan(v):
    return duel(
        math.tan(v.val),
        (1 / (math.cos(v.val)) ** 2) * v.dervative
    )

def sqrt(v):
    return duel(
        math.sqrt(v.val),
        (1 / (2 * math.sqrt(v.val))) * v.dervative
    )

def sinhx(v):
    return duel(math.sinh(v.val), math.cosh(v.val) * v.dervative)

def coshx(v):
    return duel(math.cosh(v.val), math.sinh(v.val) * v.dervative)

def tanhx(v):
    return duel(
        math.tanh(v.val),
        (1 - (math.tanh(v.val)) ** 2) * v.dervative
    )


x = duel(2, 1)
y = duel(3, 0)

a = ((3 * x) ** 2) + 2 * x
# a = tanhx(x)

print(a)
