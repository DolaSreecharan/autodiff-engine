import numpy as np
import math


class node:
    def __init__(self, val, parents=None, backprop=None):
        self.val = val
        self.der = 0
        self.parents = parents if parents else []
        self.backprop = backprop

    def __add__(self, other ):
        other = other if isinstance(other, node) else node(other)
        val = self.val + other.val
        z = node(val, parents=[self, other])

        def backprop(grad):
            self.der += 1 * grad
            other.der += 1 * grad

        z.backprop = backprop
        return z

    def __sub__(self, other):
        other = other if isinstance(other, node) else node(other)
        val = self.val - other.val
        z = node(val, parents=[self, other])

        def backprop(grad):
            self.der += 1 * grad
            other.der += -1 * grad

        z.backprop = backprop
        return z

    def __mul__(self, other):
        other = other if isinstance(other, node) else node(other)
        val = self.val * other.val
        z = node(val, parents=[self, other])

        def backprop(grad):
            self.der += grad * other.val
            other.der += grad * self.val

        z.backprop = backprop
        return z

    def __truediv__(self, other):
        other = other if isinstance(other, node) else node(other)
        val = self.val / other.val
        z = node(val, parents=[self, other])

        def backprop(grad):
            self.der += grad * (1 / other.val)
            other.der += grad * (-self.val / other.val ** 2)

        z.backprop = backprop
        return z

    def __pow__(self, other): #if other - val or other - node
        if not isinstance(other,node):
            val = self.val ** other
            z = node(val, parents=[self])

            def backprop(grad):
                self.der += grad * (other * (self.val ** (other - 1)))

            z.backprop = backprop
            return z
        else:
            val = self.val**other.val
            z = node(val,parents=[self,other])

            def backprop(grad):
                self.der += grad * (other.val * (self.val ** (other.val - 1)))
                other.der += grad * (self.val**other.val)*math.log(self.val)
            z.backprop = backprop
            return z


def sin(a):
    val = math.sin(a.val)
    z = node(val, parents=[a])

    def backprop(grad):
        a.der += grad * math.cos(a.val)

    z.backprop = backprop
    return z


def cos(a):
    val = math.cos(a.val)
    z = node(val, parents=[a])

    def backprop(grad):
        a.der += grad * (-math.sin(a.val))

    z.backprop = backprop
    return z


def log(a):
    val = math.log(a.val)
    z = node(val, parents=[a])

    def backprop(grad):
        a.der += grad * (1 / a.val)

    z.backprop = backprop
    return z


def tan(a):
    val = math.tan(a.val)
    z = node(val, parents=[a])

    def backprop(grad):
        a.der += grad * (1 / (math.cos(a.val) ** 2))

    z.backprop = backprop
    return z


def sqrt(a):
    val = math.sqrt(a.val)
    z = node(val, parents=[a])

    def backprop(grad):
        a.der += grad * (1 / (2 * z.val))

    z.backprop = backprop
    return z
#activation functions
# “function(loss fun) is like a bowl, middle we have gradient 0(bottom point), left negative gradient(one side steep), right positive gradient(one side steep)”

#relu - if node val is negative make to 0(so respective der at this node become = zero, else = der*1)
def relu(a):
    val = max(0,a.val)
    z = node(val,parents=[a])
    def backprop(grad):
        a.der += grad* (1 if a.val>0 else 0)
    z.backprop = backprop
    return z

#sigmoid - function one of the reason to use it is produce smooth curve(s-curve) not drastic change
#another reason to fit data into interval (0-1) - centroid 0.5
def sigmoid(a):
    val = 1/(1+math.exp(-a.val))
    z = node(val,parents=[a])

    def backprop(grad):
        a.der += grad *(z.val*(1-z.val)) #der of sigmoid 1/(1+e**-x) is (1/(1+e**-x))*(1-(1/(1+e**-x))
    z.backprop = backprop
    return z
#tanh - contain interval from (-1,1) - centroid 0
#reach down much more faster (grad = 0)
def tanh(a):
    val = (math.exp(a.val) - math.exp(-a.val))/(math.exp(a.val) + math.exp(-a.val))
    z = node(val,parents=[a])

    def backprop(grad):
        a.der += grad*(1-(z.val)**2)
    z.backprop = backprop
    return z


def topo(n, seen, nodes):
    if n not in seen:
        seen.add(n)
        for i in n.parents:
            topo(i, seen, nodes)
        nodes.append(n)


def backprop(y):
    y.der = 1
    seen = set()
    nodes = []
    topo(y, seen, nodes)

    for node in reversed(nodes):
        if node.backprop:
            node.backprop(grad=node.der)


x = node(2)
y = (x+2)**2

print(y.val)
backprop(y)
print(x.der)





