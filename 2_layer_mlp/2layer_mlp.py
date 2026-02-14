import numpy as np

#2 layer - function inside function

class node:
    def __init__(self, val, parents=None, backprop=None):
        self.val = np.array(val,dtype = float)
        self.der = np.zeros_like(self.val,dtype = float)
        self.parents = parents if parents else []
        self.backprop = backprop

    def __add__(self, other ):
        other = other if isinstance(other, node) else node(other)
        val = self.val + other.val
        z = node(val, parents=[self, other])

        def backprop(grad):
            '''
            self.der += 1 * grad
            other.der += 1 * grad
            '''
            # adapt to matmul backprop(A+b) ->A=w@x
            #here self = A,other = b(bias)
            self.der += 1*grad
            other.der += 1*np.sum(grad, axis=1).reshape(other.val.shape) #adjusting shape of grad to match other.der(b.der) to add by summing rows of grad and shaping like other.der


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

            self.der +=  grad * other.val
            if other.val.shape == ():  # helps to deal with scalr der ,in addtion other.der not accept self.val*grad matrix form
                other.der += np.sum(grad*self.val) #it sums up so scalr der of other match with right side sum then adding to other.der
            else:
                other.der +=  grad * self.val

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
                other.der += grad * (self.val**other.val)*np.log(np.maximum(self.val, 1e-8)) #helps to avoid error of 0 val in log
            z.backprop = backprop
            return z


def log(a):
    val = np.log(np.maximum(a.val,1e-8)) #log 0 error values so changing <=0 vals
    z = node(val, parents=[a])

    def backprop(grad):
        a.der += grad * (1 / np.maximum(a.val,1e-8)) #if a.val has 0 then get error so <=0 values are replaces with 1e-8

    z.backprop = backprop
    return z



#upto w@x - broadcasting problems not occur because dL/dW.shape = w.der.shape & dL/dX.shape = x.der.shape
#broadcasting need for +b - (w@x +b)
def matmul(w,x): #w@x
    val = w.val@x.val
    z = node(val,parents = [w,x])

    def backprop(grad):
        #grad = dL/dZ (intial all = 1)
        w.der += grad @ x.val.T  # dL/dW = size of w.der(so no problem in adding mats)
        x.der += w.val.T @ grad  # dL/dX = size of x.der



    z.backprop = backprop
    return z

#activation functions
# “function(loss fun) is like a bowl, middle we have gradient 0(bottom point), left negative gradient(one side steep), right positive gradient(one side steep)”

#relu - if node val is negative make to 0(so respective der at this node become = zero, else = der*1)
def relu(a):
    val = np.maximum(0,a.val)
    z = node(val,parents=[a])
    def backprop(grad):
        #a.der += grad* (1 if a.val>0 else 0)
        a.der += grad * (a.val > 0).astype(float) #true = >0, false = <=0 ,astype make them true-1,false-0

    z.backprop = backprop
    return z

#sigmoid - function one of the reason to use it is produce smooth curve(s-curve) not drastic change
#another reason to fit data into interval (0-1) - centroid 0.5
def sigmoid(a):
    val = 1/(1+np.exp(-a.val))
    z = node(val,parents=[a])

    def backprop(grad):
        a.der += grad *(z.val*(1-z.val)) #der of sigmoid 1/(1+e**-x) is (1/(1+e**-x))*(1-(1/(1+e**-x))
    z.backprop = backprop
    return z
#tanh - contain interval from (-1,1) - centroid 0
#reach down much more faster (grad = 0)
def tanh(a):
    val = (np.exp(a.val) - np.exp(-a.val))/(np.exp(a.val) + np.exp(-a.val))
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
    y.der = np.ones_like(y.val) #making all grad to one (size of Y(w@x) = size of grad)
    seen = set()
    nodes = []
    topo(y, seen, nodes)

    for node in reversed(nodes):
        if node.backprop:
            node.backprop(grad=node.der)


def reset_der(node):
    node.der[:] = 0

#A = w@x ->first matmul then addition
#y= A+b -> i modified sum accordingly you can see in code
#now function inside function z(y(x))
#y = w1x+b1
#z = w2*f(y)+b2  f->applying activation fun to internal layer so the line become non linear and bends

#layer 1
w = node([
    [0.1, -0.2,  0.3],
    [-0.3, 0.2,  0.1]
])

x = node([
    [1.0, 0.5, -1.0],
    [0.0, 1.0, 0.5],
    [1.5, -0.5, 1.0]
])

b = node([
    [0.0],
    [0.0]
])

# layer 2

w1 = node([
    [0.2, -0.1],
    [-0.2, 0.3]
])
b1 = node([
    [0.0],
    [0.0]
])


'''
A = matmul(w,x) #(2,3)
y = A+b 
#y = w@x+b #(2,3)

Y = sigmoid(y)
#activation funs(relu,sigmoid,tanh) to make it non linear (non linear - it bend the line make curves)

A1 = matmul(w1,Y) #(2,3)
Z = A1+b1 #(2,3)
backprop(Z)

print(Z.val)
print(Z.der)
print(w.der) #w contribution in res or loss(if it is far from target then it is contributing to loss because w.val is multiplying x.val(input) so correct w leads to correct contribution)
print(b.der) #b contribution in res or loss

#(w.der rep contrib of w in L(loss)(or)Z,n-learning rate,so we are removing removing too much pos or neg contribution from w.val so new w.val has adjusted contribution )
'''

#for learning - gradient decent
t = node([
    [0.5,  0.8,  0.3],
    [1.0,  0.4,  0.9]
])
tolerance = 0.00001
n = 0.1
while True:
    Y = matmul(w,x)+b
    #Y = sigmoid(y)
    Z = matmul(w1,Y)+b1

    diff = Z-t
    sq = diff*diff
    L = sq*0.5
    print(L.val)

    meanL = np.mean(L.val)
    print('mean - ',meanL)

    if meanL < tolerance:
        print('meanL', meanL,'tolerance' ,tolerance)
        break

    reset_der(w)
    reset_der(b)
    reset_der(w1)
    reset_der(b1)

    backprop(L)

    w.val = w.val - n * w.der
    b.val = b.val - n * b.der
    w1.val = w1.val - n * w1.der
    b1.val = b1.val - n * b1.der









