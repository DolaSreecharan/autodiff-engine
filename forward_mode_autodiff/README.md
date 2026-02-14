
# 🔁 Forward-Mode Automatic Differentiation (From Scratch)

## 📚 Overview

This project implements **forward-mode automatic differentiation from scratch** using **dual numbers** and **operator overloading** in Python.

Instead of relying on symbolic differentiation or numerical approximations (finite differences), this implementation computes:

* the **function value**
* the **exact derivative**

**simultaneously**, in a single forward pass through the computation graph.

The goal of this project is **conceptual mastery** — understanding *how modern ML frameworks compute gradients internally*.

---

## 🧠 Key Ideas Behind the Project

| Concept                   | Explanation                                          |
| ------------------------- | ---------------------------------------------------- |
| **Dual Numbers**          | Represent value and derivative together              |
| **Forward-Mode Autodiff** | Derivatives propagate forward using chain rule       |
| **Operator Overloading**  | Math expressions become differentiable automatically |
| **Exact Gradients**       | No numerical error (unlike finite differences)       |
| **Chain Rule Automation** | Nested expressions handled correctly                 |

---

## 🔁 End-to-End Flow

```
Define Dual Number
 → Seed Independent Variable
 → Overload Arithmetic Operators
 → Implement Elementary Functions
 → Compose Expressions
 → Forward Propagation
 → Obtain Value + Derivative
```

---

## 📐 Mathematical Foundation

### 1️⃣ Dual Numbers

Each scalar is represented as:


$$
x = (v, \dot v)
$$


Where:

* $( v ) → numerical value$
* $( \dot v = \frac{dv}{dx} )$ → derivative w.r.t. the chosen variable

This can be written as:

$$
x = v + \epsilon \dot v
\quad \text{with } \epsilon^2 = 0
$$

---

### 2️⃣ Seeding the Derivative

To compute derivatives with respect to ( x ):

```python
x = duel(2, 1)
y = duel(3, 0)
```

Mathematically:

$$
\frac{dx}{dx} = 1, \quad \frac{dy}{dx} = 0
$$

Only the seeded variable contributes to gradient flow.

---

## ➕ Arithmetic Operations (Chain Rule Engine)

Each operator applies **calculus rules automatically**.

---

### Addition

$$
(a + b)' = a' + b'
$$

$$
(a,\dot a) + (b,\dot b) = (a+b,\dot a + \dot b)
$$

---

### Subtraction

$$
(a - b)' = a' - b'
$$

---

### Multiplication (Product Rule)

$$
(ab)' = a'b + ab'
$$

$$
(a,\dot a)(b,\dot b) = (ab, \dot a b + a \dot b)
$$

---

### Division (Quotient Rule)

$$
\left(\frac{a}{b}\right)' =
\frac{a'b - ab'}{b^2}
$$

---

### Power (General Case)

For:

$$
f(x) = u(x)^{v(x)}
$$

Derivative:

$$
\frac{d}{dx}(u^v)
u^v
\left(
v' \ln u + v \frac{u'}{u}
\right)
$$

This supports:

* $( x^x )$
* $( a^x )$
* $( x^y )$

---

## 🔄 Reverse Operators (`__radd__`, `__rmul__`, …)

These allow expressions such as:

```python
3 * x
2 + x
```

Constants are internally converted to:

$$
c = (c, 0)
$$

ensuring correct derivative propagation.

---

## 🔗 Elementary Functions (Chain Rule)

Each function follows:

$$
\frac{d}{dx} f(x) = f'(x) \cdot x'
$$

| Function        | Derivative                          |
|-----------------|-------------------------------------|
| $ ( \sin x )$   | $( \cos x \cdot x' ) $              |
| $( \cos x )$    | $( -\sin x \cdot x' )  $            |
| $( \exp x )$    | $( e^x \cdot x' )       $           |
| $( \log x )$    | $( \frac{1}{x} \cdot x' )$          |
| $( \tan x )$    | $( \sec^2 x \cdot x' )   $          |
| $( \sqrt{x} )$  | $( \frac{1}{2\sqrt{x}} \cdot x' ) $ |
| $( \tanh x )  $ | $( (1 - \tanh^2 x)\cdot x' )   $    |

---

## 🧪 Worked Example (Numerical Forward-Mode Differentiation)

### Function

$$
f(x) = (3x)^2 + 2x
$$

---

### Step 1: Seed

$$
x = (2, 1)
$$

---

### Step 2: Compute ( 3x )

$$
3x = (6, 3)
$$

---

### Step 3: Square

$$
(3x)^2 = (36, 36)
$$

---

### Step 4: Compute ( 2x )

$$
2x = (4, 2)
$$

---

### Step 5: Add

$$
f(x) = (40, 38)
$$

---

### ✅ Final Output

```text
val = 40
der = 38
```

---

## 📌 Why This Matters

* Avoids numerical instability of finite differences
* Automatically applies chain rule correctly
* Scales to deeply nested expressions
* Forms the foundation of **backpropagation** and **deep learning frameworks**

---

## 🚫 What This Project Does NOT Do (By Design)

* No symbolic algebra
* No numerical approximation
* No ML libraries
* No computation graph storage

**Reason**:

> The focus is understanding *how differentiation flows*, not building a full framework.

---

## 🛠️ Tech Stack

* Python
* `math` module only
* No external autodiff or ML libraries

---

## 📌 Project Status

✔ Forward-mode autodiff implemented
✔ Exact gradients verified
✔ Elementary functions supported

---

## 🙋 About This Project

This project is part of my **machine learning learning journey**, focused on building **strong mathematical foundations** before moving to neural networks and backpropagation.

**Understand → Implement → Verify → Document**

---

## ⭐ Support

If you find this project helpful, consider giving the repository a **star** ⭐
It motivates and supports my ML journey.

GitHub Profile:
👉 [https://github.com/DolaSreecharan](https://github.com/DolaSreecharan)

---

