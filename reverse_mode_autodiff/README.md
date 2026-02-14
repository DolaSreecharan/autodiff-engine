
# 🔄 Reverse-Mode Automatic Differentiation (From Scratch)

## 📚 Overview

This project implements **reverse-mode automatic differentiation from scratch** using a **dynamic computation graph** and **explicit backpropagation rules**.

Reverse-mode autodiff computes gradients by:

1. Performing a **forward pass** to evaluate the function
2. Performing a **backward pass** to propagate gradients using the **chain rule**

This is the **core mechanism behind backpropagation** used in neural networks and deep learning frameworks.

The focus of this project is **conceptual clarity**, not library usage.

---

## 🧠 Key Concepts Demonstrated

| Concept                   | Explanation                        |
| ------------------------- | ---------------------------------- |
| **Computation Graph**     | Graph of primitive operations      |
| **Nodes**                 | Store value, gradient, and parents |
| **Local Gradients**       | Derivatives of each operation      |
| **Chain Rule**            | Applied backward through the graph |
| **Gradient Accumulation** | Handles branching correctly        |
| **Topological Sorting**   | Ensures correct backprop order     |

---

## 🔁 End-to-End Flow

```
Define Input Nodes
 → Build Computation Graph (Forward Pass)
 → Store Local Backprop Rules
 → Seed Output Gradient
 → Topological Sort
 → Backward Gradient Propagation
 → Obtain Gradients of Inputs
```

---

## 📐 Mathematical Foundation

### 1️⃣ Computation Graph Representation

Any function is decomposed into **primitive operations**.

Example:

$$
y = (x + 2)^2
$$

is rewritten as:

$$
\begin{aligned}
u &= x + 2 \\
y &= u^2
\end{aligned}
$$

Each equation becomes a **node** in the graph.

---

### 2️⃣ Node Definition

Each node represents:

$$
z = f(x_1, x_2, \dots)
$$

and stores:

- `val` → numerical value of the node  
- `der` → gradient \( \frac{\partial \text{output}}{\partial z} \)  
- `parents` → input nodes  
- `backprop` → local derivative rule  

---

## ➕ Arithmetic Operations (Local Derivatives)

### Addition

$$
z = a + b
$$

$$
\frac{\partial z}{\partial a} = 1
\quad
\frac{\partial z}{\partial b} = 1
$$

---

### Subtraction

$$
z = a - b
$$

$$
\frac{\partial z}{\partial a} = 1
\quad
\frac{\partial z}{\partial b} = -1
$$

---

### Multiplication

$$
z = ab
$$

$$
\frac{\partial z}{\partial a} = b
\quad
\frac{\partial z}{\partial b} = a
$$

---

### Division

$$
z = \frac{a}{b}
$$

$$
\frac{\partial z}{\partial a} = \frac{1}{b}
\quad
\frac{\partial z}{\partial b} = -\frac{a}{b^2}
$$

---

### Power

**Constant exponent**

$$
z = a^c
\Rightarrow
\frac{dz}{da} = c a^{c-1}
$$

**Variable exponent**

$$
z = a^b
$$

$$
\frac{\partial z}{\partial a} = b a^{b-1}
\quad
\frac{\partial z}{\partial b} = a^b \ln a
$$

---

## 🔗 Elementary & Activation Functions

| Function | Local Derivative |
|--------|--------------|
| sin(x) | cos(x)       |
| cos(x) | -sin(x)      |
| log(x) | 1/x          |
| tan(x) | sec²(x)      |
| sqrt(x) | 1/(2√x)      |
| ReLU   | 1 if x > 0 else 0 |
| Sigmoid | σ(x)(1 − σ(x)) |
| Tanh   | 1 − tanh²(x) |

---

## 🔀 Forward Pass (Value Computation)

During the forward pass:

- Nodes are created dynamically
- Each node computes its numerical value
- Parent relationships are recorded

No gradients are computed here.

---

## 🔙 Backward Pass (Gradient Computation)

### Gradient Seeding

$$
\frac{\partial y}{\partial y} = 1
$$

---

### Chain Rule (Reverse Direction)

For:

$$
z = f(u)
$$

$$
\frac{\partial y}{\partial u}
=
\frac{\partial y}{\partial z}
\cdot
\frac{\partial z}{\partial u}
$$

Gradients are **accumulated** when multiple paths contribute.

---

## 🔁 Topological Sorting

- Nodes are ordered so children appear before parents
- Backpropagation runs in **reverse topological order**

This guarantees correct gradient flow.

---

## 🧪 Worked Example

### Function

$$
y = (x + 2)^2
$$

---

### Forward Pass

| Node | Expression | Value |
|----|-----------|------|
| x | input | 2 |
| u | x + 2 | 4 |
| y | u² | 16 |

---

### Backward Pass

$$
\frac{dy}{dy} = 1
$$

$$
\frac{dy}{du} = 2u = 8
\quad
\frac{du}{dx} = 1
$$

$$
\frac{dy}{dx} = 8
$$

---

### Final Output

```text
y.val = 16
x.der = 8
```

---

## 🆚 Forward Mode vs Reverse Mode

| Feature       | Forward Mode    | Reverse Mode    |
| ------------- | --------------- | --------------- |
| Best for      | Few inputs      | Many inputs     |
| Gradient flow | Inputs → Output | Output → Inputs |
| ML usage      | Rare            | Dominant        |
| Cost          | Per input       | Per output      |

---

## 🎯 Why Reverse Mode Matters

* Efficient for high-dimensional models
* One backward pass computes all gradients
* Foundation of neural networks
* Scales to millions of parameters

---

## 🚫 What This Project Does NOT Do (By Design)

* No symbolic differentiation
* No numerical approximations
* No tensor optimizations
* No external ML libraries

**Reason**:

> The goal is understanding *how backprop works internally*.

---

## 🛠️ Tech Stack

* Python
* Core math (`math`)
* No autodiff or ML libraries

---

## 📌 Project Status

✔ Reverse-mode autodiff implemented
✔ Chain rule verified numerically
✔ Activation functions supported

---

## 🙋 About This Project

This project is part of my **machine learning learning journey**, focused on building **strong mathematical and algorithmic foundations** before working with full neural networks.

**Understand → Implement → Trace → Explain → Document**

---

## ⭐ Support

If you find this project helpful, consider giving the repository a **star** ⭐
It motivates and supports my ML journey.

GitHub Profile:
👉 [https://github.com/DolaSreecharan](https://github.com/DolaSreecharan)

---

