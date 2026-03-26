Here is your coursework written up **exactly into clean Markdown**, preserving structure and notation as closely as possible:

---

# Data-driven and Learning Based Methods in Mechanics and Materials

**Lent 2026**
**Dr. Burigede Liu, University of Cambridge**

---

## Project #3

* **Assigned:** Tuesday, Mar 10, 2026
* **Due:** Tuesday, March 31, 2026, 17:00

Please submit your project online through Moodle!

---

## Collaboration Policy

You are encouraged to discuss the homework problems and solution strategies with your classmates, but you must find the solutions and write down your answers to all problems by yourself.

---

## Problem 1: Recurrent Neural Operator – One Dimensional Visco-Plasticity (100 marks)

Consider the following unit-cell problem that is governed by:

### Governing Equations

[
\epsilon(x, t) = \frac{\partial u(x, t)}{\partial x}
]
*Kinematic relation*

[
\frac{d\sigma(x)}{dx} = 0
]
*Equilibrium*

[
\sigma(x, t) = E(x)\epsilon(x, t) + v(x)\frac{\partial u(x, t)}{\partial t}
]
*Constitutive relation*

---

### Initial Conditions

[
u(x, 0) = 0, \quad \dot{u}(x, 0) = 0
]

---

### Boundary Conditions

[
u(0, t) = 0, \quad u(1, t) = \bar{\epsilon}(t)
]

---

where ( E(x) ) is Young’s modulus and ( v(x) ) is viscosity. We consider a **3-phase composite** with ( E(x) ) and ( v(x) ) piecewise constant functions with 3 different values, as depicted in Fig. 1.

**Figure 1:** Viscoelastic material with 3 phases

---

### Questions

#### (a)

Define the input and output for a macroscopic constitutive model based on the provided unit cell problem.

---

#### (b)

You are provided with a dataset (`viscodata_3mat.mat`) containing 400 samples of macroscopic stresses and strains obtained by solving the unit cell problem.

Design, build, and train a **Recurrent Neural Operator (RNO)** to learn the macroscopic constitutive model of the composite using training data.

---

#### (c)

Determine the **minimum number of internal/hidden variables** required in the RNO to fully learn the constitutive relation.

Justify your choice.

---

## Notes on Conversion

* Mathematical expressions were converted into LaTeX-compatible Markdown (`$$` style avoided per your preference—can switch if needed).
* Symbols like ( \epsilon ), ( \sigma ), etc., were preserved exactly.
* Structure, wording, and formatting remain unchanged.

---

If you want, next step I can:

* Turn this into a **LaTeX report template with sections already structured for your answers**, or
* Help you **start solving part (a)** (this one has a very clean systems interpretation).
