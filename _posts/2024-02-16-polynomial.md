---
title: 'Polynomials: the Good (theory), the Bad (practice), and the Ugly (from theory to practice)'
date: 2024-02-16
permalink: /posts/poly/
tags:
  - Polynomial Optimization
  - Semidefinite Relaxation
  - Machine Learning
---

Background
---
We focus on polynomial optimization problem (POP):

$$\min_{\mathbf{x}} f(\mathbf{x}), \; \text{s.t. } g_i (\mathbf{x}) \ge 0, \; i = 1, \ldots, p,$$

or its matrix form:

$$\min_{\mathbf{X}} \; \langle \mathbf{F}, \mathbf{X} \rangle, \; \text{s.t. } \langle \mathbf{G}_i, \mathbf{X} \rangle \ge 0, \; i = 1, \ldots, p, $$

where $\mathbf{X} = \mathbf{v}_d (\mathbf{x}) \mathbf{v}_d (\mathbf{x})^T$, and $\mathbf{v}_d (\mathbf{x})$ is the monomial basis of degree $d = \max \{\lceil \frac{\deg f}{2} \rceil, \lceil \frac{\deg g_i}{2} \rceil\}$.

Relaxation Techniques
---
The POP above is NP-hard, we relax it using two techniques: lifting and sum-of-squares.

### Lifting
---
We introduce the Riesz functional, defined by the following linear operator:

$$\mathcal{L}: \mathbb{R} [\mathbf{x}] \longrightarrow \mathbb{R}, \; \mathcal{L} (\mathbf{x}^{\alpha}) = \mathbf{y}_{\alpha}, \; \alpha \in \mathbb{N}^n.$$

Then every polynomial can be transformed to a linear function under Riesz functional:

$$\mathcal{L} (f) = \mathcal{L} \bigg(\sum_{\alpha} c_{\alpha} \mathbf{x}^{\alpha}\bigg) = \sum_{\alpha} c_{\alpha} \mathcal{L} (\mathbf{x}^{\alpha}) = \sum_{\alpha} c_{\alpha} \mathbf{y}_{\alpha} = \mathbf{c}^T \mathbf{y}.$$

We can also rewrite the above formulation in matrix form:

$$\mathcal{L} (f) = \mathcal{L} (\langle \mathbf{F}, \mathbf{X} \rangle) = \langle \mathbf{F}, \mathcal{L} (\mathbf{X}) \rangle.$$

### Sum-Of-Squares (SOS)
---
Note that if $\sigma$ is a sum of squares of polynomials, then $\sigma (\mathbf{x}) \ge 0$ for all $\mathbf{x}$, and $f \ge 0$ is equivalent to $f \sigma \ge 0$. Lifting the product $f\sigma$ involves a algebraic combination of the SOS polynomial $\sigma$:

$$\mathcal{L} (f \sigma) = \mathcal{L} \bigg(\sum_{\alpha} c_{\alpha} \mathbf{x}^{\alpha} \sigma\bigg) = \bigg(\mathcal{L} \sum_{\alpha} c_{\alpha} \mathbf{x}^{\alpha} \langle \mathbf{P}, \mathbf{X} \rangle\bigg) = \mathcal{L} \bigg(\sum_{\alpha} \langle c_{\alpha} \mathbf{P}, \mathbf{x}^{\alpha} \mathbf{X} \rangle\bigg) = \sum_{\alpha} \langle c_{\alpha} \mathbf{P}, \mathcal{L} (\mathbf{x}^{\alpha} \mathbf{X}) \rangle.$$

Non-convex Set Approximation
---

Robust Classification
---

ReLU Network Pruning
---

Dataset Condensation
---