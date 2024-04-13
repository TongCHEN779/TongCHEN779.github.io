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

$$\min_{\mathbf{x} \in \mathbf{K}} f(\mathbf{x}),$$

or its dual form:

$$\max_{\lambda} \; \lambda, \; \text{s.t. } f\big\vert_{\mathbf{K}} - \lambda \ge 0,$$

where $\mathbf{K} = \{\mathbf{x}: g_i (\mathbf{x}) \ge 0, \; i = 1, \ldots, p\}$ is called a basic semialgebraic set.

Relaxation Techniques
---
The POP above is NP-hard, we relax it using two techniques: lifting (primal) and sum-of-squares (dual).

### Lifting
---
We introduce the Riesz functional, defined by the following linear operator:

$$\mathcal{L}_{\mathbf{y}}: \mathbb{R} [\mathbf{x}] \longrightarrow \mathbb{R}, \; \mathbf{x}^{\alpha} \mapsto y_{\alpha}, \; \alpha \in \mathbb{N}^n.$$

Then the objective polynomial can be relaxed to a linear function under Riesz functional:

$$\mathcal{L}_{\mathbf{y}} (f) = \mathcal{L}_{\mathbf{y}} \bigg(\sum_{\alpha} c_{\alpha} \mathbf{x}^{\alpha}\bigg) = \sum_{\alpha} c_{\alpha} \mathcal{L}_{\mathbf{y}} (\mathbf{x}^{\alpha}) = \sum_{\alpha} c_{\alpha} y_{\alpha} = \mathbf{c}^T \mathbf{y}.$$

For the constraint polynomials, we consider the multipliers $\mathbf{v}_d (\mathbf{x}) \mathbf{v}_d (\mathbf{x})^T$ and $\mathbf{v}_{d-\omega_i} (\mathbf{x}) \mathbf{v}_{d-\omega_i} (\mathbf{x})^T$, where $\mathbf{v}_d (\mathbf{x})$ is the monomial basis of degree at most $d$ and $\omega_i = \lceil \frac{\deg g_i}{2} \rceil$. Then the polynomial constraints 

$$\begin{align*}
& \mathbf{v}_d (\mathbf{x}) \mathbf{v}_d (\mathbf{x})^T \succeq 0, \\
& g_i (\mathbf{x}) \mathbf{v}_{d-\omega_i} (\mathbf{x}) \mathbf{v}_{d-\omega_i} (\mathbf{x})^T \succeq 0
\end{align*}$$

can be relaxed to SDP constraints

$$\begin{align*}
& \mathcal{L}_{\mathbf{y}} (\mathbf{v}_d (\mathbf{x}) \mathbf{v}_d (\mathbf{x})^T) =: \mathbf{M}_d (\mathbf{y}) \succeq 0, \\
& \mathcal{L}_{\mathbf{y}} (g_i (\mathbf{x}) \mathbf{v}_{d-\omega_i} (\mathbf{x}) \mathbf{v}_{d-\omega_i} (\mathbf{x})^T) =: \mathbf{M}_{d - \omega_i} (g_i \mathbf{y}) \succeq 0.
\end{align*}$$

### Sum-Of-Squares (SOS)
---
Lifting technique for the primal form of POP can be similarly derived from the dual form. In fact, the nonnegativity constraint $f\big\vert_{\mathbf{K}} - \lambda \ge 0$ can be relaxed to 

$$f - \lambda = \sigma_0 + \sum_{i = 1}^p \sigma_i g_i,$$

where $\sigma_0$ is a SOS polynomial of degree $2d$, and $\sigma_i$ are SOS polynomials of degree $2 (d - \omega_i)$. The existence and convergence guarantee of the above SOS decomposition is given by Putinar's Positivstellensatz. If one is not concerned about the theoretical guarantee, we can freely generate the SOS certificate we need. For instance,

$$f - \lambda = \bigg(1 + \sum_i x_i\bigg)^{2d} + \sum_{i = 1}^p (1+x_i)^{2 (d - \omega_i)} g_i,$$

or

$$f - \lambda = \bigg(1 + \sum_i x_i\bigg)^{2} \sigma_0 + \sum_{i = 1}^p (1+x_i)^{2} \sigma_i g_i,$$

also gives an lower bound. For more systematic and constructive ways of convergent SOS certificate, see [[Ahmadi et al. 2017](https://arxiv.org/abs/1709.09307)].

Non-convex Set Approximation
---

Robust Classification
---

ReLU Network Pruning
---

Dataset Condensation
---