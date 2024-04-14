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

$$\min_{\mathbf{x} \in \mathbf{K}} f (\mathbf{x}),$$

or its dual form:

$$\max_{\lambda} \; \lambda, \; \text{s.t. } f\big\vert_{\mathbf{K}} - \lambda \ge 0,$$

where $\mathbf{K} = \lbrace\mathbf{x}: g_i (\mathbf{x}) \ge 0, \; i = 1, \ldots, p\rbrace$ is called a basic semialgebraic set.

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
We study the output shape of the unit box $\mathcal{B}^n_1 := [-1, 1]^n$ under a semialgebraic mapping $F: \mathbb{R}^n \rightarrow \mathbb{R}^m$. We assume that $F (\mathcal{B}^n_1) \subseteq \mathcal{B}^m_M := [-M, M]^m$. Then we can approximate the volume of $F (\mathcal{B}^n_1)$ using the following convergent series:

$$\begin{align*}
v_d := & \inf_{f \in \mathbb{R} [\mathbf{z}]} \quad \int_{\mathcal{B}^m_M} f (\mathbf{z}) \text{d} \mathbf{z} \\
& \text{s.t.} \quad \begin{cases}
f - 1 \in Q_d (F (\mathcal{B}^n_1)) \subseteq \mathbb{R} [\mathbf{x}, \mathbf{z}]; \\
f \in Q_d (\mathcal{B}^m_M) \subseteq \mathbb{R} [\mathbf{z}]. 
\end{cases}
\end{align*}$$

Robust Classification
---
Given set $\mathcal{T} = \bigcup_i \mathcal{T}_i \subseteq \mathcal{B}^n_M$, find polynomial $f$ such that $f \ge 0$ over $\mathcal{B}^n_M$, and $f (\mathbf{x}) \in [i-1, i]$ for all $\mathbf{x} \in \mathcal{T}_i$. If $\mathcal{T}_i = \{\mathbf{x}_i^{(j)}\}$ is a discrete set, then the standard classification problem can be formulated as

$$\begin{align*}
\min_{f} & \quad 1 \text{ or } \int_{\mathcal{B}^n_M} f (\mathbf{x}) \text{d} \mathbf{x} \\
\text{s.t.} & \quad \begin{cases}
  f \in Q_d (\mathcal{B}^n_M); \\
  f (\mathbf{x}_i^{(j)}) - i, \; i + 1 - f (\mathbf{x}_i^{(j)}) \ge 0, \; \forall i, j.
\end{cases}
\end{align*}$$

If $\mathcal{T}_i = \{\mathcal{B}^n_{\varepsilon} (\mathbf{x}_i^{(j)})\}$, then the robust classification problem can be formulated as

$$\begin{align*}
\min_{f} & \quad 1 \text{ or } \int_{\mathcal{B}^n_M} f (\mathbf{x}) \text{d} \mathbf{x} \\
\text{s.t.} & \quad \begin{cases}
f \in Q_d (\mathcal{B}^n_M); \\
f - i, \; i+1 - f \in Q_d (\mathcal{B}^n_{\varepsilon} (\mathbf{x}_i^{(j)})), \; \forall i, j.
\end{cases}
\end{align*}$$

ReLU Network Pruning
---
Given a pretrained ReLU network $f (\mathbf{x}) = \mathbf{C} \cdot \text{ReLU} (\mathbf{Ax} + \mathbf{b})$, we are going to find an optimal mask $\mathbf{M}$ of $\mathbf{A}$, with given pruning size $\vert\vert \mathbf{M} \vert\vert_0 = k$, such that the drop of generalization loss is minimized:

$$\min_{\mathbf{M}} \; \mathbb{E} [l (f_{\mathbf{M}} (X), Y)].$$

Empirically, the expectation is computed over the empirical distribution $\{(\mathbf{x}_i, \mathbf{y}_i)\}_{i = 1}^N$, and we take the loss function $l$ to be squared loss. Then the problem becomes

$$\min_{\mathbf{M}} \; \frac{1}{N} \sum_{i = 1}^N (f_{\mathbf{M}} (\mathbf{x}_i) - \mathbf{y}_i)^2.$$

By semialgebraicity of $f$, we rewrite the above problem as a quadratic POP:

$$\begin{align*}
\min_{\mathbf{M}, \mathbf{z}_i} \; & \frac{1}{N} \sum_{i = 1}^N (\mathbf{C} \mathbf{z}_i - \mathbf{y}_i)^2 \\
\text{s.t.} \; & \begin{cases}
\mathbf{z}_i (\mathbf{z}_i - (\mathbf{A} \odot \mathbf{M}) \mathbf{x}_i - \mathbf{b}) = 0, \\
\mathbf{z}_i - (\mathbf{A} \odot \mathbf{M}) \mathbf{x}_i - \mathbf{b} \ge 0, \\
\mathbf{z}_i \ge 0.
\end{cases}
\end{align*}$$