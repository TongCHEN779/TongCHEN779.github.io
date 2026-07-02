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
We focus on the polynomial optimization problem (POP):

$$\min_{\mathbf{x} \in \mathbf{K}} f (\mathbf{x}),$$

or its dual form:

$$\max_{\lambda} \; \lambda, \; \text{s.t. } f\big\vert_{\mathbf{K}} - \lambda \ge 0,$$

where $\mathbf{K} = \lbrace\mathbf{x}: g_i (\mathbf{x}) \ge 0, \; i = 1, \ldots, p\rbrace$ is called a basic semialgebraic set. The primal form asks for the minimum value of $f$ over $\mathbf{K}$; the dual form asks for the largest lower bound $\lambda$ such that $f - \lambda$ is nonnegative over $\mathbf{K}$. The two views are equivalent, but the dual one is the entry point of the algebraic machinery below: certifying nonnegativity of a polynomial.

### Interactive illustration: the dual view

Drag the slider to move the level $\lambda$. As long as $f (x) - \lambda \ge 0$ on $\mathbf{K} = [-1, 1]$ (curve entirely above the dashed line), $\lambda$ is a valid lower bound; the optimal value of the dual problem is reached when the line touches the curve.

<div id="sosdemo-box" style="max-width:640px;margin:1em auto;padding:12px;border:1px solid #ccc;border-radius:8px;font-family:sans-serif;">
  <canvas id="sosdemo-canvas" width="600" height="320" style="width:100%;"></canvas>
  <div style="margin-top:8px;">
    <label for="sosdemo-lambda">lambda = <span id="sosdemo-lval">0.00</span></label>
    <input type="range" id="sosdemo-lambda" min="-100" max="60" value="0" style="width:100%;">
  </div>
  <div id="sosdemo-status" style="margin-top:6px;font-weight:bold;"></div>
  <button id="sosdemo-solve" style="margin-top:6px;padding:4px 12px;">Animate to optimum</button>
</div>

<script>
(function () {
  try {
  var canvas = document.getElementById('sosdemo-canvas');
  if (!canvas) return;
  var ctx = canvas.getContext('2d');
  var slider = document.getElementById('sosdemo-lambda');
  var lval = document.getElementById('sosdemo-lval');
  var status = document.getElementById('sosdemo-status');
  var W = canvas.width, H = canvas.height;
  var xmin = -1.35, xmax = 1.35, ymin = -0.75, ymax = 1.1;
  function f(x) { return Math.pow(x, 4) - x * x + 0.2 * x + 0.3; }
  // minimum of f on [-1,1], computed numerically once
  var fstar = Infinity, xstar = 0;
  for (var i = 0; i <= 2000; i++) {
    var x = -1 + 2 * i / 2000, v = f(x);
    if (v < fstar) { fstar = v; xstar = x; }
  }
  function px(x) { return (x - xmin) / (xmax - xmin) * W; }
  function py(y) { return H - (y - ymin) / (ymax - ymin) * H; }
  function draw(lam) {
    ctx.clearRect(0, 0, W, H);
    // feasible set K = [-1,1]
    ctx.fillStyle = 'rgba(120,180,255,0.15)';
    ctx.fillRect(px(-1), 0, px(1) - px(-1), H);
    // axes
    ctx.strokeStyle = '#999'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, py(0)); ctx.lineTo(W, py(0)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(px(0), 0); ctx.lineTo(px(0), H); ctx.stroke();
    // violation region: where f(x) < lambda on K
    ctx.fillStyle = 'rgba(255,80,80,0.25)';
    for (var i = 0; i <= 600; i++) {
      var x = -1 + 2 * i / 600;
      if (f(x) < lam) {
        var xp = px(x);
        ctx.fillRect(xp - 1, py(lam), 2, py(f(x)) - py(lam));
      }
    }
    // curve f
    ctx.strokeStyle = '#1f4e9c'; ctx.lineWidth = 2;
    ctx.beginPath();
    for (var j = 0; j <= 600; j++) {
      var xx = xmin + (xmax - xmin) * j / 600;
      var yy = f(xx);
      if (j === 0) ctx.moveTo(px(xx), py(yy)); else ctx.lineTo(px(xx), py(yy));
    }
    ctx.stroke();
    // level line lambda
    ctx.strokeStyle = '#c0392b'; ctx.lineWidth = 1.5; ctx.setLineDash([6, 4]);
    ctx.beginPath(); ctx.moveTo(0, py(lam)); ctx.lineTo(W, py(lam)); ctx.stroke();
    ctx.setLineDash([]);
    // minimizer marker
    ctx.fillStyle = '#27ae60';
    ctx.beginPath(); ctx.arc(px(xstar), py(fstar), 4, 0, 2 * Math.PI); ctx.fill();
    // labels
    ctx.fillStyle = '#333'; ctx.font = '13px sans-serif';
    ctx.fillText('K = [-1, 1]', px(-0.15), 16);
    ctx.fillText('f(x)', px(1.12), py(f(1.12)) - 8);
    var feasible = lam <= fstar + 1e-9;
    status.textContent = feasible
      ? (Math.abs(lam - fstar) < 0.01
        ? 'Optimal! lambda = f* = ' + fstar.toFixed(3)
        : 'Valid lower bound: f - lambda is nonnegative on K (gap = ' + (fstar - lam).toFixed(3) + ')')
      : 'Infeasible: f(x) - lambda < 0 somewhere on K (red region)';
    status.style.color = feasible ? '#27ae60' : '#c0392b';
  }
  function lamFromSlider() { return parseFloat(slider.value) / 100; }
  slider.addEventListener('input', function () {
    var lam = lamFromSlider();
    lval.textContent = lam.toFixed(2);
    draw(lam);
  });
  document.getElementById('sosdemo-solve').addEventListener('click', function () {
    var lam = lamFromSlider();
    var step = function () {
      lam += (fstar - lam) * 0.12;
      if (Math.abs(lam - fstar) < 0.002) lam = fstar;
      slider.value = Math.round(lam * 100);
      lval.textContent = lam.toFixed(2);
      draw(lam);
      if (lam !== fstar) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  });
  draw(0);
  } catch (err) {
    var box = document.getElementById('sosdemo-box');
    if (box) {
      var msg = document.createElement('div');
      msg.style.color = '#c0392b';
      msg.style.marginTop = '6px';
      msg.textContent = 'Demo failed to start: ' + err.message;
      box.appendChild(msg);
    }
  }
})();
</script>

Relaxation Techniques
---
The POP above is NP-hard in general. We relax it using two dual techniques: lifting (on the primal side) and sum-of-squares (on the dual side).

### Lifting
We introduce the Riesz functional, defined by the following linear operator:

$$\mathcal{L}_{\mathbf{y}}: \mathbb{R} [\mathbf{x}] \longrightarrow \mathbb{R}, \; \mathbf{x}^{\alpha} \mapsto y_{\alpha}, \; \alpha \in \mathbb{N}^n.$$

Then the objective polynomial can be relaxed to a linear function under the Riesz functional:

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
The lifting technique for the primal form of POP can be similarly derived from the dual form. In fact, the nonnegativity constraint $f\big\vert_{\mathbf{K}} - \lambda \ge 0$ can be relaxed to

$$f - \lambda = \sigma_0 + \sum_{i = 1}^p \sigma_i g_i,$$

where $\sigma_0$ is an SOS polynomial of degree $2d$, and $\sigma_i$ are SOS polynomials of degree $2 (d - \omega_i)$. The existence and convergence guarantee of the above SOS decomposition is given by Putinar's Positivstellensatz. If one is not concerned about the theoretical guarantee, we can freely generate the SOS certificates we need. For instance,

$$f - \lambda = \bigg(1 + \sum_i x_i\bigg)^{2d} + \sum_{i = 1}^p (1+x_i)^{2 (d - \omega_i)} g_i,$$

or

$$f - \lambda = \bigg(1 + \sum_i x_i\bigg)^{2} \sigma_0 + \sum_{i = 1}^p (1+x_i)^{2} \sigma_i g_i,$$

also give a lower bound. For more systematic and constructive ways of building convergent SOS certificates, see [[Ahmadi et al. 2017](https://arxiv.org/abs/1709.09307)].

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
Given a set $\mathcal{T} = \bigcup_i \mathcal{T}_i \subseteq \mathcal{B}^n_M$, find a polynomial $f$ such that $f \ge 0$ over $\mathcal{B}^n_M$, and $f (\mathbf{x}) \in [i-1, i]$ for all $\mathbf{x} \in \mathcal{T}_i$. If $\mathcal{T}_i = \{\mathbf{x}_i^{(j)}\}$ is a discrete set, then the standard classification problem can be formulated as

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

Empirically, the expectation is computed over the empirical distribution $\{(\mathbf{x}_i, \mathbf{y}_i)\}_{i = 1}^N$, and we take the loss function $l$ to be the squared loss. Then the problem becomes

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
