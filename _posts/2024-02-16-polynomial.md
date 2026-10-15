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

Pick a polynomial $f$ and a feasible set $\mathbf{K} = [a, b]$, then drag the level $\lambda$. As long as $f (x) - \lambda \ge 0$ on $\mathbf{K}$ (the curve stays above the dashed line), $\lambda$ is a valid lower bound; the optimal value of the dual problem is reached when the line touches the curve. The red shading marks where the certificate fails. Note how the answer can change completely when $\mathbf{K}$ cuts off the global minimizer: this is precisely the role of the constraint multipliers $\sigma_i g_i$ in the SOS certificate below.

<div class="demo" id="sosdemo-box">
  <canvas id="sosdemo-canvas" width="640" height="320"></canvas>
  <div class="demo-row">
    <label for="sosdemo-f">f(x) =</label>
    <select id="sosdemo-f">
      <option value="0">x^4 - x^2 + 0.2x + 0.3 (double well)</option>
      <option value="1">x^3 - x (cubic)</option>
      <option value="2">x^6 - 2x^4 + x^2 + 0.1 (flat wells)</option>
      <option value="3">sin(3x) + 0.5x (non-polynomial, for contrast)</option>
    </select>
  </div>
  <div class="demo-row">
    <label for="sosdemo-a">a = <span id="sosdemo-aval">-1.00</span></label>
    <input type="range" id="sosdemo-a" min="-130" max="130" value="-100">
    <label for="sosdemo-b">b = <span id="sosdemo-bval">1.00</span></label>
    <input type="range" id="sosdemo-b" min="-130" max="130" value="100">
  </div>
  <div class="demo-row">
    <label for="sosdemo-lambda">lambda = <span id="sosdemo-lval">0.00</span></label>
    <input type="range" id="sosdemo-lambda" min="-200" max="100" value="0">
    <button id="sosdemo-solve">Animate to optimum</button>
  </div>
  <div class="demo-status" id="sosdemo-status"></div>
</div>

<script>
(function () {
  try {
  var D = Demo, c = D.setup(document.getElementById('sosdemo-canvas')), ctx = c.ctx, W = c.W, H = c.H;
  var fs = [
    function (x) { return Math.pow(x, 4) - x * x + 0.2 * x + 0.3; },
    function (x) { return x * x * x - x; },
    function (x) { return Math.pow(x, 6) - 2 * Math.pow(x, 4) + x * x + 0.1; },
    function (x) { return Math.sin(3 * x) + 0.5 * x; }
  ];
  var xmin = -1.5, xmax = 1.5, ymin = -1.4, ymax = 1.6;
  var sel = document.getElementById('sosdemo-f');
  var sa = document.getElementById('sosdemo-a'), sb = document.getElementById('sosdemo-b');
  var sl = document.getElementById('sosdemo-lambda');
  var status = document.getElementById('sosdemo-status');
  var fstar = 0, xstar = 0, anim = null;
  function f(x) { return fs[parseInt(sel.value, 10)](x); }
  function px(x) { return (x - xmin) / (xmax - xmin) * W; }
  function py(y) { return H - (y - ymin) / (ymax - ymin) * H; }
  function interval() {
    var a = parseFloat(sa.value) / 100, b = parseFloat(sb.value) / 100;
    return [Math.min(a, b), Math.max(a, b)];
  }
  function solve() {
    var iv = interval();
    fstar = Infinity;
    for (var i = 0; i <= 3000; i++) {
      var x = iv[0] + (iv[1] - iv[0]) * i / 3000, v = f(x);
      if (v < fstar) { fstar = v; xstar = x; }
    }
  }
  function draw() {
    var iv = interval(), lam = parseFloat(sl.value) / 100;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = 'rgba(120,180,255,0.15)';
    ctx.fillRect(px(iv[0]), 0, px(iv[1]) - px(iv[0]), H);
    ctx.strokeStyle = '#999'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, py(0)); ctx.lineTo(W, py(0)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(px(0), 0); ctx.lineTo(px(0), H); ctx.stroke();
    ctx.fillStyle = 'rgba(255,80,80,0.28)';
    for (var i = 0; i <= 640; i++) {
      var x = iv[0] + (iv[1] - iv[0]) * i / 640, fx = f(x);
      if (fx < lam) ctx.fillRect(px(x) - 0.5, py(lam), 1.5, py(fx) - py(lam));
    }
    ctx.strokeStyle = D.C.blue; ctx.lineWidth = 2;
    ctx.beginPath();
    for (var j = 0; j <= 640; j++) {
      var xx = xmin + (xmax - xmin) * j / 640, yy = f(xx);
      if (j === 0) ctx.moveTo(px(xx), py(yy)); else ctx.lineTo(px(xx), py(yy));
    }
    ctx.stroke();
    ctx.strokeStyle = D.C.red; ctx.lineWidth = 1.5; ctx.setLineDash([6, 4]);
    ctx.beginPath(); ctx.moveTo(0, py(lam)); ctx.lineTo(W, py(lam)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = D.C.green;
    ctx.beginPath(); ctx.arc(px(xstar), py(fstar), 5, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = D.C.text; ctx.font = '13px sans-serif';
    ctx.fillText('K = [' + iv[0].toFixed(2) + ', ' + iv[1].toFixed(2) + ']', px(iv[0]) + 4, 16);
    ctx.fillText('minimizer x* = ' + xstar.toFixed(2), px(xstar) - 40, py(fstar) + 20);
    document.getElementById('sosdemo-lval').textContent = lam.toFixed(2);
    var ok = lam <= fstar + 1e-9;
    status.className = 'demo-status ' + (ok ? 'demo-ok' : 'demo-bad');
    status.textContent = ok
      ? (Math.abs(lam - fstar) < 0.011
        ? 'Optimal: lambda = f* = ' + fstar.toFixed(3)
        : 'Valid lower bound: f - lambda >= 0 on K (gap = ' + (fstar - lam).toFixed(3) + ')')
      : 'Infeasible: f(x) - lambda < 0 somewhere on K (red region)';
  }
  function refresh() {
    document.getElementById('sosdemo-aval').textContent = interval()[0].toFixed(2);
    document.getElementById('sosdemo-bval').textContent = interval()[1].toFixed(2);
    solve(); draw();
  }
  sel.addEventListener('change', refresh);
  sa.addEventListener('input', refresh);
  sb.addEventListener('input', refresh);
  sl.addEventListener('input', draw);
  document.getElementById('sosdemo-solve').addEventListener('click', function () {
    if (anim) cancelAnimationFrame(anim);
    var lam = parseFloat(sl.value) / 100;
    var step = function () {
      lam += (fstar - lam) * 0.1;
      if (Math.abs(lam - fstar) < 0.002) lam = fstar;
      sl.value = Math.floor(lam * 100 + 1e-9);
      draw();
      if (lam !== fstar) anim = requestAnimationFrame(step);
    };
    anim = requestAnimationFrame(step);
  });
  refresh();
  } catch (err) { Demo.fail('sosdemo-box', err); }
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

where $\sigma_0$ is an SOS polynomial of degree $2d$, and $\sigma_i$ are SOS polynomials of degree $2 (d - \omega_i)$. The set of polynomials admitting such a representation is the degree-$d$ truncated quadratic module, denoted $Q_d (\mathbf{K})$ in the sections below. The existence and convergence guarantee of the above SOS decomposition is given by Putinar's Positivstellensatz. If one is not concerned about the theoretical guarantee, we can freely generate the SOS certificates we need. For instance,

$$f - \lambda = \bigg(1 + \sum_i x_i\bigg)^{2d} + \sum_{i = 1}^p (1+x_i)^{2 (d - \omega_i)} g_i,$$

or

$$f - \lambda = \bigg(1 + \sum_i x_i\bigg)^{2} \sigma_0 + \sum_{i = 1}^p (1+x_i)^{2} \sigma_i g_i,$$

also give a lower bound. For more systematic and constructive ways of building convergent SOS certificates, see [[Ahmadi et al. 2017](https://arxiv.org/abs/1709.09307)].

### The Lasserre hierarchy
Increasing $d$ gives a hierarchy of SDPs. Let $\lambda_d$ denote the optimal value of the SOS relaxation of order $d$ (equivalently, of the moment relaxation of order $d$, by SDP duality). Then

$$\lambda_1 \le \lambda_2 \le \cdots \le f^* := \min_{\mathbf{x} \in \mathbf{K}} f (\mathbf{x}),$$

and $\lambda_d \to f^*$ as $d \to \infty$ under Putinar's assumption (the quadratic module is Archimedean, e.g., $\mathbf{K}$ is compact and one of the $g_i$ is $R - \Vert \mathbf{x} \Vert^2$). In practice, convergence is often finite: the relaxation is exact at a small $d$, and exactness can be certified by a rank condition on the moment matrix (flat extension), which also extracts the global minimizers. The price is size: the moment matrix $\mathbf{M}_d (\mathbf{y})$ has dimension $\binom{n + d}{d}$, which is why exploiting sparsity and symmetry is essential for any practical use.

Not every nonnegative polynomial is SOS: the Motzkin polynomial $x^4 y^2 + x^2 y^4 - 3 x^2 y^2 + 1 \ge 0$ is the classical counterexample. This gap between nonnegativity and SOS is exactly what the hierarchy above bridges when constraints are present, and the reason the lower bounds $\lambda_d$ can be strictly below $f^*$ at low order.

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
