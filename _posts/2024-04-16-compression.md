---
title: 'Model Compression: Theory, Practice, and Beyond'
date: 2024-04-16
permalink: /posts/compr/
tags:
  - Knowledge Distillation
  - Model Pruning
  - Quantization
  - Tensor Decomposition
---

Background
---
Given a pretrained model $f_{\theta}$ with parameters $\theta \in \mathbb{R}^D$, model compression seeks a smaller model $f_{\theta_c}$ with $\theta_c \in \mathbb{R}^{D_c}$, $D_c \ll D$, such that the drop of generalization performance is minimized:

$$\min_{\theta_c} \; \vert L (\theta_c) - L (\theta) \vert, \; \text{s.t. } \text{size} (\theta_c) \le B,$$

where the budget $B$ can be measured in parameters, FLOPs, memory, or energy. This is the model-space counterpart of [dataset condensation](https://tongchen779.github.io/posts/dc/): instead of compressing the dataset while fixing the model class, we compress the model while fixing the data distribution. The four classical families are knowledge distillation, pruning, quantization, and tensor decomposition; they are orthogonal and often combined in practice.

Knowledge Distillation
---
Knowledge distillation (KD) [[Hinton et al. 2015](https://arxiv.org/abs/1503.02531)] transfers knowledge from a large teacher $f_t$ to a small student $f_s$ by matching softened output distributions. With logits $z$ and temperature $T$, define $\sigma_T (z) = \text{softmax} (z / T)$. The student is trained with

$$L_{KD} = (1 - \lambda) \, \text{CE} (y, \sigma_1 (z_s)) + \lambda T^2 \, \mathcal{D}_{KL} (\sigma_T (z_t) \vert\vert \sigma_T (z_s)).$$

The factor $T^2$ keeps the gradient magnitude of the soft term comparable to the hard term. Beyond logits, one can match:

- Intermediate features (FitNets) [[Romero et al. 2015](https://arxiv.org/abs/1412.6550)]: $L_{hint} = \vert\vert \phi_t (x) - r (\phi_s (x)) \vert\vert^2$, where $r$ is a learned regressor aligning dimensions.

- Attention maps [[Zagoruyko et al. 2017](https://arxiv.org/abs/1612.03928)]: match spatial attention $\sum_c \vert A_c \vert^2$ of teacher and student.

- Relations between samples [[Park et al. 2019](https://arxiv.org/abs/1904.05068)]: match pairwise distances/angles $\psi (x_i, x_j)$ instead of individual outputs, i.e., distill the geometry of the feature space rather than the features themselves.

Note the structural similarity with distribution-oriented dataset condensation: both minimize a discrepancy between two representations of the same data, only the optimization variable differs (student weights vs. synthetic samples).

### Interactive illustration: temperature and dark knowledge

A teacher produces logits for a 5-class problem (drag the sliders to edit them; the defaults describe a "cat" image that looks a bit like a "dog" and almost nothing like a "truck"). At $T = 1$ the softmax is nearly one-hot and the relative similarities between wrong classes are invisible to the student. Raising $T$ flattens the distribution and exposes this *dark knowledge*. The KL term then shows how far a student with a fixed, over-confident prediction (hollow bars) is from the teacher at the current temperature.

<div class="demo" id="kddemo-box">
  <canvas id="kddemo-canvas" width="640" height="240"></canvas>
  <div class="demo-row">
    <label for="kddemo-T">temperature T = <span id="kddemo-Tv">1.0</span></label>
    <input type="range" id="kddemo-T" min="10" max="200" value="10">
  </div>
  <div class="demo-row" id="kddemo-sliders"></div>
  <div class="demo-status" id="kddemo-info"></div>
</div>

<script>
(function () {
  try {
  var D = Demo, c = D.setup(document.getElementById('kddemo-canvas')), ctx = c.ctx, W = c.W, H = c.H;
  var names = ['cat', 'dog', 'fox', 'car', 'truck'], logits = [6, 3.5, 2.5, -1, -2];
  var student = [5, 0, 0, 0, 0];
  var holder = document.getElementById('kddemo-sliders');
  var TS = D.bindRange('kddemo-T', 'kddemo-Tv', function (v) { return (v / 10).toFixed(1); }, function () { draw(); });
  logits.forEach(function (l, i) {
    var lab = document.createElement('label'), inp = document.createElement('input');
    lab.textContent = names[i]; lab.style.minWidth = '2.6em';
    inp.type = 'range'; inp.min = -100; inp.max = 100; inp.value = l * 10; inp.style.flex = '1 1 90px'; inp.style.minWidth = '70px';
    inp.setAttribute('aria-label', 'logit for ' + names[i]);
    inp.addEventListener('input', function () { logits[i] = parseFloat(inp.value) / 10; draw(); });
    holder.appendChild(lab); holder.appendChild(inp);
  });
  function softmax(z, T) {
    var mx = Math.max.apply(null, z), e = z.map(function (v) { return Math.exp((v - mx) / T); });
    var s = e.reduce(function (a, b) { return a + b; }, 0);
    return e.map(function (v) { return v / s; });
  }
  function draw() {
    var T = TS.el.value / 10, pt = softmax(logits, T), ps = softmax(student, T), kl = 0, i;
    ctx.clearRect(0, 0, W, H);
    ctx.font = '12px sans-serif';
    for (i = 0; i < 5; i++) {
      var x = 40 + i * 120, h1 = pt[i] * 170;
      ctx.fillStyle = D.C.blue; ctx.fillRect(x, 200 - h1, 36, h1);
      ctx.strokeStyle = D.C.red; ctx.lineWidth = 2; ctx.strokeRect(x + 42, 200 - ps[i] * 170, 36, ps[i] * 170);
      ctx.fillStyle = D.C.text; ctx.fillText(names[i], x + 30, 218);
      ctx.fillText(pt[i].toFixed(3), x, 194 - h1);
      kl += pt[i] * Math.log((pt[i] + 1e-12) / (ps[i] + 1e-12));
    }
    ctx.fillStyle = D.C.blue; ctx.fillRect(10, 8, 12, 12); ctx.fillStyle = D.C.text; ctx.fillText('teacher softmax(z / T)', 28, 18);
    ctx.strokeStyle = D.C.red; ctx.strokeRect(190, 8, 12, 12); ctx.fillText('student (fixed, over-confident)', 208, 18);
    document.getElementById('kddemo-info').textContent =
      'KL(teacher || student) at T = ' + T.toFixed(1) + ': ' + kl.toFixed(3) + '   (scaled by T^2: ' + (kl * T * T).toFixed(3) + ')';
  }
  draw();
  } catch (err) { Demo.fail('kddemo-box', err); }
})();
</script>

Model Pruning
---
Pruning removes parameters by applying a binary mask $\mathbf{m} \in \lbrace 0, 1 \rbrace^D$:

$$\min_{\mathbf{m}, \theta} \; L (\theta \odot \mathbf{m}), \; \text{s.t. } \vert\vert \mathbf{m} \vert\vert_0 \le k.$$

This is a combinatorial problem; all practical methods are heuristics for scoring the saliency of each weight.

- Magnitude pruning [[Han et al. 2015](https://arxiv.org/abs/1506.02626)]: saliency $s_i = \vert \theta_i \vert$; iteratively prune and fine-tune. Simple and still a strong baseline.

- Hessian-based pruning (Optimal Brain Damage / Surgeon) [[LeCun et al. 1990](https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html), [Hassibi et al. 1993](https://proceedings.neurips.cc/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html)]: second-order Taylor expansion of the loss gives $\delta L \approx \frac{1}{2} \theta_i^2 H_{ii}$ (OBD, diagonal approximation), or the exact rank-one update $\delta L = \frac{\theta_i^2}{2 [H^{-1}]_{ii}}$ (OBS).

- Lottery Ticket Hypothesis [[Frankle et al. 2019](https://arxiv.org/abs/1803.03635)]: a randomly initialized dense network contains a sparse subnetwork ("winning ticket") that, trained in isolation from the original initialization, matches the dense accuracy.

- Pruning at initialization: SNIP [[Lee et al. 2019](https://arxiv.org/abs/1810.02340)] uses connection sensitivity $\vert \theta_i \cdot \nabla_{\theta_i} L \vert$; GraSP [[Wang et al. 2020](https://arxiv.org/abs/2002.07376)] preserves gradient flow via the Hessian-gradient product; SynFlow [[Tanaka et al. 2020](https://arxiv.org/abs/2006.05467)] is data-free and avoids layer collapse.

Structured pruning removes entire neurons/channels/heads instead of individual weights, trading a worse accuracy-sparsity frontier for actual hardware speedup. For an exact (but expensive) formulation of pruning as a polynomial optimization problem, see the ReLU network pruning section of my [POP post](https://tongchen779.github.io/posts/poly/).

### Interactive illustration: magnitude pruning

A small MLP (1-24-24-1, tanh) is trained in your browser to fit a 1D function (gray: target; blue: network output). Drag the slider to preview pruning the smallest-magnitude weights globally: the fit survives surprisingly high sparsity, then collapses. Then press "Commit mask" to make the current mask permanent and "Fine-tune" to retrain the surviving weights with the mask fixed. Compare the error right after pruning with the error after fine-tuning: this is the iterative prune-and-retrain loop of Han et al., in miniature. The bars at the bottom show how many weights of each layer are still alive.

<div class="demo" id="prdemo-box">
  <canvas id="prdemo-canvas" width="640" height="320"></canvas>
  <div class="demo-row">
    <label for="prdemo-s">sparsity = <span id="prdemo-sval">0</span>%</label>
    <input type="range" id="prdemo-s" min="0" max="99" value="0">
  </div>
  <div class="demo-row">
    <button id="prdemo-commit">Commit mask</button>
    <button id="prdemo-ft">Fine-tune (400 epochs)</button>
    <button id="prdemo-reset">Retrain from scratch</button>
  </div>
  <div class="demo-status" id="prdemo-info">training...</div>
</div>

<script>
(function () {
  try {
  var D = Demo, c = D.setup(document.getElementById('prdemo-canvas')), ctx = c.ctx, W = c.W, H = c.H;
  var h = 24;
  var target = function (x) { return Math.sin(3 * x) + 0.5 * Math.cos(7 * x) * x; };
  var P, Mk, b1, b2, b3, epoch, trainer = null, data = [], i;
  for (i = 0; i < 60; i++) { var xx = -1.5 + 3 * i / 59; data.push([xx, target(xx)]); }
  var slider = document.getElementById('prdemo-s'), info = document.getElementById('prdemo-info');
  D.bindRange('prdemo-s', 'prdemo-sval', null, function () { draw(); });
  function mat(r, cc, f) {
    var m = [];
    for (var a = 0; a < r; a++) { var row = []; for (var b = 0; b < cc; b++) row.push(f()); m.push(row); }
    return m;
  }
  function init() {
    P = { W1: mat(h, 1, function () { return 4 * (Math.random() - 0.5); }),
          W2: mat(h, h, function () { return 2 * (Math.random() - 0.5) / Math.sqrt(h); }),
          W3: mat(1, h, function () { return 2 * (Math.random() - 0.5) / Math.sqrt(h); }) };
    Mk = { W1: mat(h, 1, function () { return 1; }), W2: mat(h, h, function () { return 1; }), W3: mat(1, h, function () { return 1; }) };
    b1 = []; b2 = []; b3 = 0;
    for (var a = 0; a < h; a++) { b1.push(2 * (Math.random() - 0.5)); b2.push(0); }
    epoch = 0;
  }
  function thresholdFor(sp) {
    var ws = [];
    ['W1', 'W2', 'W3'].forEach(function (k) {
      P[k].forEach(function (row, r) { row.forEach(function (w, q) { ws.push(Mk[k][r][q] ? Math.abs(w) : 0); }); });
    });
    ws.sort(function (a, b) { return a - b; });
    if (sp <= 0) return 0;
    return ws[Math.min(ws.length - 1, Math.floor(sp / 100 * ws.length))];
  }
  function eff(k, r, q, t) {
    var w = P[k][r][q];
    return (Mk[k][r][q] && Math.abs(w) >= t) ? w : 0;
  }
  function forward(x, t) {
    var a1 = [], a2 = [], u, v;
    for (u = 0; u < h; u++) a1.push(Math.tanh(eff('W1', u, 0, t) * x + b1[u]));
    for (u = 0; u < h; u++) {
      var s = b2[u];
      for (v = 0; v < h; v++) s += eff('W2', u, v, t) * a1[v];
      a2.push(Math.tanh(s));
    }
    var out = b3;
    for (u = 0; u < h; u++) out += eff('W3', 0, u, t) * a2[u];
    return { y: out, a1: a1, a2: a2 };
  }
  function trainStep(lr) {
    for (var n = 0; n < data.length; n++) {
      var x = data[n][0], f = forward(x, 0), err = f.y - data[n][1], u, v;
      var ga1 = [];
      for (u = 0; u < h; u++) ga1.push(0);
      for (u = 0; u < h; u++) {
        var d2 = err * P.W3[0][u] * Mk.W3[0][u] * (1 - f.a2[u] * f.a2[u]);
        if (Mk.W3[0][u]) P.W3[0][u] -= lr * err * f.a2[u];
        b2[u] -= lr * d2;
        for (v = 0; v < h; v++) {
          if (!Mk.W2[u][v]) continue;
          ga1[v] += d2 * P.W2[u][v];
          P.W2[u][v] -= lr * d2 * f.a1[v];
        }
      }
      for (u = 0; u < h; u++) {
        var d1 = ga1[u] * (1 - f.a1[u] * f.a1[u]);
        if (Mk.W1[u][0]) P.W1[u][0] -= lr * d1 * x;
        b1[u] -= lr * d1;
      }
      b3 -= lr * err;
    }
  }
  function px(x) { return (x + 1.6) / 3.2 * W; }
  function py(y) { return H / 2 - 20 - y * H / 4.6; }
  function alive() {
    var counts = [0, 0, 0];
    P.W1.forEach(function (r, a) { counts[0] += Mk.W1[a][0]; });
    P.W2.forEach(function (r, a) { r.forEach(function (w, q) { counts[1] += Mk.W2[a][q]; }); });
    P.W3[0].forEach(function (w, q) { counts[2] += Mk.W3[0][q]; });
    return counts;
  }
  function draw(msg) {
    var sp = parseInt(slider.value, 10), t = thresholdFor(sp), j;
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = '#bbb'; ctx.lineWidth = 2.5; ctx.beginPath();
    for (j = 0; j <= 200; j++) {
      var x = -1.6 + 3.2 * j / 200;
      if (j === 0) ctx.moveTo(px(x), py(target(x))); else ctx.lineTo(px(x), py(target(x)));
    }
    ctx.stroke();
    ctx.strokeStyle = D.C.blue; ctx.lineWidth = 2; ctx.beginPath();
    var mse = 0;
    for (j = 0; j <= 200; j++) {
      var x2 = -1.6 + 3.2 * j / 200, y2 = forward(x2, t).y;
      mse += Math.pow(y2 - target(x2), 2) / 201;
      if (j === 0) ctx.moveTo(px(x2), py(y2)); else ctx.lineTo(px(x2), py(y2));
    }
    ctx.stroke();
    var cn = alive(), tot = [h, h * h, h], names = ['layer 1', 'layer 2', 'layer 3'];
    ctx.font = '11px sans-serif';
    for (j = 0; j < 3; j++) {
      var frac = cn[j] / tot[j];
      ctx.fillStyle = '#e5e8eb'; ctx.fillRect(20 + j * 200, H - 26, 160, 8);
      ctx.fillStyle = D.C.blue; ctx.fillRect(20 + j * 200, H - 26, 160 * frac, 8);
      ctx.fillStyle = D.C.text; ctx.fillText(names[j] + ': ' + cn[j] + '/' + tot[j] + ' weights committed', 20 + j * 200, H - 8);
    }
    info.textContent = (msg || (epoch < 1200 ? 'training... epoch ' + epoch : '')) +
      '  previewing ' + sp + '% extra sparsity, MSE = ' + mse.toFixed(4);
  }
  function runTrain(n, lr, done) {
    if (trainer) trainer.stop();
    var target_ep = epoch + n;
    trainer = D.loop(function () {
      for (var k = 0; k < 20 && epoch < target_ep; k++) { trainStep(lr); epoch++; }
      draw();
      if (epoch >= target_ep) { if (done) done(); draw('done. '); return false; }
      return true;
    });
    trainer.start();
  }
  document.getElementById('prdemo-commit').addEventListener('click', function () {
    var t = thresholdFor(parseInt(slider.value, 10));
    ['W1', 'W2', 'W3'].forEach(function (k) {
      P[k].forEach(function (row, r) { row.forEach(function (w, q) { if (Math.abs(w) < t) { Mk[k][r][q] = 0; P[k][r][q] = 0; } }); });
    });
    slider.value = 0; document.getElementById('prdemo-sval').textContent = '0';
    draw('mask committed. ');
  });
  document.getElementById('prdemo-ft').addEventListener('click', function () { runTrain(400, 0.01); });
  document.getElementById('prdemo-reset').addEventListener('click', function () {
    init(); slider.value = 0; document.getElementById('prdemo-sval').textContent = '0'; runTrain(1200, 0.02);
  });
  init(); runTrain(1200, 0.02);
  } catch (err) { Demo.fail('prdemo-box', err); }
})();
</script>

Quantization
---
Quantization maps full-precision weights/activations to a low-bit grid. The uniform affine quantizer with scale $s$, zero-point $z$, and bit-width $b$ is

$$Q (x) = s \cdot \bigg(\text{clip} \bigg(\bigg\lfloor \frac{x}{s} \bigg\rceil + z, \; 0, \; 2^b - 1\bigg) - z\bigg).$$

Two regimes:

- Post-training quantization (PTQ): quantize a trained model using a small calibration set to fit $(s, z)$ per tensor or per channel; no retraining. Works well down to 8 bits, degrades below.

- Quantization-aware training (QAT): insert fake-quantization in the forward pass and train through it. The rounding operator has zero gradient almost everywhere, so one uses the straight-through estimator (STE) [[Bengio et al. 2013](https://arxiv.org/abs/1308.3432)]: $\frac{\partial Q(x)}{\partial x} \approx \mathbf{1}_{\lbrace x \in [x_{\min}, x_{\max}] \rbrace}$.

The extreme case is binary/ternary networks [[Courbariaux et al. 2016](https://arxiv.org/abs/1602.02830), [Rastegari et al. 2016](https://arxiv.org/abs/1603.05279)], where weights (and possibly activations) live in $\lbrace -1, +1 \rbrace$ and convolutions reduce to XNOR-popcount. For modern LLM-scale PTQ, see GPTQ [[Frantar et al. 2022](https://arxiv.org/abs/2210.17323)] (Hessian-based, closely related to OBS above) and AWQ [[Lin et al. 2023](https://arxiv.org/abs/2306.00978)].

### Interactive illustration: uniform quantization

A weight vector (gray, 96 values drawn from a Gaussian) is mapped to a $b$-bit grid with the affine quantizer above (blue staircase = dequantized values). The scale $s$ is set from the calibration range. With the default min-max calibration, one outlier is enough to stretch the grid and waste most of the levels on empty space; clip the range to a percentile instead and the bulk becomes accurate, at the price of saturating the outlier. This trade-off between rounding error and clipping error is the essence of PTQ calibration.

<div class="demo" id="qtdemo-box">
  <canvas id="qtdemo-canvas" width="640" height="300"></canvas>
  <div class="demo-row">
    <label for="qtdemo-b">bits b = <span id="qtdemo-bv">4</span></label>
    <input type="range" id="qtdemo-b" min="1" max="8" value="4">
    <label for="qtdemo-clip">calibration range = <span id="qtdemo-clipv">100</span>% of min-max</label>
    <input type="range" id="qtdemo-clip" min="10" max="100" value="100">
  </div>
  <div class="demo-row">
    <label><input type="checkbox" id="qtdemo-out"> add an outlier weight</label>
    <button id="qtdemo-new">Resample weights</button>
  </div>
  <div class="demo-status" id="qtdemo-info"></div>
</div>

<script>
(function () {
  try {
  var D = Demo, c = D.setup(document.getElementById('qtdemo-canvas')), ctx = c.ctx, W = c.W, H = c.H;
  var base = [], n = 96;
  function sample() { base = []; for (var i = 0; i < n; i++) base.push(0.5 * D.randn()); }
  var bS = D.bindRange('qtdemo-b', 'qtdemo-bv', null, function () { draw(); });
  var cS = D.bindRange('qtdemo-clip', 'qtdemo-clipv', null, function () { draw(); });
  var outEl = document.getElementById('qtdemo-out');
  function draw() {
    var w = base.slice();
    if (outEl.checked) w[7] = 4.5;
    var bits = parseInt(bS.el.value, 10), levels = Math.pow(2, bits) - 1;
    var lo = Math.min.apply(null, w) * cS.el.value / 100, hi = Math.max.apply(null, w) * cS.el.value / 100;
    var s = (hi - lo) / levels, z = Math.round(-lo / s), q = [], err = 0, sig = 0, i;
    for (i = 0; i < n; i++) {
      var code = Math.max(0, Math.min(levels, Math.round(w[i] / s) + z));
      q.push(s * (code - z));
      err += Math.pow(q[i] - w[i], 2); sig += w[i] * w[i];
    }
    var ymax = Math.max(Math.abs(Math.min.apply(null, w)), Math.max.apply(null, w)) * 1.1;
    function X(k) { return 30 + k / (n - 1) * (W - 60); }
    function Y(v) { return H / 2 - v / ymax * (H / 2 - 20); }
    ctx.clearRect(0, 0, W, H);
    if (levels <= 63) {
      ctx.strokeStyle = 'rgba(31,78,156,0.18)'; ctx.lineWidth = 1;
      for (var L = 0; L <= levels; L++) {
        var gy = Y(s * (L - z));
        ctx.beginPath(); ctx.moveTo(30, gy); ctx.lineTo(W - 30, gy); ctx.stroke();
      }
    }
    ctx.strokeStyle = '#888'; ctx.beginPath(); ctx.moveTo(30, Y(0)); ctx.lineTo(W - 30, Y(0)); ctx.stroke();
    ctx.strokeStyle = D.C.red; ctx.setLineDash([5, 4]);
    [lo, hi].forEach(function (b) { ctx.beginPath(); ctx.moveTo(30, Y(b)); ctx.lineTo(W - 30, Y(b)); ctx.stroke(); });
    ctx.setLineDash([]);
    for (i = 0; i < n; i++) {
      ctx.fillStyle = '#999'; ctx.beginPath(); ctx.arc(X(i), Y(w[i]), 3, 0, 2 * Math.PI); ctx.fill();
      ctx.strokeStyle = D.C.blue; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(X(i) - 3, Y(q[i])); ctx.lineTo(X(i) + 3, Y(q[i])); ctx.stroke();
    }
    ctx.fillStyle = D.C.red; ctx.font = '12px sans-serif';
    ctx.fillText('calibration range [' + lo.toFixed(2) + ', ' + hi.toFixed(2) + ']', 34, 14);
    var snr = 10 * Math.log(sig / (err + 1e-12)) / Math.LN10;
    document.getElementById('qtdemo-info').textContent =
      levels + 1 + ' levels, scale s = ' + s.toFixed(3) + ', MSE = ' + (err / n).toFixed(5) + ', SNR = ' + snr.toFixed(1) + ' dB, compression = ' + (32 / bits).toFixed(1) + 'x vs fp32';
  }
  outEl.addEventListener('change', draw);
  document.getElementById('qtdemo-new').addEventListener('click', function () { sample(); draw(); });
  sample(); draw();
  } catch (err) { Demo.fail('qtdemo-box', err); }
})();
</script>

Tensor Decomposition
---
Linear and convolutional layers are (multi-)linear maps, hence compressible by low-rank factorization:

- SVD for fully-connected layers: $\mathbf{W} \approx \mathbf{U}_r \mathbf{\Sigma}_r \mathbf{V}_r^T$ replaces one $m \times n$ layer by two layers of sizes $m \times r$ and $r \times n$; storage drops from $mn$ to $r (m + n)$.

- CP decomposition of conv kernels [[Lebedev et al. 2015](https://arxiv.org/abs/1412.6553)]: a 4-way kernel $\mathcal{K} \in \mathbb{R}^{d \times d \times C_{in} \times C_{out}}$ is written as a sum of $r$ rank-one tensors, turning one convolution into a sequence of cheap 1D convolutions.

- Tucker decomposition [[Kim et al. 2016](https://arxiv.org/abs/1511.06530)]: factor only the channel modes, $\mathcal{K} \approx \mathcal{G} \times_3 \mathbf{U}_{in} \times_4 \mathbf{U}_{out}$, with ranks selected by variational Bayesian matrix factorization.

- Tensor-train (TT) format [[Novikov et al. 2015](https://arxiv.org/abs/1509.06569)]: reshape a weight matrix into a high-order tensor and store it as a chain of 3-way cores; parameter count becomes linear in the order.

The rank $r$ is the crucial hyperparameter and is usually tuned empirically — automating this choice with ML models is precisely the low-rank estimation problem mentioned in my research interests ([ML4POP](https://tongchen779.github.io/)).

### Interactive illustration: low-rank approximation

The left image is a $48 \times 48$ "weight matrix" with smooth structure plus a little noise; the middle is its rank-$r$ SVD truncation $\mathbf{U}_r \mathbf{\Sigma}_r \mathbf{V}_r^T$ (optimal in Frobenius norm by the Eckart-Young theorem); the right is the residual. Drag $r$: a few components capture most of the energy because the singular values decay fast, and the storage $r (m + n)$ is much smaller than $mn$ until $r$ approaches $m / 2$, where factorization stops paying off. Switch to "noise-like" matrices to see the failure case: when singular values decay slowly, low-rank factorization cannot compress.

<div class="demo" id="svddemo-box">
  <canvas id="svddemo-canvas" width="640" height="250"></canvas>
  <div class="demo-row">
    <label for="svddemo-r">rank r = <span id="svddemo-rv">4</span></label>
    <input type="range" id="svddemo-r" min="1" max="48" value="4">
    <label for="svddemo-kind">matrix</label>
    <select id="svddemo-kind">
      <option value="smooth">smooth + noise (compressible)</option>
      <option value="noise">i.i.d. noise (incompressible)</option>
    </select>
  </div>
  <div class="demo-status" id="svddemo-info"></div>
</div>

<script>
(function () {
  try {
  var D = Demo, c = D.setup(document.getElementById('svddemo-canvas')), ctx = c.ctx, W = c.W, H = c.H;
  var m = 48, A, U, sv, V;
  function build(kind) {
    A = [];
    for (var i = 0; i < m; i++) {
      var row = [];
      for (var j = 0; j < m; j++) {
        var x = i / m, y = j / m, v;
        if (kind === 'noise') v = D.randn();
        else v = Math.sin(6 * x) * Math.cos(4 * y) + 0.8 * Math.exp(-30 * ((x - 0.3) * (x - 0.3) + (y - 0.7) * (y - 0.7))) + 0.5 * x * y + 0.05 * D.randn();
        row.push(v);
      }
      A.push(row);
    }
    svd();
  }
  function svd() {
    var Wm = A.map(function (r) { return r.slice(); });
    var Vm = [];
    for (var i = 0; i < m; i++) { var r = []; for (var j = 0; j < m; j++) r.push(i === j ? 1 : 0); Vm.push(r); }
    for (var sweep = 0; sweep < 12; sweep++) {
      for (var p = 0; p < m - 1; p++) {
        for (var q = p + 1; q < m; q++) {
          var a = 0, b = 0, g = 0, k;
          for (k = 0; k < m; k++) { a += Wm[k][p] * Wm[k][p]; b += Wm[k][q] * Wm[k][q]; g += Wm[k][p] * Wm[k][q]; }
          if (Math.abs(g) < 1e-12) continue;
          var zeta = (b - a) / (2 * g), t = Math.sign(zeta || 1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
          var cs = 1 / Math.sqrt(1 + t * t), sn = cs * t;
          for (k = 0; k < m; k++) {
            var wp = Wm[k][p], wq = Wm[k][q];
            Wm[k][p] = cs * wp - sn * wq; Wm[k][q] = sn * wp + cs * wq;
            var vp = Vm[k][p], vq = Vm[k][q];
            Vm[k][p] = cs * vp - sn * vq; Vm[k][q] = sn * vp + cs * vq;
          }
        }
      }
    }
    var idx = [], norms = [];
    for (var j2 = 0; j2 < m; j2++) {
      var s = 0;
      for (var k2 = 0; k2 < m; k2++) s += Wm[k2][j2] * Wm[k2][j2];
      norms.push(Math.sqrt(s)); idx.push(j2);
    }
    idx.sort(function (x, y) { return norms[y] - norms[x]; });
    sv = idx.map(function (j) { return norms[j]; });
    U = idx.map(function (j) { return Wm.map(function (row) { return row[j] / (norms[j] || 1); }); });
    V = idx.map(function (j) { return Vm.map(function (row) { return row[j]; }); });
  }
  var rS = D.bindRange('svddemo-r', 'svddemo-rv', null, function () { draw(); });
  function heat(M, ox, oy, size, vmax, diverge) {
    var cs = size / m;
    for (var i = 0; i < m; i++) for (var j = 0; j < m; j++) {
      var t = Math.max(-1, Math.min(1, M[i][j] / vmax));
      ctx.fillStyle = t >= 0 ? 'rgba(31,78,156,' + t + ')' : 'rgba(192,57,43,' + (-t) + ')';
      ctx.fillRect(ox + j * cs, oy + i * cs, cs + 0.5, cs + 0.5);
    }
    ctx.strokeStyle = '#ccc'; ctx.strokeRect(ox, oy, size, size);
  }
  function draw() {
    var r = parseInt(rS.el.value, 10), R = [], i, j, k;
    for (i = 0; i < m; i++) { var row = []; for (j = 0; j < m; j++) row.push(0); R.push(row); }
    for (k = 0; k < r; k++) for (i = 0; i < m; i++) for (j = 0; j < m; j++) R[i][j] += sv[k] * U[k][i] * V[k][j];
    var E = A.map(function (row, a) { return row.map(function (v, b) { return v - R[a][b]; }); });
    var vmax = 0;
    A.forEach(function (row) { row.forEach(function (v) { vmax = Math.max(vmax, Math.abs(v)); }); });
    ctx.clearRect(0, 0, W, H);
    var size = 190;
    heat(A, 10, 30, size, vmax); heat(R, 220, 30, size, vmax); heat(E, 430, 30, size, vmax);
    ctx.fillStyle = D.C.text; ctx.font = '12px sans-serif';
    ctx.fillText('original A', 10, 20); ctx.fillText('rank-' + r + ' approximation', 220, 20); ctx.fillText('residual (same color scale)', 430, 20);
    var tot = 0, kept = 0;
    sv.forEach(function (s, idx) { tot += s * s; if (idx < r) kept += s * s; });
    for (k = 0; k < m; k++) {
      ctx.fillStyle = k < r ? D.C.blue : '#c9cfd6';
      ctx.fillRect(10 + k * 13, 245 - sv[k] / sv[0] * 22, 9, sv[k] / sv[0] * 22);
    }
    document.getElementById('svddemo-info').textContent =
      'energy kept = ' + (100 * kept / tot).toFixed(1) + '%   relative error = ' + Math.sqrt(1 - kept / tot).toFixed(3) +
      '   storage = ' + r * 2 * m + ' / ' + m * m + ' (' + (100 * r * 2 * m / (m * m)).toFixed(0) + '%)';
  }
  document.getElementById('svddemo-kind').addEventListener('change', function () { build(this.value); draw(); });
  build('smooth'); draw();
  } catch (err) { Demo.fail('svddemo-box', err); }
})();
</script>

Training-free Compression
---
Most methods above require access to the training data (for fine-tuning, calibration, or distillation). A more challenging setting is compression without (real) data:

- Data-free KD [[Nayak et al. 2019](https://arxiv.org/abs/1905.08114), [Yin et al. 2020 (DeepInversion)](https://arxiv.org/abs/1912.08795)]: synthesize pseudo-samples by inverting the teacher (matching batch-norm statistics, maximizing class confidence), then distill on them. Note that the synthesis step is essentially dataset condensation with the teacher as the only supervision signal.

- Data-free pruning: SynFlow [[Tanaka et al. 2020](https://arxiv.org/abs/2006.05467)] scores weights using an all-ones input, requiring no data at all; merging/averaging neurons by similarity [[Srinivas et al. 2015](https://arxiv.org/abs/1507.06149)].

- Data-free quantization: DFQ [[Nagel et al. 2019](https://arxiv.org/abs/1906.04721)] equalizes weight ranges across layers and corrects biases analytically, using only batch-norm statistics.

The interplay between data-free compression and dataset condensation is, in my view, the most interesting open direction: both ask what the minimal sufficient statistics of a dataset (or of a model trained on it) are, one from the data side and one from the model side. See my [dataset condensation post](https://tongchen779.github.io/posts/dc/) for the mirror image of this question.
