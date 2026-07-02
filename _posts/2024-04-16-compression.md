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

A small MLP (1-24-24-1) is trained in your browser to fit a 1D function (gray curve: target; blue curve: network output). Drag the slider to prune the smallest-magnitude weights — the fit survives surprisingly high sparsity, then collapses.

<div id="prdemo-box" style="max-width:640px;margin:1em auto;padding:12px;border:1px solid #ccc;border-radius:8px;font-family:sans-serif;">
  <canvas id="prdemo-canvas" width="600" height="300" style="width:100%;"></canvas>
  <div style="margin-top:8px;">
    <label for="prdemo-s">sparsity = <span id="prdemo-sval">0</span>%</label>
    <input type="range" id="prdemo-s" min="0" max="99" value="0" style="width:100%;">
  </div>
  <div id="prdemo-info" style="margin-top:6px;">training...</div>
</div>

<script>
(function () {
  try {
  var canvas = document.getElementById('prdemo-canvas');
  if (!canvas) return;
  var ctx = canvas.getContext('2d');
  var W = canvas.width, H = canvas.height;
  var h = 24;
  var target = function (x) { return Math.sin(3 * x) + 0.5 * Math.cos(7 * x) * x; };
  var W1 = [], b1 = [], W2 = [], b2 = [], W3 = [], b3 = 0;
  function init() {
    W1 = []; b1 = []; W2 = []; b2 = []; W3 = [];
    for (var i = 0; i < h; i++) {
      W1.push(2 * (Math.random() - 0.5) * 2);
      b1.push(2 * (Math.random() - 0.5));
      W3.push(2 * (Math.random() - 0.5) / Math.sqrt(h));
      b2.push(0);
      var row = [];
      for (var j = 0; j < h; j++) row.push(2 * (Math.random() - 0.5) / Math.sqrt(h));
      W2.push(row);
    }
    b3 = 0;
  }
  function forward(x, m) {
    var t = m || 0;
    var a1 = [], a2 = [];
    for (var i = 0; i < h; i++) {
      var w = Math.abs(W1[i]) < t ? 0 : W1[i];
      a1.push(Math.tanh(w * x + b1[i]));
    }
    for (var i2 = 0; i2 < h; i2++) {
      var s = b2[i2];
      for (var j = 0; j < h; j++) {
        var w2 = Math.abs(W2[i2][j]) < t ? 0 : W2[i2][j];
        s += w2 * a1[j];
      }
      a2.push(Math.tanh(s));
    }
    var out = b3;
    for (var i3 = 0; i3 < h; i3++) {
      var w3 = Math.abs(W3[i3]) < t ? 0 : W3[i3];
      out += w3 * a2[i3];
    }
    return { y: out, a1: a1, a2: a2 };
  }
  var data = [];
  for (var i = 0; i < 60; i++) {
    var x = -1.5 + 3 * i / 59;
    data.push([x, target(x)]);
  }
  function trainStep(lr) {
    for (var n = 0; n < data.length; n++) {
      var x = data[n][0], y = data[n][1];
      var f = forward(x), err = f.y - y;
      var gW3 = [], ga2 = [];
      for (var i = 0; i < h; i++) { gW3.push(err * f.a2[i]); ga2.push(err * W3[i]); }
      var ga1 = [];
      for (var j = 0; j < h; j++) ga1.push(0);
      for (var i2 = 0; i2 < h; i2++) {
        var d2 = ga2[i2] * (1 - f.a2[i2] * f.a2[i2]);
        b2[i2] -= lr * d2;
        for (var j2 = 0; j2 < h; j2++) {
          ga1[j2] += d2 * W2[i2][j2];
          W2[i2][j2] -= lr * d2 * f.a1[j2];
        }
      }
      for (var i3 = 0; i3 < h; i3++) {
        var d1 = ga1[i3] * (1 - f.a1[i3] * f.a1[i3]);
        W1[i3] -= lr * d1 * x;
        b1[i3] -= lr * d1;
        W3[i3] -= lr * gW3[i3];
      }
      b3 -= lr * err;
    }
  }
  function allWeights() {
    var ws = [];
    for (var i = 0; i < h; i++) { ws.push(Math.abs(W1[i])); ws.push(Math.abs(W3[i])); }
    for (var i2 = 0; i2 < h; i2++) for (var j = 0; j < h; j++) ws.push(Math.abs(W2[i2][j]));
    return ws.sort(function (a, b) { return a - b; });
  }
  function px(x) { return (x + 1.6) / 3.2 * W; }
  function py(y) { return H / 2 - y * H / 4.2; }
  function draw(sp) {
    var ws = allWeights();
    var idx = Math.min(ws.length - 1, Math.floor(sp / 100 * ws.length));
    var t = sp === 0 ? 0 : ws[idx];
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = '#bbb'; ctx.lineWidth = 2;
    ctx.beginPath();
    for (var i = 0; i <= 200; i++) {
      var x = -1.6 + 3.2 * i / 200;
      var y = target(x);
      if (i === 0) ctx.moveTo(px(x), py(y)); else ctx.lineTo(px(x), py(y));
    }
    ctx.stroke();
    ctx.strokeStyle = '#1f4e9c';
    ctx.beginPath();
    var mse = 0;
    for (var j = 0; j <= 200; j++) {
      var x2 = -1.6 + 3.2 * j / 200;
      var y2 = forward(x2, t).y;
      mse += Math.pow(y2 - target(x2), 2) / 201;
      if (j === 0) ctx.moveTo(px(x2), py(y2)); else ctx.lineTo(px(x2), py(y2));
    }
    ctx.stroke();
    document.getElementById('prdemo-info').textContent =
      'pruned ' + sp + '% of weights, test MSE = ' + mse.toFixed(4);
  }
  var slider = document.getElementById('prdemo-s');
  slider.addEventListener('input', function () {
    document.getElementById('prdemo-sval').textContent = slider.value;
    draw(parseInt(slider.value, 10));
  });
  init();
  var epoch = 0;
  function trainLoop() {
    for (var k = 0; k < 20; k++) { trainStep(0.02); epoch++; }
    draw(parseInt(slider.value, 10));
    if (epoch < 1200) requestAnimationFrame(trainLoop);
    else document.getElementById('prdemo-info').textContent += ' (training done)';
  }
  requestAnimationFrame(trainLoop);
  } catch (err) {
    var box = document.getElementById('prdemo-box');
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

Quantization
---
Quantization maps full-precision weights/activations to a low-bit grid. The uniform affine quantizer with scale $s$, zero-point $z$, and bit-width $b$ is

$$Q (x) = s \cdot \bigg(\text{clip} \bigg(\bigg\lfloor \frac{x}{s} \bigg\rceil + z, \; 0, \; 2^b - 1\bigg) - z\bigg).$$

Two regimes:

- Post-training quantization (PTQ): quantize a trained model using a small calibration set to fit $(s, z)$ per tensor or per channel; no retraining. Works well down to 8 bits, degrades below.

- Quantization-aware training (QAT): insert fake-quantization in the forward pass and train through it. The rounding operator has zero gradient almost everywhere, so one uses the straight-through estimator (STE) [[Bengio et al. 2013](https://arxiv.org/abs/1308.3432)]: $\frac{\partial Q(x)}{\partial x} \approx \mathbf{1}_{\lbrace x \in [x_{\min}, x_{\max}] \rbrace}$.

The extreme case is binary/ternary networks [[Courbariaux et al. 2016](https://arxiv.org/abs/1602.02830), [Rastegari et al. 2016](https://arxiv.org/abs/1603.05279)], where weights (and possibly activations) live in $\lbrace -1, +1 \rbrace$ and convolutions reduce to XNOR-popcount. For modern LLM-scale PTQ, see GPTQ [[Frantar et al. 2022](https://arxiv.org/abs/2210.17323)] (Hessian-based, closely related to OBS above) and AWQ [[Lin et al. 2023](https://arxiv.org/abs/2306.00978)].

Tensor Decomposition
---
Linear and convolutional layers are (multi-)linear maps, hence compressible by low-rank factorization:

- SVD for fully-connected layers: $\mathbf{W} \approx \mathbf{U}_r \mathbf{\Sigma}_r \mathbf{V}_r^T$ replaces one $m \times n$ layer by two layers of sizes $m \times r$ and $r \times n$; storage drops from $mn$ to $r (m + n)$.

- CP decomposition of conv kernels [[Lebedev et al. 2015](https://arxiv.org/abs/1412.6553)]: a 4-way kernel $\mathcal{K} \in \mathbb{R}^{d \times d \times C_{in} \times C_{out}}$ is written as a sum of $r$ rank-one tensors, turning one convolution into a sequence of cheap 1D convolutions.

- Tucker decomposition [[Kim et al. 2016](https://arxiv.org/abs/1511.06530)]: factor only the channel modes, $\mathcal{K} \approx \mathcal{G} \times_3 \mathbf{U}_{in} \times_4 \mathbf{U}_{out}$, with ranks selected by variational Bayesian matrix factorization.

- Tensor-train (TT) format [[Novikov et al. 2015](https://arxiv.org/abs/1509.06569)]: reshape a weight matrix into a high-order tensor and store it as a chain of 3-way cores; parameter count becomes linear in the order.

The rank $r$ is the crucial hyperparameter and is usually tuned empirically — automating this choice with ML models is precisely the low-rank estimation problem mentioned in my research interests ([ML4POP](https://tongchen779.github.io/)).

Training-free Compression
---
Most methods above require access to the training data (for fine-tuning, calibration, or distillation). A more challenging setting is compression without (real) data:

- Data-free KD [[Nayak et al. 2019](https://arxiv.org/abs/1905.08114), [Yin et al. 2020 (DeepInversion)](https://arxiv.org/abs/1912.08795)]: synthesize pseudo-samples by inverting the teacher (matching batch-norm statistics, maximizing class confidence), then distill on them. Note that the synthesis step is essentially dataset condensation with the teacher as the only supervision signal.

- Data-free pruning: SynFlow [[Tanaka et al. 2020](https://arxiv.org/abs/2006.05467)] scores weights using an all-ones input, requiring no data at all; merging/averaging neurons by similarity [[Srinivas et al. 2015](https://arxiv.org/abs/1507.06149)].

- Data-free quantization: DFQ [[Nagel et al. 2019](https://arxiv.org/abs/1906.04721)] equalizes weight ranges across layers and corrects biases analytically, using only batch-norm statistics.

The interplay between data-free compression and dataset condensation is, in my view, the most interesting open direction: both ask what the minimal sufficient statistics of a dataset (or of a model trained on it) are, one from the data side and one from the model side. See my [dataset condensation post](https://tongchen779.github.io/posts/dc/) for the mirror image of this question.
