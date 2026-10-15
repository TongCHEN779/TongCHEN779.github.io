window.Demo = (function () {
  var C = {
    blue: '#1f4e9c', red: '#c0392b', green: '#27ae60', orange: '#e67e22',
    gray: '#888', light: '#ddd', text: '#333'
  };

  function randn() {
    var u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function setup(canvas) {
    var W = canvas.width, H = canvas.height;
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    var ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    canvas.setAttribute('role', 'img');
    return { ctx: ctx, W: W, H: H, scale: function () { return canvas.getBoundingClientRect().width / W; } };
  }

  function pointer(canvas, W, H, e) {
    var r = canvas.getBoundingClientRect();
    return [(e.clientX - r.left) * W / r.width, (e.clientY - r.top) * H / r.height];
  }

  function drag(canvas, W, H, hit, onMove, onDown) {
    var active = -1;
    canvas.style.touchAction = 'none';
    canvas.addEventListener('pointerdown', function (e) {
      var p = pointer(canvas, W, H, e), i = hit(p);
      if (i < 0) return;
      active = i;
      canvas.setPointerCapture(e.pointerId);
      if (onDown) onDown(i, p);
      e.preventDefault();
    });
    canvas.addEventListener('pointermove', function (e) {
      var p = pointer(canvas, W, H, e);
      if (active < 0) {
        canvas.style.cursor = hit(p) >= 0 ? 'grab' : 'default';
        return;
      }
      canvas.style.cursor = 'grabbing';
      onMove(active, p);
      e.preventDefault();
    });
    function end() { active = -1; }
    canvas.addEventListener('pointerup', end);
    canvas.addEventListener('pointercancel', end);
  }

  function loop(fn) {
    var id = null, running = false;
    function tick(ts) {
      if (!running) return;
      if (fn(ts) === false) { running = false; return; }
      id = requestAnimationFrame(tick);
    }
    return {
      start: function () { if (running) return; running = true; id = requestAnimationFrame(tick); },
      stop: function () { running = false; if (id) cancelAnimationFrame(id); },
      isRunning: function () { return running; }
    };
  }

  function fail(boxId, err) {
    var box = document.getElementById(boxId);
    if (!box) return;
    var msg = document.createElement('div');
    msg.className = 'demo-error';
    msg.textContent = 'Demo failed to start: ' + err.message;
    box.appendChild(msg);
  }

  function bindRange(id, valId, fmt, cb) {
    var el = document.getElementById(id), out = valId && document.getElementById(valId);
    function update() {
      var v = parseFloat(el.value);
      if (out) out.textContent = fmt ? fmt(v) : v;
      if (cb) cb(v);
    }
    el.addEventListener('input', update);
    return { el: el, update: update };
  }

  function arrow(ctx, x1, y1, x2, y2, head) {
    var ang = Math.atan2(y2 - y1, x2 - x1), h = head || 7;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - h * Math.cos(ang - 0.4), y2 - h * Math.sin(ang - 0.4));
    ctx.lineTo(x2 - h * Math.cos(ang + 0.4), y2 - h * Math.sin(ang + 0.4));
    ctx.closePath(); ctx.fill();
  }

  return { C: C, randn: randn, setup: setup, drag: drag, loop: loop, fail: fail, bindRange: bindRange, arrow: arrow, pointer: pointer };
})();
