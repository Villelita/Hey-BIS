/**
 * ============================================================================
 * BIS VOICE ASSISTANT — MORPHING CONSTELLATION & CENTRAL STAR ENGINE (Canvas 2D)
 * ============================================================================
 * States:
 *   - idle (Standby): 12 curved 4-petal pinwheel figures floating gracefully
 *     around the screen with soft neon glow and subtle ambient sound reactivity.
 *   - listening: The floating figures illuminate with vivid neon bloom,
 *     reactive soundwave rings, faster spin, and dynamic audio pulsing.
 *   - thinking: All floating figures smoothly converge into the center and
 *     merge into a single unified 8-petal star/flower figure rotating smoothly.
 *   - speaking: The unified central 8-petal figure expands, shifts to vibrant
 *     accent lighting, and pulses its petals rhythmically with the voice.
 * ============================================================================
 */

class NeuralCore {
  constructor(canvasElement, options = {}) {
    this.canvas = canvasElement;
    this.ctx = this.canvas.getContext('2d');
    this.options = Object.assign({
      palette: 'green',
      onInit: null
    }, options);

    this.state = 'idle'; // 'idle', 'listening', 'thinking', 'speaking', 'error'
    this.palette = this.options.palette;

    // Color Palettes (RGB 0..255)
    this.palettes = {
      green: {
        base: [124, 232, 70],       // Vibrant Lime-Emerald matching reference (#7ce846)
        glow: [0, 255, 136],        // Neon Emerald (#00ff88)
        listen: [0, 255, 200],      // Electric Mint-Cyan (#00ffc8)
        trigger: [0, 240, 255],     // Electric Cyan (#00f0ff)
        triggerAlt: [255, 0, 127],  // Neon Magenta (#ff007f)
        think: [132, 245, 86]       // Luminous Lime (#84f556)
      },
      cyberpunk: {
        base: [255, 0, 127],        // Neon Pink (#ff007f)
        glow: [255, 64, 160],       // Hot Magenta
        listen: [0, 240, 255],      // Electric Cyan (#00f0ff)
        trigger: [0, 255, 136],     // Neon Emerald (#00ff88)
        triggerAlt: [174, 255, 0],  // Cyber Lime (#aeff00)
        think: [192, 80, 255]       // Electric Violet (#c050ff)
      }
    };

    // Normalized constellation anchors matching Reference Image 1:
    // 8 main curved 4-petal figures + 4 smaller corner accent figures
    this.figureDefs = [
      // Inner diamond (4 medium-large)
      { nx: 0.50, ny: 0.21, size: 1.00, phase: 0.0, speed: 0.45 },
      { nx: 0.32, ny: 0.50, size: 1.00, phase: 1.2, speed: -0.40 },
      { nx: 0.68, ny: 0.50, size: 1.00, phase: 2.4, speed: 0.42 },
      { nx: 0.50, ny: 0.78, size: 1.00, phase: 3.6, speed: -0.44 },
      // Outer ring (4 medium-large)
      { nx: 0.24, ny: 0.17, size: 1.05, phase: 0.7, speed: -0.38 },
      { nx: 0.76, ny: 0.16, size: 1.05, phase: 1.9, speed: 0.40 },
      { nx: 0.24, ny: 0.83, size: 1.05, phase: 3.1, speed: 0.36 },
      { nx: 0.76, ny: 0.84, size: 1.05, phase: 4.5, speed: -0.39 },
      // 4 smaller corner accents
      { nx: 0.15, ny: 0.07, size: 0.44, phase: 0.4, speed: 0.65 },
      { nx: 0.85, ny: 0.07, size: 0.44, phase: 2.1, speed: -0.60 },
      { nx: 0.16, ny: 0.91, size: 0.44, phase: 3.8, speed: -0.62 },
      { nx: 0.86, ny: 0.90, size: 0.44, phase: 5.2, speed: 0.64 }
    ];

    // Subtle floating ambient dust/spark particles
    this.sparks = Array.from({ length: 32 }, (_, i) => ({
      nx: (Math.sin(i * 12.9898) * 43758.5453) % 1,
      ny: (Math.cos(i * 78.233) * 24634.6345) % 1,
      r: 1.2 + (i % 4) * 0.7,
      phase: i * 0.85,
      speed: 0.3 + (i % 5) * 0.12
    })).map(s => ({
      ...s,
      nx: Math.abs(s.nx),
      ny: Math.abs(s.ny)
    }));

    // Smooth animation state parameters
    this.params = {
      convergence: 0.0,     // 0 = scattered constellation, 1 = merged central 8-petal star
      glowStrength: 0.35,   // 0..1.5 neon bloom intensity
      spinMultiplier: 1.0,  // rotation speed multiplier
      centralScale: 1.0,    // scale of the merged central 8-petal star
      audioPulse: 0.0,      // live microphone amplitude
      ringEnergy: 0.0       // reactive listening ring wave intensity
    };

    this.targets = { ...this.params };

    const initPal = this.palettes[this.palette] || this.palettes.green;
    this.currentColor = [...initPal.base];
    this.targetColor = [...initPal.base];
    this.currentGlow = [...initPal.glow];
    this.targetGlow = [...initPal.glow];

    this.liveAudioLevel = 0.0;
    this.startTime = performance.now();
    this.lastFrameTime = performance.now();
    this.centralAngle = 0;
    this.animationFrameId = null;

    this.resizeCanvas();
    window.addEventListener('resize', () => this.resizeCanvas());

    if (this.options.onInit) this.options.onInit(this);
    this.animate();
  }

  // ==========================================================================
  // PALETTE & STATE CONTROL
  // ==========================================================================
  setPalette(paletteName) {
    if (this.palettes[paletteName]) {
      this.palette = paletteName;
      const curState = this.state || 'idle';
      this.state = null;
      this.setState(curState);
      this.currentColor = [...this.targetColor];
      this.currentGlow = [...this.targetGlow];
    }
  }

  setState(newState) {
    if (this.state === newState) return;
    this.state = newState;
    const pal = this.palettes[this.palette] || this.palettes.green;

    switch (newState) {
      case 'idle':
        // Standby: scattered floating 4-petal figures around the screen
        this.targets.convergence = 0.0;
        this.targets.glowStrength = 0.38;
        this.targets.spinMultiplier = 0.9;
        this.targets.centralScale = 0.95;
        this.targets.ringEnergy = 0.0;
        this.targetColor = [...pal.base];
        this.targetGlow = [...pal.glow];
        break;

      case 'listening':
        // Listening: figures brighten, glow intensely, emit reactive audio ripples & shift color
        this.targets.convergence = 0.14;
        this.targets.glowStrength = 1.25;
        this.targets.spinMultiplier = 2.4;
        this.targets.centralScale = 1.05;
        this.targets.ringEnergy = 1.0;
        this.targetColor = [...pal.listen];
        this.targetGlow = [...pal.trigger];
        break;

      case 'thinking':
        // Thinking: all figures glide together & merge into the single central 8-petal star
        this.targets.convergence = 1.0;
        this.targets.glowStrength = 1.05;
        this.targets.spinMultiplier = 2.8;
        this.targets.centralScale = 1.04;
        this.targets.ringEnergy = 0.35;
        this.targetColor = [...pal.think];
        this.targetGlow = [...pal.glow];
        break;

      case 'speaking':
        // Speaking: unified central 8-petal star expands larger, shifts color & pulses with voice
        this.targets.convergence = 1.0;
        this.targets.glowStrength = 1.4;
        this.targets.spinMultiplier = 1.5;
        this.targets.centralScale = 1.22;
        this.targets.ringEnergy = 0.85;
        this.targetColor = [...pal.trigger];
        this.targetGlow = [...pal.triggerAlt];
        break;

      case 'error':
        this.targets.convergence = 0.5;
        this.targets.glowStrength = 0.9;
        this.targets.spinMultiplier = 0.5;
        this.targets.centralScale = 0.9;
        this.targets.ringEnergy = 0.2;
        this.targetColor = [255, 160, 40];
        this.targetGlow = [255, 70, 70];
        break;
    }
  }

  setAudioLevel(level) {
    this.liveAudioLevel = Math.max(0, Math.min(1.0, level));
  }

  // ==========================================================================
  // CANVAS SIZING & MAIN ANIMATION LOOP
  // ==========================================================================
  resizeCanvas() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = this.canvas.clientWidth || window.innerWidth;
    const h = this.canvas.clientHeight || window.innerHeight;
    if (this.canvas.width !== Math.floor(w * dpr) || this.canvas.height !== Math.floor(h * dpr)) {
      this.canvas.width = Math.floor(w * dpr);
      this.canvas.height = Math.floor(h * dpr);
    }
    this.width = w;
    this.height = h;
    this.dpr = dpr;
  }

  animate() {
    this.animationFrameId = requestAnimationFrame(() => this.animate());

    const now = performance.now();
    const dt = Math.min((now - this.lastFrameTime) / 1000, 0.1);
    this.lastFrameTime = now;
    const elapsed = (now - this.startTime) / 1000;

    // Smoothly interpolate parameters
    const ease = Math.min(1.0, 4.2 * dt);
    this.params.convergence += (this.targets.convergence - this.params.convergence) * ease;
    this.params.glowStrength += (this.targets.glowStrength - this.params.glowStrength) * ease;
    this.params.spinMultiplier += (this.targets.spinMultiplier - this.params.spinMultiplier) * ease;
    this.params.centralScale += (this.targets.centralScale - this.params.centralScale) * ease;
    this.params.ringEnergy += (this.targets.ringEnergy - this.params.ringEnergy) * ease;

    // Smoothly interpolate RGB colors
    const colorEase = Math.min(1.0, 5.0 * dt);
    for (let i = 0; i < 3; i++) {
      this.currentColor[i] += (this.targetColor[i] - this.currentColor[i]) * colorEase;
      this.currentGlow[i] += (this.targetGlow[i] - this.currentGlow[i]) * colorEase;
    }

    // Live audio reactivity (reacts to ANY sound even in standby)
    const targetAudio = this.state === 'speaking'
      ? Math.max(this.liveAudioLevel, Math.sin(elapsed * 8.0) * 0.22 + 0.28)
      : this.liveAudioLevel;
    this.params.audioPulse += (targetAudio - this.params.audioPulse) * Math.min(1.0, 14.0 * dt);

    // Advance central star rotation
    this.centralAngle += dt * (0.45 * this.params.spinMultiplier + this.params.audioPulse * 0.9);

    this.render(elapsed);
  }

  // ==========================================================================
  // SHAPE DRAWING HELPERS (Matching Reference Images 1 & 2)
  // ==========================================================================

  /**
   * Draws the organic curved 4-petal swirling pinwheel figure (Reference Image 1)
   */
  drawPinwheel(ctx, x, y, radius, angle, fillStyle, glowStyle, glowBlur, alpha) {
    if (alpha <= 0.01 || radius <= 0.5) return;
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);
    ctx.globalAlpha = alpha;
    ctx.fillStyle = fillStyle;
    ctx.shadowColor = glowStyle;
    ctx.shadowBlur = glowBlur;

    const petals = 4;
    ctx.beginPath();
    for (let i = 0; i < petals; i++) {
      const a0 = (i * Math.PI * 2) / petals;
      const aTip = a0 + 0.35;
      const aNext = a0 + (Math.PI * 2) / petals;

      const tipX = Math.cos(aTip) * radius;
      const tipY = Math.sin(aTip) * radius;

      // Swept curved petal control points for the signature pinwheel swirl
      const cp1x = Math.cos(a0 - 0.15) * radius * 0.72;
      const cp1y = Math.sin(a0 - 0.15) * radius * 0.72;
      const cp2x = Math.cos(aTip + 0.42) * radius * 0.62;
      const cp2y = Math.sin(aTip + 0.42) * radius * 0.62;
      const rootX = Math.cos(aNext - 0.25) * radius * 0.24;
      const rootY = Math.sin(aNext - 0.25) * radius * 0.24;

      if (i === 0) {
        const prevRootX = Math.cos(-0.25) * radius * 0.24;
        const prevRootY = Math.sin(-0.25) * radius * 0.24;
        ctx.moveTo(prevRootX, prevRootY);
      }
      ctx.quadraticCurveTo(cp1x, cp1y, tipX, tipY);
      ctx.quadraticCurveTo(cp2x, cp2y, rootX, rootY);
    }
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  /**
   * Draws the unified central 8-petal elongated star/flower figure (Reference Image 2)
   */
  drawCentralStar(ctx, cx, cy, radius, angle, fillStyle, glowStyle, glowBlur, alpha, audioPulse, elapsed) {
    if (alpha <= 0.01 || radius <= 1.0) return;
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(angle);
    ctx.globalAlpha = alpha;
    ctx.fillStyle = fillStyle;
    ctx.shadowColor = glowStyle;
    ctx.shadowBlur = glowBlur;

    const petals = 8;
    const petalWidth = radius * (0.145 + audioPulse * 0.03);

    for (let i = 0; i < petals; i++) {
      const a = (i * Math.PI * 2) / petals;
      const harmonic = 1.0 + Math.sin(elapsed * 5.0 + i * 0.9) * (0.04 + audioPulse * 0.12);
      const len = radius * harmonic;

      ctx.save();
      ctx.rotate(a);
      ctx.beginPath();
      ctx.ellipse(0, -len * 0.5, petalWidth, len * 0.52, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // Smooth solid center hub uniting all 8 petals
    ctx.beginPath();
    ctx.arc(0, 0, radius * 0.17, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  // ==========================================================================
  // RENDER FRAME
  // ==========================================================================
  render(elapsed) {
    const ctx = this.ctx;
    if (!ctx) return;
    this.resizeCanvas();

    ctx.save();
    ctx.scale(this.dpr, this.dpr);
    ctx.clearRect(0, 0, this.width, this.height);

    const cx = this.width * 0.5;
    const cy = this.height * 0.5;
    const minDim = Math.min(this.width, this.height);
    const baseFigRadius = Math.max(22, minDim * 0.068);
    const centralRadius = Math.max(75, minDim * 0.25) * (this.params.centralScale + this.params.audioPulse * 0.16);

    const r = Math.round(this.currentColor[0]);
    const g = Math.round(this.currentColor[1]);
    const b = Math.round(this.currentColor[2]);
    const gr = Math.round(this.currentGlow[0]);
    const gg = Math.round(this.currentGlow[1]);
    const gb = Math.round(this.currentGlow[2]);

    const fillStyle = `rgb(${r}, ${g}, ${b})`;
    const glowStyle = `rgba(${gr}, ${gg}, ${gb}, 0.85)`;

    const conv = this.params.convergence;
    const audio = this.params.audioPulse;

    // 1. Subtle floating ambient sparks that also pull inward during thinking
    for (const sp of this.sparks) {
      const driftX = Math.sin(elapsed * sp.speed + sp.phase) * 14;
      const driftY = Math.cos(elapsed * sp.speed * 0.8 + sp.phase) * 14;
      const homeX = sp.nx * this.width + driftX;
      const homeY = sp.ny * this.height + driftY;
      const sx = homeX + (cx - homeX) * conv * 0.88;
      const sy = homeY + (cy - homeY) * conv * 0.88;
      const twinkle = 0.22 + 0.35 * (0.5 + 0.5 * Math.sin(elapsed * 2.5 + sp.phase));
      const sparkAlpha = twinkle * (1.0 - conv * 0.5) + audio * 0.35;

      ctx.save();
      ctx.globalAlpha = Math.min(0.9, sparkAlpha);
      ctx.fillStyle = `rgb(${gr}, ${gg}, ${gb})`;
      ctx.shadowColor = glowStyle;
      ctx.shadowBlur = 8 + audio * 14;
      ctx.beginPath();
      ctx.arc(sx, sy, sp.r * (1.0 + audio * 0.8), 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // 2. Scattered 4-petal pinwheel figures (visible in idle & listening; converge into center in thinking)
    // Smooth fade out of individual pinwheels only at the very end of convergence (0.72..1.0)
    const pinwheelAlpha = Math.max(0.0, Math.min(1.0, (1.0 - conv) / 0.35));

    if (pinwheelAlpha > 0.01) {
      for (const fig of this.figureDefs) {
        // Gentle floating drift in standby, livelier reaction when sound is heard
        const driftAmp = (10 + audio * 18) * (1.0 - conv);
        const homeX = fig.nx * this.width + Math.sin(elapsed * 0.75 + fig.phase) * driftAmp;
        const homeY = fig.ny * this.height + Math.cos(elapsed * 0.62 + fig.phase * 1.3) * driftAmp;

        // Smooth spiral pull toward center as convergence increases
        const dx = homeX - cx;
        const dy = homeY - cy;
        const spiralAngle = conv * 1.35;
        const cosA = Math.cos(spiralAngle);
        const sinA = Math.sin(spiralAngle);
        const rx = dx * cosA - dy * sinA;
        const ry = dx * sinA + dy * cosA;

        const x = cx + rx * (1.0 - conv);
        const y = cy + ry * (1.0 - conv);

        // React to ANY sound in idle, and pulse strongly in listening
        const soundBoost = 1.0 + audio * (this.state === 'listening' ? 0.42 : 0.26);
        const breath = 1.0 + Math.sin(elapsed * 1.8 + fig.phase) * 0.05;
        const figRadius = baseFigRadius * fig.size * breath * soundBoost * (1.0 - conv * 0.35);

        const figAngle = elapsed * fig.speed * this.params.spinMultiplier + fig.phase + audio * 0.6;
        const glowBlur = (10 + this.params.glowStrength * 24 + audio * 30) * fig.size;

        // In listening mode (or loud sound), draw expanding reactive soundwave ring around each figure
        if (this.params.ringEnergy > 0.05 || audio > 0.22) {
          const ringPhase = (elapsed * 1.4 + fig.phase * 0.35) % 1.0;
          const ringRadius = figRadius * (1.05 + ringPhase * (0.95 + audio * 0.8));
          const ringAlpha = (1.0 - ringPhase) * Math.max(this.params.ringEnergy * 0.55, audio * 0.45) * pinwheelAlpha;
          if (ringAlpha > 0.02) {
            ctx.save();
            ctx.globalAlpha = ringAlpha;
            ctx.strokeStyle = `rgb(${gr}, ${gg}, ${gb})`;
            ctx.lineWidth = 1.5 + audio * 1.8;
            ctx.shadowColor = glowStyle;
            ctx.shadowBlur = 12;
            ctx.beginPath();
            ctx.arc(x, y, ringRadius, 0, Math.PI * 2);
            ctx.stroke();
            ctx.restore();
          }
        }

        this.drawPinwheel(ctx, x, y, figRadius, figAngle, fillStyle, glowStyle, glowBlur, pinwheelAlpha);
      }
    }

    // 3. Unified Central 8-Petal Star/Flower Figure (Reference Image 2)
    // Emerges seamlessly as convergence passes 0.55 -> 1.0 (thinking & speaking)
    const starAlpha = Math.max(0.0, Math.min(1.0, (conv - 0.45) / 0.45));
    if (starAlpha > 0.01) {
      const morphScale = 0.55 + 0.45 * starAlpha;
      const starGlow = 18 + this.params.glowStrength * 28 + audio * 36;

      // Expanding aura pulse ring behind central figure during thinking/speaking
      if (this.params.ringEnergy > 0.05) {
        const rPhase = (elapsed * 0.9) % 1.0;
        const rRad = centralRadius * (0.75 + rPhase * (0.75 + audio * 0.4));
        const rAlpha = (1.0 - rPhase) * this.params.ringEnergy * 0.45 * starAlpha;
        ctx.save();
        ctx.globalAlpha = rAlpha;
        ctx.strokeStyle = `rgb(${gr}, ${gg}, ${gb})`;
        ctx.lineWidth = 2.0;
        ctx.shadowColor = glowStyle;
        ctx.shadowBlur = 16;
        ctx.beginPath();
        ctx.arc(cx, cy, rRad, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }

      this.drawCentralStar(
        ctx,
        cx,
        cy,
        centralRadius * morphScale,
        this.centralAngle,
        fillStyle,
        glowStyle,
        starGlow,
        starAlpha,
        audio,
        elapsed
      );
    }

    ctx.restore();
  }
}

window.NeuralCore = NeuralCore;
