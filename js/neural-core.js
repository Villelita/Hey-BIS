/**
 * ============================================================================
 * BIS VOICE ASSISTANT — 3D NEURAL CORE ENGINE (WebGL)
 * ============================================================================
 * Renders the exact organic deformed quad-wireframe sphere with dual-tone neon
 * emerald-green to cyber-lime gradient lighting, surface intersection nodes,
 * and a dense perimeter halo of glowing synapse particles.
 * ============================================================================
 */

class NeuralCore {
  constructor(canvasElement, options = {}) {
    this.canvas = canvasElement;
    this.options = Object.assign({
      latBands: 44,          // Latitude rings
      lonBands: 64,          // Longitude ribs (forms clean quad grid)
      baseRadius: 1.6,       // Base radius of core sphere
      haloParticleCount: 650,// Dense perimeter halo synapse particles
      palette: 'green',      // 'green' or 'cyberpunk'
      onInit: null
    }, options);

    this.gl = null;
    this.animationFrameId = null;

    // View & Interaction
    this.rotation = { x: 0.15, y: 0.0 };
    this.targetRotation = { x: 0.15, y: 0.0 };
    this.isDragging = false;
    this.lastPointer = { x: 0, y: 0 };

    // Assistant State
    this.state = 'idle'; // 'idle', 'listening', 'thinking', 'speaking', 'error'
    this.palette = this.options.palette;

    // Color Palettes (RGB 0.0 to 1.0)
    this.palettes = {
      green: {
        primary: [0.0, 1.0, 0.53],    // Neon Emerald (#00ff88)
        secondary: [0.68, 1.0, 0.0],  // Cyber Lime (#aeff00)
        accent: [0.0, 1.0, 0.85]      // Mint Cyan (#00ffd9)
      },
      cyberpunk: {
        primary: [0.0, 0.94, 1.0],    // Neon Cyan (#00f0ff)
        secondary: [1.0, 0.0, 0.52],  // Neon Magenta (#ff007f)
        accent: [0.65, 0.1, 0.95]     // Electric Violet (#a855f7)
      }
    };

    // Smooth Lerping Parameters
    this.params = {
      rotationSpeed: 0.005,
      deformScale: 0.22,
      deformFreq: 1.5,
      glowIntensity: 1.2,
      noiseSpeed: 0.55,
      audioPulse: 0.0,
      colorShift: 0.0
    };

    this.targets = {
      rotationSpeed: 0.005,
      deformScale: 0.22,
      deformFreq: 1.5,
      glowIntensity: 1.2,
      noiseSpeed: 0.55,
      audioPulse: 0.0,
      colorShift: 0.0
    };

    this.startTime = performance.now();
    this.lastFrameTime = performance.now();
    this.liveAudioLevel = 0.0;

    this.initWebGL();
    this.initGeometry();
    this.initHaloParticles();
    this.initEvents();

    if (this.options.onInit) this.options.onInit(this);
    this.animate();
  }

  // ==========================================================================
  // WebGL Context & Shaders Setup
  // ==========================================================================
  initWebGL() {
    const gl = this.canvas.getContext('webgl', {
      alpha: true,
      antialias: true,
      depth: false,               // Transparent holographic blending
      premultipliedAlpha: false
    });

    if (!gl) {
      console.error('WebGL not supported on this device.');
      return;
    }

    this.gl = gl;

    // Pure holographic additive blending with no depth clipping
    gl.disable(gl.DEPTH_TEST);
    gl.depthMask(false);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE); // Additive laser neon glow

    // 1. Core Quad Wireframe & Triangles Shader
    this.coreProgram = this.createProgram(this.getCoreVertexShader(), this.getCoreFragmentShader());
    this.coreUniforms = this.getUniformLocations(this.coreProgram, [
      'u_matrix_mvp', 'u_time', 'u_deform_scale', 'u_deform_freq',
      'u_audio_pulse', 'u_speed', 'u_glow_intensity', 'u_color_primary',
      'u_color_secondary', 'u_color_accent', 'u_color_shift', 'u_is_wire'
    ]);
    this.coreAttribs = {
      position: gl.getAttribLocation(this.coreProgram, 'a_position'),
      normal: gl.getAttribLocation(this.coreProgram, 'a_normal')
    };

    // 2. Glowing Synapse Particles Shader (Halo & Nodes)
    this.particleProgram = this.createProgram(this.getParticleVertexShader(), this.getParticleFragmentShader());
    this.particleUniforms = this.getUniformLocations(this.particleProgram, [
      'u_matrix_mvp', 'u_time', 'u_glow_intensity', 'u_audio_pulse',
      'u_color_primary', 'u_color_secondary', 'u_color_accent'
    ]);
    this.particleAttribs = {
      position: gl.getAttribLocation(this.particleProgram, 'a_position'),
      meta: gl.getAttribLocation(this.particleProgram, 'a_meta') // x: size, y: phase, z: colorMix
    };

    this.resizeCanvas();
  }

  // ==========================================================================
  // GLSL SHADERS
  // ==========================================================================
  getCoreVertexShader() {
    return `
      precision highp float;

      attribute vec3 a_position;
      attribute vec3 a_normal;

      uniform mat4 u_matrix_mvp;
      uniform float u_time;
      uniform float u_deform_scale;
      uniform float u_deform_freq;
      uniform float u_audio_pulse;
      uniform float u_speed;

      varying vec3 v_normal;
      varying vec3 v_position;
      varying float v_displacement;

      // Stefan Gustavson 3D Simplex Noise
      vec4 permute(vec4 x) { return mod(((x*34.0)+1.0)*x, 289.0); }
      vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

      float snoise(vec3 v) {
        const vec2 C = vec2(1.0/6.0, 1.0/3.0);
        const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
        vec3 i  = floor(v + dot(v, C.yyy));
        vec3 x0 = v - i + dot(i, C.xxx);
        vec3 g = step(x0.yzx, x0.xyz);
        vec3 l = 1.0 - g;
        vec3 i1 = min(g.xyz, l.zxy);
        vec3 i2 = max(g.xyz, l.zxy);
        vec3 x1 = x0 - i1 + 1.0 * C.xxx;
        vec3 x2 = x0 - i2 + 2.0 * C.xxx;
        vec3 x3 = x0 - 1.0 + 3.0 * C.xxx;
        i = mod(i, 289.0);
        vec4 p = permute(permute(permute(
                  i.z + vec4(0.0, i1.z, i2.z, 1.0))
                + i.y + vec4(0.0, i1.y, i2.y, 1.0))
                + i.x + vec4(0.0, i1.x, i2.x, 1.0));
        float n_ = 0.142857142857;
        vec3 ns = n_ * D.wyz - D.xzx;
        vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
        vec4 x_ = floor(j * ns.z);
        vec4 y_ = floor(j - 7.0 * x_);
        vec4 x = x_ *ns.x + ns.yyyy;
        vec4 y = y_ *ns.x + ns.yyyy;
        vec4 h = 1.0 - abs(x) - abs(y);
        vec4 b0 = vec4(x.xy, y.xy);
        vec4 b1 = vec4(x.zw, y.zw);
        vec4 s0 = floor(b0)*2.0 + 1.0;
        vec4 s1 = floor(b1)*2.0 + 1.0;
        vec4 sh = -step(h, vec4(0.0));
        vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
        vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
        vec3 p0 = vec3(a0.xy, h.x);
        vec3 p1 = vec3(a0.zw, h.y);
        vec3 p2 = vec3(a1.xy, h.z);
        vec3 p3 = vec3(a1.zw, h.w);
        vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
        p0 *= norm.x;
        p1 *= norm.y;
        p2 *= norm.z;
        p3 *= norm.w;
        vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
        m = m * m;
        return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
      }

      void main() {
        v_normal = a_normal;
        v_position = a_position;

        float t = u_time * u_speed;
        vec3 np = normalize(a_position);

        // Organic undulating bulges (matches reference image lobes)
        vec3 samplePos1 = np * u_deform_freq + vec3(0.0, t * 0.4, sin(t * 0.25) * 0.35);
        float n1 = snoise(samplePos1);

        // Secondary harmonic frequency for detailed wavy contour
        vec3 samplePos2 = np * (u_deform_freq * 2.1) + vec3(t * 0.25, 0.0, -t * 0.2);
        float n2 = snoise(samplePos2) * 0.45;

        // Acoustic speech reactivity: explosive dynamic waves when speaking
        float acousticRipple = sin(np.y * 6.0 + t * 4.5) * cos(np.x * 6.0 - t * 3.5) * u_audio_pulse * 0.45;
        
        float totalDisplacement = (n1 + n2) * u_deform_scale + acousticRipple;
        v_displacement = totalDisplacement;

        vec3 displaced = a_position + a_normal * totalDisplacement;
        gl_Position = u_matrix_mvp * vec4(displaced, 1.0);
      }
    `;
  }

  getCoreFragmentShader() {
    return `
      precision highp float;

      varying vec3 v_normal;
      varying vec3 v_position;
      varying float v_displacement;

      uniform float u_time;
      uniform float u_glow_intensity;
      uniform vec3 u_color_primary;    // Neon Emerald Green
      uniform vec3 u_color_secondary;  // Cyber Lime
      uniform vec3 u_color_accent;     // Mint Cyan
      uniform float u_color_shift;
      uniform float u_is_wire;         // 1.0 = Wireframe lines, 0.0 = Inner shell

      void main() {
        // Dual Green gradient (Emerald Green <-> Cyber Lime)
        float gradMix = clamp(v_position.y * 0.45 + 0.5 + sin(v_position.x * 2.0 + u_time * 0.6) * 0.25, 0.0, 1.0);
        gradMix = clamp(gradMix + u_color_shift * 0.4, 0.0, 1.0);
        vec3 neonColor = mix(u_color_primary, u_color_secondary, gradMix);

        // Edge Fresnel rim lighting
        vec3 viewDir = vec3(0.0, 0.0, 1.0);
        float fresnel = pow(1.0 - max(dot(normalize(v_normal), viewDir), 0.0), 2.2);

        if (u_is_wire > 0.5) {
          // WIREFRAME LINES: Brilliant laser-bright lines
          vec3 wireColor = neonColor * (1.6 + fresnel * 1.5) + u_color_accent * 0.4;
          float alpha = (0.85 + fresnel * 0.5) * u_glow_intensity;
          gl_FragColor = vec4(wireColor, clamp(alpha, 0.0, 1.0));
        } else {
          // INNER SHELL: Volumetric holographic depth
          vec3 shellColor = neonColor * (fresnel * 0.45);
          float alpha = (fresnel * 0.25) * u_glow_intensity;
          gl_FragColor = vec4(shellColor, clamp(alpha, 0.0, 1.0));
        }
      }
    `;
  }

  getParticleVertexShader() {
    return `
      precision highp float;

      attribute vec3 a_position;
      attribute vec3 a_meta; // x: size, y: phase, z: colorMix

      uniform mat4 u_matrix_mvp;
      uniform float u_time;
      uniform float u_audio_pulse;

      varying float v_color_mix;
      varying float v_alpha;

      void main() {
        v_color_mix = a_meta.z;

        // Gentle orbital drift & organic synapse pulsing around contour
        float phase = a_meta.y + u_time * 0.85;
        float pulse = sin(phase) * 0.35 + 0.65;
        v_alpha = pulse * (1.0 + u_audio_pulse * 1.5);

        // Dynamic vibration
        vec3 pos = a_position;
        pos.y += sin(phase * 1.5) * 0.05;
        pos.x += cos(phase * 1.2) * 0.05;
        pos.z += sin(phase * 0.9) * 0.05;

        gl_Position = u_matrix_mvp * vec4(pos, 1.0);
        
        // Large, highly visible glowing points with distance scaling
        float pSize = a_meta.x * (18.0 / gl_Position.w) * (1.0 + u_audio_pulse * 0.7);
        gl_PointSize = clamp(pSize, 4.0, 48.0);
      }
    `;
  }

  getParticleFragmentShader() {
    return `
      precision highp float;

      varying float v_color_mix;
      varying float v_alpha;

      uniform float u_glow_intensity;
      uniform vec3 u_color_primary;
      uniform vec3 u_color_secondary;
      uniform vec3 u_color_accent;

      void main() {
        // Antialiased glowing circular particle
        vec2 coord = gl_PointCoord - vec2(0.5);
        float dist = length(coord);
        if (dist > 0.5) discard;

        // Soft gaussian-like falloff
        float intensity = pow(1.0 - dist * 2.0, 2.0);

        vec3 color = mix(u_color_primary, u_color_secondary, v_color_mix);

        // Core white-hot synapse spark
        color += (u_color_accent + vec3(0.5)) * pow(intensity, 3.2);

        float alpha = intensity * v_alpha * u_glow_intensity;
        gl_FragColor = vec4(color, clamp(alpha, 0.0, 1.0));
      }
    `;
  }

  // ==========================================================================
  // GEOMETRY GENERATION: Quad Wireframe Grid + Inner Shell
  // ==========================================================================
  initGeometry() {
    const gl = this.gl;
    const latBands = this.options.latBands;
    const lonBands = this.options.lonBands;
    const radius = this.options.baseRadius;

    const vertices = [];
    const normals = [];
    const lineIndices = [];
    const triangleIndices = [];

    // 1. Generate Sphere Vertices
    for (let lat = 0; lat <= latBands; lat++) {
      const theta = (lat * Math.PI) / latBands;
      const sinTheta = Math.sin(theta);
      const cosTheta = Math.cos(theta);

      for (let lon = 0; lon <= lonBands; lon++) {
        const phi = (lon * 2 * Math.PI) / lonBands;
        const sinPhi = Math.sin(phi);
        const cosPhi = Math.cos(phi);

        const x = cosPhi * sinTheta;
        const y = cosTheta;
        const z = sinPhi * sinTheta;

        vertices.push(radius * x, radius * y, radius * z);
        normals.push(x, y, z);
      }
    }

    // 2. Generate Pure Quad Grid Lines (gl.LINES - Zero Diagonals!)
    for (let lat = 0; lat <= latBands; lat++) {
      for (let lon = 0; lon < lonBands; lon++) {
        const curr = lat * (lonBands + 1) + lon;
        const next = curr + 1;

        // Horizontal ring segment (parallels)
        lineIndices.push(curr, next);

        // Vertical rib segment (meridians)
        if (lat < latBands) {
          const below = curr + lonBands + 1;
          lineIndices.push(curr, below);
        }
      }
    }

    // 3. Generate Triangles for Inner Depth Shell
    for (let lat = 0; lat < latBands; lat++) {
      for (let lon = 0; lon < lonBands; lon++) {
        const first = lat * (lonBands + 1) + lon;
        const second = first + lonBands + 1;

        triangleIndices.push(first, second, first + 1);
        triangleIndices.push(second, second + 1, first + 1);
      }
    }

    this.vertexCount = vertices.length / 3;
    this.lineIndexCount = lineIndices.length;
    this.triangleIndexCount = triangleIndices.length;

    // Buffers
    this.posBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);

    this.normBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.normBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.STATIC_DRAW);

    this.lineIndexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.lineIndexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(lineIndices), gl.STATIC_DRAW);

    this.triIndexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.triIndexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(triangleIndices), gl.STATIC_DRAW);

    // Also create surface intersection node points (sparkles on every 2nd vertex)
    const nodePositions = [];
    const nodeMeta = [];
    for (let i = 0; i < vertices.length; i += 6) { // Every second vertex
      nodePositions.push(vertices[i], vertices[i + 1], vertices[i + 2]);
      nodeMeta.push(6.0, Math.random() * Math.PI * 2, Math.random());
    }

    this.nodeCount = nodePositions.length / 3;
    this.nodePosBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.nodePosBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(nodePositions), gl.STATIC_DRAW);

    this.nodeMetaBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.nodeMetaBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(nodeMeta), gl.STATIC_DRAW);
  }

  // ==========================================================================
  // PERIMETER HALO SYNAPSE PARTICLES (Matching Reference Image)
  // ==========================================================================
  initHaloParticles() {
    const gl = this.gl;
    const count = this.options.haloParticleCount;
    const positions = [];
    const meta = []; // x: size, y: phase, z: colorMix

    for (let i = 0; i < count; i++) {
      // Concentrate in a dense corona framing the perimeter contour of the sphere
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      
      // Radius: clustered around the perimeter (1.04 to 1.45)
      const r = this.options.baseRadius * (1.04 + Math.pow(Math.random(), 1.5) * 0.42);

      const x = r * Math.sin(phi) * Math.cos(theta);
      const y = r * Math.sin(phi) * Math.sin(theta);
      const z = r * Math.cos(phi);

      positions.push(x, y, z);

      // Distinct, prominent point sizes (4px to 14px)
      const size = 5.0 + Math.random() * 9.0;
      const phase = Math.random() * Math.PI * 2;
      const colorMix = Math.random();

      meta.push(size, phase, colorMix);
    }

    this.haloCount = count;

    this.haloPosBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.haloPosBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

    this.haloMetaBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.haloMetaBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(meta), gl.STATIC_DRAW);
  }

  // ==========================================================================
  // PALETTE & COLOR CONTROL
  // ==========================================================================
  setPalette(paletteName) {
    if (this.palettes[paletteName]) {
      this.palette = paletteName;
    }
  }

  // ==========================================================================
  // STATE MACHINE & TRANSITIONS
  // ==========================================================================
  setState(newState) {
    if (this.state === newState) return;
    this.state = newState;

    switch (newState) {
      case 'idle':
        this.targets.rotationSpeed = 0.005;
        this.targets.deformScale = 0.22;
        this.targets.deformFreq = 1.45;
        this.targets.glowIntensity = 1.15;
        this.targets.noiseSpeed = 0.5;
        this.targets.colorShift = 0.0;
        break;

      case 'listening':
        this.targets.rotationSpeed = 0.012;
        this.targets.deformScale = 0.28;
        this.targets.deformFreq = 1.9;
        this.targets.glowIntensity = 1.45;
        this.targets.noiseSpeed = 0.95;
        this.targets.colorShift = -0.4;
        break;

      case 'thinking':
        this.targets.rotationSpeed = 0.034;
        this.targets.deformScale = 0.34;
        this.targets.deformFreq = 2.4;
        this.targets.glowIntensity = 1.35;
        this.targets.noiseSpeed = 1.5;
        this.targets.colorShift = 0.6;
        break;

      case 'speaking':
        // Pronounced organic undulating hills & peaks matching reference image
        this.targets.rotationSpeed = 0.018;
        this.targets.deformScale = 0.42;   // High-amplitude undulating lobes
        this.targets.deformFreq = 1.6;    // Broad organic bulbous hills & valleys
        this.targets.glowIntensity = 1.65;  // Brilliant neon luminescence
        this.targets.noiseSpeed = 1.1;    // Liquid morphing pace
        this.targets.colorShift = 0.2;
        break;

      case 'error':
        this.targets.rotationSpeed = 0.007;
        this.targets.deformScale = 0.18;
        this.targets.deformFreq = 1.3;
        this.targets.glowIntensity = 1.1;
        this.targets.noiseSpeed = 0.4;
        this.targets.colorShift = 0.8;
        break;
    }
  }

  setAudioLevel(level) {
    this.liveAudioLevel = Math.max(0, Math.min(1.0, level));
  }

  // ==========================================================================
  // ANIMATION LOOP & RENDERING
  // ==========================================================================
  animate() {
    this.animationFrameId = requestAnimationFrame(() => this.animate());

    const now = performance.now();
    const dt = Math.min((now - this.lastFrameTime) / 1000, 0.1);
    this.lastFrameTime = now;
    const elapsed = (now - this.startTime) / 1000;

    // Smooth lerping of state parameters
    const lerpRate = 4.5 * dt;
    this.params.rotationSpeed += (this.targets.rotationSpeed - this.params.rotationSpeed) * lerpRate;
    this.params.deformScale += (this.targets.deformScale - this.params.deformScale) * lerpRate;
    this.params.deformFreq += (this.targets.deformFreq - this.params.deformFreq) * lerpRate;
    this.params.glowIntensity += (this.targets.glowIntensity - this.params.glowIntensity) * lerpRate;
    this.params.noiseSpeed += (this.targets.noiseSpeed - this.params.noiseSpeed) * lerpRate;
    this.params.colorShift += (this.targets.colorShift - this.params.colorShift) * lerpRate;

    // Audio reactivity
    const targetAudio = this.state === 'speaking' || this.state === 'listening' 
      ? Math.max(this.liveAudioLevel, Math.sin(elapsed * 7.5) * 0.2 + 0.2)
      : this.liveAudioLevel;
    this.params.audioPulse += (targetAudio - this.params.audioPulse) * (12.0 * dt);

    // Continuous rotation
    this.targetRotation.y += this.params.rotationSpeed;
    this.rotation.x += (this.targetRotation.x - this.rotation.x) * (6.0 * dt);
    this.rotation.y += (this.targetRotation.y - this.rotation.y) * (6.0 * dt);

    this.render(elapsed);
  }

  resizeCanvas() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.floor(this.canvas.clientWidth * dpr);
    const h = Math.floor(this.canvas.clientHeight * dpr);

    if (w > 0 && h > 0 && (this.canvas.width !== w || this.canvas.height !== h)) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  render(time) {
    const gl = this.gl;
    if (!gl) return;

    // Keep canvas dimensions strictly in sync with display size
    this.resizeCanvas();
    if (this.canvas.width === 0 || this.canvas.height === 0) return;

    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0.0, 0.0, 0.0, 0.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    const pal = this.palettes[this.palette] || this.palettes.green;

    // Camera Matrices
    const aspect = this.canvas.width / this.canvas.height;
    const fov = 45 * (Math.PI / 180);
    const projMatrix = this.createPerspectiveMatrix(fov, aspect, 0.1, 100.0);

    const viewMatrix = this.createIdentityMatrix();
    this.translateMatrix(viewMatrix, 0, 0, -5.2);

    const modelMatrix = this.createIdentityMatrix();
    this.rotateXMatrix(modelMatrix, this.rotation.x);
    this.rotateYMatrix(modelMatrix, this.rotation.y);

    const mvMatrix = this.multiplyMatrices(viewMatrix, modelMatrix);
    const mvpMatrix = this.multiplyMatrices(projMatrix, mvMatrix);

    // -------------------------------------------------------------
    // PASS 1: Volumetric Inner Holographic Shell (gl.TRIANGLES)
    // -------------------------------------------------------------
    gl.useProgram(this.coreProgram);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuffer);
    gl.enableVertexAttribArray(this.coreAttribs.position);
    gl.vertexAttribPointer(this.coreAttribs.position, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.normBuffer);
    gl.enableVertexAttribArray(this.coreAttribs.normal);
    gl.vertexAttribPointer(this.coreAttribs.normal, 3, gl.FLOAT, false, 0, 0);

    gl.uniformMatrix4fv(this.coreUniforms.u_matrix_mvp, false, mvpMatrix);
    gl.uniform1f(this.coreUniforms.u_time, time);
    gl.uniform1f(this.coreUniforms.u_deform_scale, this.params.deformScale);
    gl.uniform1f(this.coreUniforms.u_deform_freq, this.params.deformFreq);
    gl.uniform1f(this.coreUniforms.u_audio_pulse, this.params.audioPulse);
    gl.uniform1f(this.coreUniforms.u_speed, this.params.noiseSpeed);
    gl.uniform1f(this.coreUniforms.u_glow_intensity, this.params.glowIntensity);
    gl.uniform1f(this.coreUniforms.u_color_shift, this.params.colorShift);
    gl.uniform3f(this.coreUniforms.u_color_primary, pal.primary[0], pal.primary[1], pal.primary[2]);
    gl.uniform3f(this.coreUniforms.u_color_secondary, pal.secondary[0], pal.secondary[1], pal.secondary[2]);
    gl.uniform3f(this.coreUniforms.u_color_accent, pal.accent[0], pal.accent[1], pal.accent[2]);

    gl.uniform1f(this.coreUniforms.u_is_wire, 0.0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.triIndexBuffer);
    gl.drawElements(gl.TRIANGLES, this.triangleIndexCount, gl.UNSIGNED_SHORT, 0);

    // -------------------------------------------------------------
    // PASS 2: Pure Quad Grid Wireframe (gl.LINES - Zero Diagonals!)
    // -------------------------------------------------------------
    gl.uniform1f(this.coreUniforms.u_is_wire, 1.0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.lineIndexBuffer);
    gl.drawElements(gl.LINES, this.lineIndexCount, gl.UNSIGNED_SHORT, 0);

    // Clean up attribute bindings
    gl.disableVertexAttribArray(this.coreAttribs.normal);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);

    // -------------------------------------------------------------
    // PASS 3: Perimeter Halo Synapse Particles (gl.POINTS - Matches Image)
    // -------------------------------------------------------------
    gl.useProgram(this.particleProgram);

    // Counter-rotate slightly for rich 3D orbital depth
    const haloModelMatrix = this.createIdentityMatrix();
    this.rotateXMatrix(haloModelMatrix, this.rotation.x * 0.85);
    this.rotateYMatrix(haloModelMatrix, -this.rotation.y * 0.45);
    const haloMv = this.multiplyMatrices(viewMatrix, haloModelMatrix);
    const haloMvp = this.multiplyMatrices(projMatrix, haloMv);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.haloPosBuffer);
    gl.enableVertexAttribArray(this.particleAttribs.position);
    gl.vertexAttribPointer(this.particleAttribs.position, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.haloMetaBuffer);
    gl.enableVertexAttribArray(this.particleAttribs.meta);
    gl.vertexAttribPointer(this.particleAttribs.meta, 3, gl.FLOAT, false, 0, 0);

    gl.uniformMatrix4fv(this.particleUniforms.u_matrix_mvp, false, haloMvp);
    gl.uniform1f(this.particleUniforms.u_time, time);
    gl.uniform1f(this.particleUniforms.u_glow_intensity, this.params.glowIntensity * 1.3);
    gl.uniform1f(this.particleUniforms.u_audio_pulse, this.params.audioPulse);
    gl.uniform3f(this.particleUniforms.u_color_primary, pal.primary[0], pal.primary[1], pal.primary[2]);
    gl.uniform3f(this.particleUniforms.u_color_secondary, pal.secondary[0], pal.secondary[1], pal.secondary[2]);
    gl.uniform3f(this.particleUniforms.u_color_accent, pal.accent[0], pal.accent[1], pal.accent[2]);

    gl.drawArrays(gl.POINTS, 0, this.haloCount);

    // -------------------------------------------------------------
    // PASS 4: Surface Grid Intersection Nodes (gl.POINTS)
    // -------------------------------------------------------------
    gl.bindBuffer(gl.ARRAY_BUFFER, this.nodePosBuffer);
    gl.vertexAttribPointer(this.particleAttribs.position, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.nodeMetaBuffer);
    gl.vertexAttribPointer(this.particleAttribs.meta, 3, gl.FLOAT, false, 0, 0);

    gl.uniformMatrix4fv(this.particleUniforms.u_matrix_mvp, false, mvpMatrix);
    gl.drawArrays(gl.POINTS, 0, this.nodeCount);

    // Clean up
    gl.disableVertexAttribArray(this.particleAttribs.meta);
  }

  // ==========================================================================
  // EVENT LISTENERS
  // ==========================================================================
  initEvents() {
    window.addEventListener('resize', () => this.resizeCanvas());

    const onPointerDown = (e) => {
      this.isDragging = true;
      this.lastPointer.x = e.clientX || (e.touches && e.touches[0].clientX);
      this.lastPointer.y = e.clientY || (e.touches && e.touches[0].clientY);
    };

    const onPointerMove = (e) => {
      if (!this.isDragging) return;
      const clientX = e.clientX || (e.touches && e.touches[0].clientX);
      const clientY = e.clientY || (e.touches && e.touches[0].clientY);
      const dx = clientX - this.lastPointer.x;
      const dy = clientY - this.lastPointer.y;

      this.targetRotation.y += dx * 0.006;
      this.targetRotation.x = Math.max(-1.2, Math.min(1.2, this.targetRotation.x + dy * 0.006));

      this.lastPointer.x = clientX;
      this.lastPointer.y = clientY;
    };

    const onPointerUp = () => {
      this.isDragging = false;
    };

    this.canvas.addEventListener('mousedown', onPointerDown);
    window.addEventListener('mousemove', onPointerMove);
    window.addEventListener('mouseup', onPointerUp);

    this.canvas.addEventListener('touchstart', onPointerDown, { passive: true });
    window.addEventListener('touchmove', onPointerMove, { passive: true });
    window.addEventListener('touchend', onPointerUp, { passive: true });
  }

  // ==========================================================================
  // HELPER MATH & WEBGL UTILS
  // ==========================================================================
  createShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error('Shader compile error:', gl.getShaderInfoLog(shader));
      gl.deleteShader(shader);
      return null;
    }
    return shader;
  }

  createProgram(vertSource, fragSource) {
    const gl = this.gl;
    const vertShader = this.createShader(gl, gl.VERTEX_SHADER, vertSource);
    const fragShader = this.createShader(gl, gl.FRAGMENT_SHADER, fragSource);
    const program = gl.createProgram();

    gl.attachShader(program, vertShader);
    gl.attachShader(program, fragShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error('Program link error:', gl.getProgramInfoLog(program));
      return null;
    }
    return program;
  }

  getUniformLocations(program, uniformNames) {
    const gl = this.gl;
    const uniforms = {};
    for (const name of uniformNames) {
      uniforms[name] = gl.getUniformLocation(program, name);
    }
    return uniforms;
  }

  createIdentityMatrix() {
    return new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1
    ]);
  }

  createPerspectiveMatrix(fov, aspect, near, far) {
    const f = 1.0 / Math.tan(fov / 2);
    const nf = 1 / (near - far);
    return new Float32Array([
      f / aspect, 0, 0, 0,
      0, f, 0, 0,
      0, 0, (far + near) * nf, -1,
      0, 0, (2 * far * near) * nf, 0
    ]);
  }

  translateMatrix(out, x, y, z) {
    out[12] += out[0] * x + out[4] * y + out[8] * z;
    out[13] += out[1] * x + out[5] * y + out[9] * z;
    out[14] += out[2] * x + out[6] * y + out[10] * z;
    out[15] += out[3] * x + out[7] * y + out[11] * z;
  }

  rotateXMatrix(m, angle) {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    const m1 = m[1], m2 = m[2], m5 = m[5], m6 = m[6], m9 = m[9], m10 = m[10];
    m[1] = m1 * c - m2 * s;
    m[2] = m1 * s + m2 * c;
    m[5] = m5 * c - m6 * s;
    m[6] = m5 * s + m6 * c;
    m[9] = m9 * c - m10 * s;
    m[10] = m9 * s + m10 * c;
  }

  rotateYMatrix(m, angle) {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    const m0 = m[0], m2 = m[2], m4 = m[4], m6 = m[6], m8 = m[8], m10 = m[10];
    m[0] = m0 * c + m2 * s;
    m[2] = -m0 * s + m2 * c;
    m[4] = m4 * c + m6 * s;
    m[6] = -m4 * s + m6 * c;
    m[8] = m8 * c + m10 * s;
    m[10] = -m8 * s + m10 * c;
  }

  multiplyMatrices(a, b) {
    const out = new Float32Array(16);
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        out[i * 4 + j] =
          a[i * 4 + 0] * b[0 * 4 + j] +
          a[i * 4 + 1] * b[1 * 4 + j] +
          a[i * 4 + 2] * b[2 * 4 + j] +
          a[i * 4 + 3] * b[3 * 4 + j];
      }
    }
    return out;
  }

  destroy() {
    if (this.animationFrameId) cancelAnimationFrame(this.animationFrameId);
  }
}

// Export to window
window.NeuralCore = NeuralCore;
