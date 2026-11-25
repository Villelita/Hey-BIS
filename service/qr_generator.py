"""Generador ligero de códigos QR (ISO/IEC 18004) en Python puro sin dependencias externas.

Soporta modo Byte en Versiones 1 a 6 con corrección de errores Nivel M (hasta 106 bytes),
suficiente para cualquier URL local de Hey-BIS (p. ej. https://192.168.0.9:8443/kiosk).
"""

from __future__ import annotations

import html
from typing import List, Tuple

# Parámetros oficiales ISO/IEC 18004 para Nivel M (Versiones 1..6):
# version: (total_data_codewords, ec_codewords_per_block, num_blocks, alignment_centers)
_VERSION_SPECS: dict[int, Tuple[int, int, int, Tuple[int, ...]]] = {
    1: (16, 10, 1, ()),
    2: (28, 16, 1, (6, 18)),
    3: (44, 26, 1, (6, 22)),
    4: (64, 18, 2, (6, 26)),
    5: (86, 24, 2, (6, 30)),
    6: (108, 16, 4, (6, 34)),
}

# Secuencias de información de formato (15 bits) para Nivel M (indicador 00) y máscaras 0..7
# Calculadas con polinomio BCH(15, 5) y XOR con 0x5412
_FORMAT_INFO_M: Tuple[int, ...] = (
    0x5412,  # mask 0
    0x5125,  # mask 1
    0x5E7C,  # mask 2
    0x5B4B,  # mask 3
    0x45F9,  # mask 4
    0x40CE,  # mask 5
    0x4F97,  # mask 6
    0x4AA0,  # mask 7
)


def _gf_mul(a: int, b: int) -> int:
    """Multiplicación en GF(256) con polinomio primitivo 0x11D."""
    res = 0
    while b:
        if b & 1:
            res ^= a
        a <<= 1
        if a & 0x100:
            a ^= 0x11D
        b >>= 1
    return res


def _rs_generator_poly(degree: int) -> List[int]:
    """Construye el polinomio generador de Reed-Solomon de grado `degree`."""
    poly = [1]
    alpha = 1
    for _ in range(degree):
        next_poly = [0] * (len(poly) + 1)
        for i, coeff in enumerate(poly):
            next_poly[i] ^= coeff
            next_poly[i + 1] ^= _gf_mul(coeff, alpha)
        poly = next_poly
        alpha = _gf_mul(alpha, 2)
    return poly


def _rs_remainder(data: List[int], ec_count: int) -> List[int]:
    """Calcula los bytes de corrección de errores Reed-Solomon para un bloque."""
    gen = _rs_generator_poly(ec_count)
    rem = list(data) + [0] * ec_count
    for i in range(len(data)):
        factor = rem[i]
        if factor != 0:
            for j in range(1, len(gen)):
                rem[i + j] ^= _gf_mul(gen[j], factor)
    return rem[len(data):]


def _select_version(byte_len: int) -> int:
    for version, (total_data, _, _, _) in _VERSION_SPECS.items():
        # En modo Byte (V1-V9): 4 bits de modo + 8 bits de longitud = 12 bits -> 2 bytes de overhead
        if byte_len <= total_data - 2:
            return version
    raise ValueError(f"Texto demasiado largo para QR v1-v6 ({byte_len} bytes, máximo 106).")


def _encode_data_codewords(payload: bytes, total_data_codewords: int) -> List[int]:
    bits: List[int] = []

    def append_bits(val: int, length: int) -> None:
        for shift in range(length - 1, -1, -1):
            bits.append((val >> shift) & 1)

    # Modo Byte (0100) + longitud en 8 bits + bytes UTF-8
    append_bits(0b0100, 4)
    append_bits(len(payload), 8)
    for b in payload:
        append_bits(b, 8)

    capacity_bits = total_data_codewords * 8
    terminator_len = min(4, capacity_bits - len(bits))
    append_bits(0, terminator_len)

    while len(bits) % 8 != 0:
        bits.append(0)

    codewords = []
    for i in range(0, len(bits), 8):
        byte_val = 0
        for b in bits[i : i + 8]:
            byte_val = (byte_val << 1) | b
        codewords.append(byte_val)

    pad_bytes = (0xEC, 0x11)
    pad_idx = 0
    while len(codewords) < total_data_codewords:
        codewords.append(pad_bytes[pad_idx & 1])
        pad_idx += 1

    return codewords


def _interleave_blocks(data_codewords: List[int], ec_per_block: int, num_blocks: int) -> List[int]:
    block_size = len(data_codewords) // num_blocks
    data_blocks = [
        data_codewords[i * block_size : (i + 1) * block_size]
        for i in range(num_blocks)
    ]
    ec_blocks = [_rs_remainder(block, ec_per_block) for block in data_blocks]

    result: List[int] = []
    for col in range(block_size):
        for block in data_blocks:
            result.append(block[col])
    for col in range(ec_per_block):
        for block in ec_blocks:
            result.append(block[col])
    return result


def _mask_condition(mask_id: int, r: int, c: int) -> bool:
    if mask_id == 0:
        return (r + c) % 2 == 0
    if mask_id == 1:
        return r % 2 == 0
    if mask_id == 2:
        return c % 3 == 0
    if mask_id == 3:
        return (r + c) % 3 == 0
    if mask_id == 4:
        return ((r // 2) + (c // 3)) % 2 == 0
    if mask_id == 5:
        return ((r * c) % 2 + (r * c) % 3) == 0
    if mask_id == 6:
        return (((r * c) % 2) + ((r * c) % 3)) % 2 == 0
    return (((r + c) % 2) + ((r * c) % 3)) % 2 == 0


def _place_format_info(modules: List[List[bool]], mask_id: int) -> None:
    size = len(modules)
    fmt = _FORMAT_INFO_M[mask_id]

    def bit(i: int) -> bool:
        return bool((fmt >> i) & 1)

    # Copia 1 alrededor del patrón superior izquierdo
    coords_a = [
        (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 7), (8, 8),
        (7, 8), (5, 8), (4, 8), (3, 8), (2, 8), (1, 8), (0, 8),
    ]
    for idx, (r, c) in enumerate(coords_a):
        modules[r][c] = bit(14 - idx)

    # Copia 2 dividida entre inferior izquierdo y superior derecho
    for i in range(7):
        modules[size - 1 - i][8] = bit(14 - i)
    for i in range(8):
        modules[8][size - 8 + i] = bit(7 - i)


def _penalty_score(modules: List[List[bool]]) -> int:
    size = len(modules)
    score = 0

    # Regla 1: rachas horizontales y verticales >= 5
    for r in range(size):
        run_color = modules[r][0]
        run_len = 1
        for c in range(1, size):
            if modules[r][c] == run_color:
                run_len += 1
            else:
                if run_len >= 5:
                    score += 3 + (run_len - 5)
                run_color = modules[r][c]
                run_len = 1
        if run_len >= 5:
            score += 3 + (run_len - 5)

    for c in range(size):
        run_color = modules[0][c]
        run_len = 1
        for r in range(1, size):
            if modules[r][c] == run_color:
                run_len += 1
            else:
                if run_len >= 5:
                    score += 3 + (run_len - 5)
                run_color = modules[r][c]
                run_len = 1
        if run_len >= 5:
            score += 3 + (run_len - 5)

    # Regla 2: bloques 2x2 del mismo color
    for r in range(size - 1):
        for c in range(size - 1):
            v = modules[r][c]
            if v == modules[r][c + 1] == modules[r + 1][c] == modules[r + 1][c + 1]:
                score += 3

    # Regla 4: balance de módulos oscuros respecto a 50%
    dark = sum(1 for row in modules for cell in row if cell)
    total = size * size
    percent = (dark * 100) // total
    score += (abs(percent - 50) // 5) * 10
    return score


def encode_qr_matrix(text: str) -> List[List[bool]]:
    """Codifica un texto UTF-8 en una matriz booleana QR (True = oscuro, False = claro)."""
    payload = text.encode("utf-8")
    version = _select_version(len(payload))
    total_data, ec_per_block, num_blocks, align_centers = _VERSION_SPECS[version]
    size = version * 4 + 17

    data_cw = _encode_data_codewords(payload, total_data)
    all_cw = _interleave_blocks(data_cw, ec_per_block, num_blocks)

    bit_stream: List[bool] = []
    for cw in all_cw:
        for shift in range(7, -1, -1):
            bit_stream.append(bool((cw >> shift) & 1))

    # Inicializar matriz base y mapa de módulos funcionales (reservados)
    base: List[List[bool]] = [[False] * size for _ in range(size)]
    is_func: List[List[bool]] = [[False] * size for _ in range(size)]

    def set_func(r: int, c: int, val: bool) -> None:
        if 0 <= r < size and 0 <= c < size:
            base[r][c] = val
            is_func[r][c] = True

    # 1. Patrones de búsqueda (3 esquinas) + separadores
    for top, left in ((0, 0), (0, size - 7), (size - 7, 0)):
        for dr in range(-1, 8):
            for dc in range(-1, 8):
                r, c = top + dr, left + dc
                if 0 <= r < size and 0 <= c < size:
                    in_outer = 0 <= dr <= 6 and 0 <= dc <= 6
                    on_border = dr in (0, 6) or dc in (0, 6)
                    in_center = 2 <= dr <= 4 and 2 <= dc <= 4
                    set_func(r, c, in_outer and (on_border or in_center))

    # 2. Patrones de temporización (fila 6 y columna 6)
    for i in range(8, size - 8):
        set_func(6, i, i % 2 == 0)
        set_func(i, 6, i % 2 == 0)

    # 3. Patrones de alineación (Versiones 2..6)
    for r_center in align_centers:
        for c_center in align_centers:
            if is_func[r_center][c_center]:
                continue
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    val = max(abs(dr), abs(dc)) != 1
                    set_func(r_center + dr, c_center + dc, val)

    # 4. Módulo oscuro fijo y reserva de áreas de información de formato
    set_func(4 * version + 9, 8, True)
    for i in range(9):
        if not is_func[8][i]:
            set_func(8, i, False)
        if not is_func[i][8]:
            set_func(i, 8, False)
    for i in range(size - 8, size):
        if not is_func[8][i]:
            set_func(8, i, False)
        if not is_func[i][8]:
            set_func(i, 8, False)

    # 5. Colocar bits de datos en zig-zag (de derecha a izquierda en pares de columnas)
    bit_idx = 0
    col = size - 1
    upward = True
    while col > 0:
        if col == 6:
            col -= 1
        rows = range(size - 1, -1, -1) if upward else range(size)
        for r in rows:
            for dc in (0, -1):
                c = col + dc
                if not is_func[r][c]:
                    val = bit_stream[bit_idx] if bit_idx < len(bit_stream) else False
                    base[r][c] = val
                    bit_idx += 1
        upward = not upward
        col -= 2

    # 6. Elegir la mejor de las 8 máscaras estándar
    best_matrix: List[List[bool]] | None = None
    best_score = 10**9
    for mask_id in range(8):
        cand = [row[:] for row in base]
        for r in range(size):
            for c in range(size):
                if not is_func[r][c] and _mask_condition(mask_id, r, c):
                    cand[r][c] = not cand[r][c]
        _place_format_info(cand, mask_id)
        sc = _penalty_score(cand)
        if sc < best_score:
            best_score = sc
            best_matrix = cand

    assert best_matrix is not None
    return best_matrix


def generate_qr_svg(
    text: str,
    *,
    border: int = 4,
    dark: str = "#090d16",
    light: str = "#ffffff",
) -> str:
    """Genera un SVG vectorial compacto y nítido del código QR para `text`."""
    matrix = encode_qr_matrix(text)
    size = len(matrix)
    view_size = size + border * 2

    path_parts: List[str] = []
    for r, row in enumerate(matrix):
        y = r + border
        for c, cell in enumerate(row):
            if cell:
                x = c + border
                path_parts.append(f"M{x},{y}h1v1h-1z")

    path_d = "".join(path_parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {view_size} {view_size}" '
        f'shape-rendering="crispEdges" role="img" aria-label="Código QR de {html.escape(text)}">'
        f'<rect width="{view_size}" height="{view_size}" fill="{light}"/>'
        f'<path d="{path_d}" fill="{dark}"/>'
        f"</svg>"
    )


def build_qr_page_html(
    kiosk_url: str,
    panel_url: str | None = None,
) -> str:
    """Construye la página HTML /qr con el código QR listo para escanear."""
    qr_svg = generate_qr_svg(kiosk_url)
    esc_kiosk = html.escape(kiosk_url)
    panel_link_html = ""
    if panel_url:
        esc_panel = html.escape(panel_url)
        panel_link_html = (
            f'<a class="btn secondary" href="{esc_panel}">Abrir panel de control</a>'
        )

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Hey BIS — Código QR de Acceso</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #070b14;
      --card: rgba(15, 23, 42, 0.82);
      --border: rgba(56, 189, 248, 0.25);
      --accent: #38bdf8;
      --accent2: #22c55e;
      --text: #f8fafc;
      --muted: #94a3b8;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: radial-gradient(circle at top, #0f172a 0%, var(--bg) 65%);
      color: var(--text);
    }}
    .card {{
      width: 100%;
      max-width: 440px;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 24px;
      padding: 28px 24px;
      text-align: center;
      box-shadow: 0 24px 60px rgba(0, 0, 0, 0.5);
      backdrop-filter: blur(12px);
    }}
    .badge {{
      display: inline-block;
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent2);
      background: rgba(34, 197, 94, 0.12);
      border: 1px solid rgba(34, 197, 94, 0.3);
      padding: 5px 12px;
      border-radius: 999px;
      margin-bottom: 14px;
    }}
    h1 {{
      font-size: 24px;
      font-weight: 700;
      margin-bottom: 6px;
    }}
    .subtitle {{
      color: var(--muted);
      font-size: 14px;
      margin-bottom: 20px;
    }}
    .qr-box {{
      background: #ffffff;
      border-radius: 18px;
      padding: 14px;
      width: min(280px, 78vw);
      height: min(280px, 78vw);
      margin: 0 auto 18px;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }}
    .qr-box svg {{
      width: 100%;
      height: 100%;
      display: block;
    }}
    .url-pill {{
      display: block;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 13px;
      color: var(--accent);
      background: rgba(56, 189, 248, 0.09);
      border: 1px solid rgba(56, 189, 248, 0.25);
      border-radius: 12px;
      padding: 10px 12px;
      margin-bottom: 16px;
      word-break: break-all;
      text-decoration: none;
    }}
    .actions {{
      display: flex;
      gap: 10px;
      justify-content: center;
      flex-wrap: wrap;
      margin-bottom: 20px;
    }}
    .btn {{
      cursor: pointer;
      border: none;
      border-radius: 10px;
      padding: 10px 14px;
      font-size: 13px;
      font-weight: 600;
      text-decoration: none;
      color: #07101f;
      background: var(--accent);
      transition: transform 0.12s ease, opacity 0.12s ease;
    }}
    .btn:active {{ transform: scale(0.97); }}
    .btn.secondary {{
      color: var(--text);
      background: rgba(148, 163, 184, 0.16);
      border: 1px solid rgba(148, 163, 184, 0.28);
    }}
    .steps {{
      text-align: left;
      font-size: 13px;
      color: var(--muted);
      line-height: 1.55;
      border-top: 1px solid rgba(148, 163, 184, 0.16);
      padding-top: 16px;
    }}
    .steps ol {{
      padding-left: 18px;
    }}
    .steps li {{
      margin-bottom: 6px;
    }}
    .steps strong {{
      color: var(--text);
    }}
  </style>
</head>
<body>
  <main class="card">
    <span class="badge">● Listo para escanear</span>
    <h1>Prueba Hey BIS</h1>
    <p class="subtitle">Escanea este código QR desde tu celular para abrir el asistente de voz.</p>
    <div class="qr-box">
      {qr_svg}
    </div>
    <a class="url-pill" id="kioskUrl" href="{esc_kiosk}">{esc_kiosk}</a>
    <div class="actions">
      <button class="btn" id="copyBtn" type="button">Copiar enlace</button>
      <a class="btn secondary" href="/api/qr.svg" download="hey-bis-qr.svg">Descargar QR</a>
      {panel_link_html}
    </div>
    <div class="steps">
      <ol>
        <li>Conéctate a la <strong>misma red Wi-Fi</strong> que la Raspberry Pi.</li>
        <li>Escanea el QR con la cámara de tu teléfono.</li>
        <li>Si aparece aviso de certificado local, toca <strong>Configuración avanzada → Continuar</strong> y acepta el micrófono.</li>
      </ol>
    </div>
  </main>
  <script>
    const btn = document.getElementById('copyBtn');
    const url = document.getElementById('kioskUrl').textContent.trim();
    btn.addEventListener('click', async () => {{
      try {{
        await navigator.clipboard.writeText(url);
        btn.textContent = '¡Enlace copiado!';
        setTimeout(() => {{ btn.textContent = 'Copiar enlace'; }}, 1800);
      }} catch (e) {{
        prompt('Copia este enlace:', url);
      }}
    }});
  </script>
</body>
</html>
"""
