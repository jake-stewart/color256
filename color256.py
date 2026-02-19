#!/usr/bin/env python3

from argparse import ArgumentParser
import os
import sys
import re
import json
import math

def clamp(low, high, n):
    return max(low, min(high, n))

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return f"{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

def rgb_to_lab(rgb):
    r, g, b = (
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
            for c in (c / 255 for c in rgb)
    )
    xyz = (
        (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047,
        (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.0,
        (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883
    )
    fx, fy, fz = (
        t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
        for t in xyz
    )
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)

def lab_to_rgb(lab):
    l, a, b = lab
    fy = (l + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    x, y, z = (
        t**3 if t**3 > 0.008856 else (t - 16/116) / 7.787
        for t in (fx, fy, fz)
    )
    x, y, z = x * 0.95047, y * 1.0, z * 1.08883
    r = x * 3.2406 + y * -1.5372 + z * -0.4986
    g = x * -0.9689 + y * 1.8758 + z * 0.0415
    b_lin = x * 0.0557 + y * -0.2040 + z * 1.0570
    r, g, b_lin = (
        12.92 * c if c <= 0.0031308 else 1.055 * c**(1/2.4) - 0.055
        for c in (r, g, b_lin)
    )
    r = clamp(0, 255, int(r * 255 + 0.5))
    g = clamp(0, 255, int(g * 255 + 0.5))
    b_rgb = clamp(0, 255, int(b_lin * 255 + 0.5))
    return (r, g, b_rgb)

def _srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

def _linear_to_srgb(c):
    return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1/2.4)) - 0.055

def rgb_to_oklab(rgb):
    r_s, g_s, b_s = (x / 255.0 for x in rgb)
    r, g, b = (_srgb_to_linear(c) for c in (r_s, g_s, b_s))

    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

    l_ = l ** (1/3)
    m_ = m ** (1/3)
    s_ = s ** (1/3)

    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

    return (L, a, b)

def oklab_to_rgb(lab):
    L, a, b = lab

    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b

    l = l_ ** 3
    m = m_ ** 3
    s = s_ ** 3

    r_lin =  4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g_lin = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b_lin = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    r_s = _linear_to_srgb(r_lin)
    g_s = _linear_to_srgb(g_lin)
    b_s = _linear_to_srgb(b_lin)

    r8 = clamp(0, 255, int(r_s * 255 + 0.5))
    g8 = clamp(0, 255, int(g_s * 255 + 0.5))
    b8 = clamp(0, 255, int(b_s * 255 + 0.5))

    return (r8, g8, b8)


def _oklab_to_linear_srgb(L, a, b):
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l = l_ ** 3
    m = m_ ** 3
    s = s_ ** 3
    r =  4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return (r, g, b)

def _compute_max_saturation(a_, b_):
    if -1.88170328 * a_ - 0.80936493 * b_ > 1:
        k0 = +1.19086277; k1 = +1.76576728; k2 = +0.59662641; k3 = +0.75515197; k4 = +0.56771245
        wl = +4.0767416621; wm = -3.3077115913; ws = +0.2309699292
    elif 1.81444104 * a_ - 1.19445276 * b_ > 1:
        k0 = +0.73956515; k1 = -0.45954404; k2 = +0.08285427; k3 = +0.12541070; k4 = +0.14503204
        wl = -1.2684380046; wm = +2.6097574011; ws = -0.3413193965
    else:
        k0 = +1.35733652; k1 = -0.00915799; k2 = -1.15130210; k3 = -0.50559606; k4 = +0.00692167
        wl = -0.0041960863; wm = -0.7034186147; ws = +1.7076147010

    S = k0 + k1 * a_ + k2 * b_ + k3 * a_ * a_ + k4 * a_ * b_

    k_l = +0.3963377774 * a_ + 0.2158037573 * b_
    k_m = -0.1055613458 * a_ - 0.0638541728 * b_
    k_s = -0.0894841775 * a_ - 1.2914855480 * b_

    l_ = 1.0 + S * k_l
    m_ = 1.0 + S * k_m
    s_ = 1.0 + S * k_s

    l = l_ * l_ * l_
    m = m_ * m_ * m_
    s = s_ * s_ * s_

    l_dS = 3.0 * k_l * l_ * l_
    m_dS = 3.0 * k_m * m_ * m_
    s_dS = 3.0 * k_s * s_ * s_

    l_dS2 = 6.0 * k_l * k_l * l_
    m_dS2 = 6.0 * k_m * k_m * m_
    s_dS2 = 6.0 * k_s * k_s * s_

    f  = wl * l     + wm * m     + ws * s
    f1 = wl * l_dS  + wm * m_dS  + ws * s_dS
    f2 = wl * l_dS2 + wm * m_dS2 + ws * s_dS2

    S = S - f * f1 / (f1 * f1 - 0.5 * f * f2)

    return S

def _find_cusp(a_, b_):
    S_cusp = _compute_max_saturation(a_, b_)
    rgb = _oklab_to_linear_srgb(1, S_cusp * a_, S_cusp * b_)
    L_cusp = (1.0 / max(max(rgb[0], rgb[1]), max(rgb[2], 0.0))) ** (1/3)
    C_cusp = L_cusp * S_cusp
    return (L_cusp, C_cusp)

def _find_gamut_intersection(a_, b_, L1, C1, L0, cusp):
    cusp_L, cusp_C = cusp
    if ((L1 - L0) * cusp_C - (cusp_L - L0) * C1) <= 0:
        t = cusp_C * L0 / (C1 * cusp_L + cusp_C * (L0 - L1))
    else:
        t = cusp_C * (L0 - 1.0) / (C1 * (cusp_L - 1.0) + cusp_C * (L0 - L1))

        dL = L1 - L0
        dC = C1

        k_l = +0.3963377774 * a_ + 0.2158037573 * b_
        k_m = -0.1055613458 * a_ - 0.0638541728 * b_
        k_s = -0.0894841775 * a_ - 1.2914855480 * b_

        l_dt = dL + dC * k_l
        m_dt = dL + dC * k_m
        s_dt = dL + dC * k_s

        L = L0 * (1.0 - t) + t * L1
        C = t * C1

        l_ = L + C * k_l
        m_ = L + C * k_m
        s_ = L + C * k_s

        l = l_ * l_ * l_
        m = m_ * m_ * m_
        s = s_ * s_ * s_

        ldt = 3 * l_dt * l_ * l_
        mdt = 3 * m_dt * m_ * m_
        sdt = 3 * s_dt * s_ * s_

        ldt2 = 6 * l_dt * l_dt * l_
        mdt2 = 6 * m_dt * m_dt * m_
        sdt2 = 6 * s_dt * s_dt * s_

        r  =  4.0767416621 * l     - 3.3077115913 * m     + 0.2309699292 * s     - 1
        r1 =  4.0767416621 * ldt   - 3.3077115913 * mdt   + 0.2309699292 * sdt
        r2 =  4.0767416621 * ldt2  - 3.3077115913 * mdt2  + 0.2309699292 * sdt2

        u_r = r1 / (r1 * r1 - 0.5 * r * r2)
        t_r = -r * u_r

        g  = -1.2684380046 * l     + 2.6097574011 * m     - 0.3413193965 * s     - 1
        g1 = -1.2684380046 * ldt   + 2.6097574011 * mdt   - 0.3413193965 * sdt
        g2 = -1.2684380046 * ldt2  + 2.6097574011 * mdt2  - 0.3413193965 * sdt2

        u_g = g1 / (g1 * g1 - 0.5 * g * g2)
        t_g = -g * u_g

        b_  = -0.0041960863 * l     - 0.7034186147 * m     + 1.7076147010 * s     - 1
        b1  = -0.0041960863 * ldt   - 0.7034186147 * mdt   + 1.7076147010 * sdt
        b2  = -0.0041960863 * ldt2  - 0.7034186147 * mdt2  + 1.7076147010 * sdt2

        u_b = b1 / (b1 * b1 - 0.5 * b_ * b2)
        t_b = -b_ * u_b

        t_r = t_r if u_r >= 0.0 else 1e5
        t_g = t_g if u_g >= 0.0 else 1e5
        t_b = t_b if u_b >= 0.0 else 1e5

        t += min(t_r, min(t_g, t_b))

    return t

def _toe2(x):
    k1 = 0.250  # normally 0.206 but 0.250 is more consistent with CIELAB, which is known for its accurate lightness
    k2 = 0.03
    k3 = (1.0 + k1) / (1.0 + k2)
    return 0.5 * (k3 * x - k1 + math.sqrt((k3 * x - k1) ** 2 + 4 * k2 * k3 * x))

def _toe_inv2(x):
    k1 = 0.250  # normally 0.206 but 0.250 is more consistent with CIELAB, which is known for its accurate lightness
    k2 = 0.03
    k3 = (1.0 + k1) / (1.0 + k2)
    return (x * x + k1 * x) / (k3 * (x + k2))

def rgb_to_corrected_oklab(rgb):
    L, a, b = rgb_to_oklab(rgb)
    return (_toe2(L), a, b)

def _tc_untoe2(lab):
    Lr, a, b = lab
    L = _toe_inv2(max(0.0, min(1.0, Lr)))
    return (L, a, b)

def corrected_oklab_to_rgb(lab):
    return oklab_gc_to_rgb(_tc_untoe2(lab))

def oklab_gc_to_rgb(lab):
    L, a, b = lab

    C = math.sqrt(a * a + b * b)
    if C < 1e-10:
        L0 = max(0.0, min(1.0, L))
        r, g, b = _oklab_to_linear_srgb(L0, 0, 0)
        grey = clamp(0, 255, int(_linear_to_srgb(max(0, r)) * 255 + 0.5))
        return (grey, grey, grey)

    a_ = a / C
    b_ = b / C

    L0 = max(0.0, min(1.0, L))
    cusp = _find_cusp(a_, b_)
    t = _find_gamut_intersection(a_, b_, L, C, L0, cusp)
    t = min(t, 1.0)
    L_clipped = L0 + t * (L - L0)
    C_clipped = t * C

    r, g, b = _oklab_to_linear_srgb(L_clipped, C_clipped * a_, C_clipped * b_)
    r8 = clamp(0, 255, int(_linear_to_srgb(max(0, r)) * 255 + 0.5))
    g8 = clamp(0, 255, int(_linear_to_srgb(max(0, g)) * 255 + 0.5))
    b8 = clamp(0, 255, int(_linear_to_srgb(max(0, b)) * 255 + 0.5))
    return (r8, g8, b8)

COLORSPACES = {
    "lab":   (rgb_to_lab, lab_to_rgb),
    "oklab": (rgb_to_corrected_oklab, corrected_oklab_to_rgb),
}

DEFAULT_COLORSPACE = "lab"
to_colorspace, from_colorspace = COLORSPACES[DEFAULT_COLORSPACE]

def adjust_lightness(rgb, l_delta):
    l, a, b = to_colorspace(rgb)
    new_l = clamp(0, 100, l + l_delta)
    return from_colorspace((new_l, a, b))

class Style:
    def __init__(
        self,
        bold=False,
        italic=False,
        underline=False,
        dim=False,
        blink=False,
        reverse=False,
        hidden=False,
        strikethrough=False,
        fg=None,
        bg=None,
    ):
        self.bold = bold
        self.italic = italic
        self.underline = underline
        self.dim = dim
        self.blink = blink
        self.reverse = reverse
        self.hidden = hidden
        self.strikethrough = strikethrough
        self.fg = fg
        self.bg = bg

    def clone(self):
        return Style(
            bold = self.bold,
            italic = self.italic,
            underline = self.underline,
            dim = self.dim,
            blink = self.blink,
            reverse = self.reverse,
            hidden = self.hidden,
            strikethrough = self.strikethrough,
            fg = self.fg,
            bg = self.bg,
        )
    
    def apply(self, text):
        codes = []
        
        if self.bold: codes.append('1')
        if self.dim: codes.append('2')
        if self.italic: codes.append('3')
        if self.underline: codes.append('4')
        if self.blink: codes.append('5')
        if self.reverse: codes.append('7')
        if self.hidden: codes.append('8')
        if self.strikethrough: codes.append('9')

        for color, is_fg in ((self.fg, True), (self.bg, False)):
            if color is not None:
                offset = 0 if is_fg else 10
                if isinstance(color, str):
                    r, g, b = hex_to_rgb(color)
                    codes.append(f'{38 + offset};2;{r};{g};{b}')
                elif isinstance(color, tuple):
                    r, g, b = color
                    codes.append(f'{38 + offset};2;{r};{g};{b}')
                elif color < 8:
                    codes.append(str(30 + offset + color))
                elif color < 16:
                    codes.append(str(90 + offset + (color - 8)))
                else:
                    codes.append(f'{38 + offset};5;{color}')

        if not codes:
            return text

        return f"\033[{';'.join(codes)}m{text}\033[0m"

class Block:
    @staticmethod
    def vertical(*args, gap=0):
        return Block(*args, axis=1, gap=gap)

    @staticmethod
    def horizontal(*args, gap=0):
        return Block(*args, axis=0, gap=gap)

    def __init__(self, *args, width=None, axis=1, gap=0):
        self.lines = []
        blocks = Block._normalize_args(*args)
        if axis == 0:
            max_lines = max((len(block.lines) for block in blocks), default=0)
            col_widths = []
            for block in blocks:
                max_width = max((width for _, width in block.lines), default=0)
                col_widths.append(max_width)
            for line_idx in range(max_lines):
                line_parts = []
                for col_idx, block in enumerate(blocks):
                    if line_idx < len(block.lines):
                        content, width = block.lines[line_idx]
                        padding_needed = col_widths[col_idx] - width
                        line_parts.append(content + " " * padding_needed)
                    else:
                        line_parts.append(" " * col_widths[col_idx])
                separator = " " * gap
                combined_line = separator.join(line_parts)
                total_width = sum(col_widths) + gap * (len(blocks) - 1)
                self.lines.append((combined_line, total_width))
        else:
            for i, block in enumerate(blocks):
                self.lines.extend(block.lines)
                if i < len(blocks) - 1:
                    for _ in range(gap):
                        self.lines.append(("", 0))
        if width is not None:
            for i, line in enumerate(self.lines):
                self.lines[i] = (line[0], width)
    
    def append(self, content, width=None):
        if width is None:
            width = len(content)
        self.lines.append((content, width))
        return self
    
    def extend(self, block):
        self.lines.extend(block.lines)
        return self
    
    @staticmethod
    def _normalize_args(*args):
        blocks = []
        for arg in args:
            if isinstance(arg, Block):
                blocks.append(arg)
            elif hasattr(arg, "__iter__") and not isinstance(arg, str):
                items = list(arg)
                if not items:
                    continue
                converted = []
                for item in items:
                    if isinstance(item, Block):
                        converted.append(item)
                    elif isinstance(item, str):
                        converted.append(Block(item))
                blocks.extend(converted)
            else:
                child = Block()
                s = str(arg)
                child.lines.append((s, len(s)))
                blocks.append(child)
        return blocks
    
    def print(self):
        for content, _ in self.lines:
            print(content)

def generate_base16_extras(theme):
    bg_lab = to_colorspace(theme.bg)
    fg_lab = to_colorspace(theme.fg)
    light = bg_lab[0] > fg_lab[0]

    for i in range(8):
        if theme[i + 8] == theme[i]:
            l, a, b = to_colorspace(theme[i])
            l = clamp(0, 100, l * 1.1)
            theme[i + 8] = from_colorspace((l, a, b))

    if theme[0] == theme.bg:
        l = clamp(0, 100, bg_lab[0] - (5 if light else 3))
        theme[0] = from_colorspace((l, bg_lab[1], bg_lab[2]))

    l = clamp(0, 100, bg_lab[0] + (-20.0 if light else 20))
    theme[8] = from_colorspace((l, bg_lab[1], bg_lab[2]))

def lerp_lab(t, lab1, lab2):
    return tuple(a + t * (b - a) for a, b in zip(lab1, lab2))


def lab2lch(lab):
    l, a, b = lab
    c = (a ** 2 + b ** 2) ** 0.5
    h = 0
    if abs(a) > 0.0001 or abs(b) > 0.0001:
        h = math.atan2(b, a)
        h = (h + 4 * math.pi) % (2 * math.pi)
    return l, c, h


def lch2lab(lch):
    l, c, h = lch
    a = math.cos(h) * c
    b = math.sin(h) * c
    return l, a, b


def generate_primary(black, white, dark, light):
    l, c, h = lab2lch(lerp_lab(0.5, dark, light))
    l_black, c_black, _ = lab2lch(black)
    l_white, c_white, _ = lab2lch(white)
    l_dark, c_dark, _ = lab2lch(dark)
    l_light, c_light, _ = lab2lch(light)

    try:
        l = (
            (
                (c_white - c_black) * (l_dark - l_black) * (l_light - l_white)
                + (c_dark - c_black) * (l_light - l_white) * l_black
                - (c_light - c_white) * (l_dark - l_black) * l_white
            ) / (
                (c_dark - c_black) * (l_light - l_white)
                - (c_light - c_white) * (l_dark - l_black)
            )
        )
        l = min(max(l, l_dark), l_light)

        c = c_black + (c_dark - c_black) * (l - l_black) / (l_dark - l_black)
        c = max(c, c_dark, c_light)
    except ZeroDivisionError:
        pass

    return lch2lab((l, c, h))


def generate_256_palette(base16, bg=None, fg=None):
    base16_lab = [to_colorspace(c) for c in base16]
    bg_lab = to_colorspace(bg) if bg else base16_lab[0]
    fg_lab = to_colorspace(fg) if fg else base16_lab[7]
    if bg_lab[0] < fg_lab[0]:
        black, white = bg_lab, fg_lab
    else:
        black, white = fg_lab, bg_lab
    base8_lab = [
        generate_primary(black, white, base16_lab[i], base16_lab[i + 8])
        for i in range(8)
    ]

    palette = [*base16]

    for r in range(6):
        c0 = lerp_lab(r / 5, bg_lab, base8_lab[1])
        c1 = lerp_lab(r / 5, base8_lab[2], base8_lab[3])
        c2 = lerp_lab(r / 5, base8_lab[4], base8_lab[5])
        c3 = lerp_lab(r / 5, base8_lab[6], fg_lab)
        for g in range(6):
            c4 = lerp_lab(g / 5, c0, c1)
            c5 = lerp_lab(g / 5, c2, c3)
            for b in range(6):
                c6 = lerp_lab(b / 5, c4, c5)
                palette.append(from_colorspace(c6))

    for i in range(24):
        t = (i + 1) / 25
        lab = lerp_lab(t, bg_lab, fg_lab)
        palette.append(from_colorspace(lab))

    return palette


class Theme:
    def __init__(self, name, palette, bg=None, fg=None):
        self.name = name
        self.palette = palette
        self.bg = bg or palette[0]
        self.fg = fg or palette[7]

    def __setitem__(self, index, value):
        self.palette.__setitem__(index, value)

    def __getitem__(self, index):
        return self.palette.__getitem__(index)

    def greyscale(self, index):
        return self[232 + index]

    def rgb(self, r, g, b):
        return self[16 + r * 36 + g * 6 + b]

    @property
    def selection(self):
        return self.greyscale(5)

def parse_theme(fname):
    hex_key_re = re.compile(r"([a-z0-9]+[^a-z0-9].*)([a-f0-9]{6})")
    int_key_re = re.compile(r"([a-z0-9]+[^a-z0-9].*?)([0-9]+)")
    int_re = re.compile(r"([0-9]+)")
    hex_re = re.compile(r"#?([a-f0-9]{6})")
    palette_group = None
    color_names = [
        "black",
        "red",
        "green",
        "yellow",
        "blue",
        "magenta",
        "cyan",
        "white",
    ]

    palette = [hex_to_rgb(c) for c in BASELINE_BASE_16[:8]] + [None for _ in range(8)]
    name = None
    fg=None
    bg=None
    idx = 0

    with open(fname) as f:
        content = f.read()
        try:
            content = json.dumps(json.loads(content), indent=4)
        except:
            pass

    for line in content.splitlines():
        line = line.strip().lower()
        match = hex_key_re.search(line)
        if match:
            key = match.group(1).lower()
            color = hex_to_rgb(match.group(2))
            if "cursor" in key or "selection" in key:
                pass
            elif "foreground" in key or "fg" in key:
                fg = color
            elif "background" in key or "bg" in key:
                bg = color
            else:
                for i, color_name in enumerate(color_names):
                    if color_name in key:
                        idx = i
                        if "bright" in key or palette_group == "bright":
                            idx += 8
                        if palette_group != "dim":
                            palette[idx] = color
                        break
                else:
                    match = int_re.search(key)
                    if match:
                        idx = int(match.group(1))
                        if "bright" in key or palette_group == "bright":
                            idx += 8
                        if palette_group != "dim":
                            if 0 <= idx < 16:
                                palette[idx] = color
            continue

        match = int_key_re.search(line)
        if match:
            key = match.group(1).lower()
            idx = int(match.group(2))
            if idx < len(palette):
                if "cursor" in key or "selection" in key:
                    pass
                elif "foreground" in key or "fg" in key:
                    fg = palette[idx]
                elif "background" in key or "bg" in key:
                    bg = palette[idx]
            continue

        match = hex_re.search(line)
        if match:
            if idx < len(palette):
                palette[idx] = hex_to_rgb(match.group(1))
            elif idx == len(palette):
                palette.append(hex_to_rgb(match.group(1)))
            idx += 1
            continue

        if "bright" in line:
            palette_group = "bright"
        elif "normal" in line or "default" in line or "standard" in line:
            palette_group = None
        elif "dim" in line or "faint" in line:
            palette_group = "dim"
    
    for i in range(8):
        color = palette[i]
        assert color
        bright = palette[i + 8]
        if bright is None:
            palette[i + 8] = palette[i]
    
    instance = Theme(
        name or os.path.split(os.path.splitext(fname)[0])[-1],
        palette,
        bg=bg,
        fg=fg
    )
    return instance

def apply_color(type_index, palette_index, color):
    codes = [str(type_index)]
    if palette_index is not None:
        codes.append(str(palette_index))
    codes.append("rgb:" + "/".join(f"{c:02x}" for c in color))
    print(f"\033]{';'.join(codes)}\033\\", end="")

def apply_theme(theme):
    for i, color in enumerate(theme.palette):
        apply_color(4, i, color)
    apply_color(10, None, theme.fg)
    apply_color(11, None, theme.bg)
    apply_color(12, None, theme.fg)

def generate_base8_theme(theme):
    buffer = [];
    buffer.append("#%s" % rgb_to_hex(theme.bg))
    for i in range(1, 7):
        buffer.append("#%s" % rgb_to_hex(theme[i]))
    buffer.append("#%s" % rgb_to_hex(theme.fg))
    return "\n".join(buffer)

def generate_kitty_theme(theme):
    buffer = [];
    buffer.append("background #%s" % rgb_to_hex(theme.bg))
    buffer.append("foreground #%s" % rgb_to_hex(theme.fg))
    buffer.append("cursor #%s" % rgb_to_hex(theme.fg))
    buffer.append("selection_background #%s" % rgb_to_hex(theme.selection))
    buffer.append("selection_foreground none")
    for i in range(0, 256):
        buffer.append("color%d #%s" % (i, rgb_to_hex(theme[i])))
    return "\n".join(buffer)

def generate_ghostty_theme(theme):
    buffer = [];
    buffer.append("background = #%s" % rgb_to_hex(theme.bg))
    buffer.append("foreground = #%s" % rgb_to_hex(theme.fg))
    buffer.append("cursor = #%s" % rgb_to_hex(theme.fg))
    buffer.append("selection-background = #%s" % rgb_to_hex(theme.selection))
    buffer.append("selection-foreground = cell-foreground")
    for i in range(0, 256):
        buffer.append("palette = %d = #%s" % (i, rgb_to_hex(theme[i])))
    return "\n".join(buffer)

def generate_wezterm_theme(theme):
    buffer = [];
    buffer.append("colors = {")
    buffer.append('    background = "#%s",' % rgb_to_hex(theme.bg))
    buffer.append('    foreground = "#%s",' % rgb_to_hex(theme.fg))
    buffer.append('    cursor_bg = "#%s",' % rgb_to_hex(theme.fg))
    buffer.append('    cursor_border = "#%s",' % rgb_to_hex(theme.fg))
    buffer.append('    ansi = {')
    for i in range(0, 8):
        buffer.append('        "#%s",' % rgb_to_hex(theme[i]))
    buffer.append('    },')
    buffer.append('    brights = {')
    for i in range(8, 16):
        buffer.append('        "#%s",' % rgb_to_hex(theme[i]))
    buffer.append('    },')
    buffer.append('    indexed = {')
    for i in range(16, 256):
        buffer.append('        [%d] = "#%s",' % (i, rgb_to_hex(theme[i])))
    buffer.append('    }')
    buffer.append('}')
    return "\n".join(buffer)

def generate_alacritty_theme(theme):
    buffer = []
    buffer.append("[colors]")
    buffer.append("cursor = { text = 'CellForeground', cursor = '#%s' }" % rgb_to_hex(theme.fg))
    buffer.append("selection = { text = 'CellForeground', background = '#%s' }" % rgb_to_hex(theme.selection))
    buffer.append("indexed_colors = [")
    for i in range(16, 256):
        buffer.append("    { index = %d, color = '#%s' }," % (i, rgb_to_hex(theme[i])))
    buffer.append("]")
    buffer.append("[colors.primary]")
    buffer.append("background = '#%s'" % rgb_to_hex(theme.bg))
    buffer.append("foreground = '#%s'" % rgb_to_hex(theme.fg))
    buffer.append("[colors.normal]")
    color_names = ["black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"]
    for i in range(0, 8):
        buffer.append("%s = '#%s'" % (color_names[i], rgb_to_hex(theme[i])))
    buffer.append("[colors.bright]")
    for i in range(8, 16):
        buffer.append("%s = '#%s'" % (color_names[i-8], rgb_to_hex(theme[i])))
    return "\n".join(buffer)

def generate_foot_theme(theme):
    buffer = []
    buffer.append("[colors]")
    buffer.append("background=%s" % rgb_to_hex(theme.bg))
    buffer.append("foreground=%s" % rgb_to_hex(theme.fg))
    buffer.append("cursor=%s %s" % (rgb_to_hex(theme.bg), rgb_to_hex(theme.fg)))
    buffer.append("selection-background=%s" % rgb_to_hex(theme.selection))
    for i in range(0, 8):
        buffer.append("regular%d=%s" % (i, rgb_to_hex(theme[i])))
    for i in range(8, 16):
        buffer.append("bright%d=%s" % (i-8, rgb_to_hex(theme[i])))
    for i in range(16, 256):
        buffer.append("%d=%s" % (i, rgb_to_hex(theme[i])))
    return "\n".join(buffer)

# https://github.com/Roliga/urxvt-xresources-256
def generate_xresources_theme(theme):
    buffer = []
    buffer.append("*.foreground: #%s" % rgb_to_hex(theme.fg))
    buffer.append("*.background: #%s" % rgb_to_hex(theme.bg))
    buffer.append("*.cursorColor: #%s" % rgb_to_hex(theme.fg))
    for i in range(256):
        buffer.append("*.color%d: #%s" % (i, rgb_to_hex(theme[i])))
    return "\n".join(buffer)

def generate_st_theme(theme):
    buffer = []
    buffer.append("static const char *colorname[] = {")
    for i in range(256):
        buffer.append('\t"#%s",' % rgb_to_hex(theme[i]))
    buffer.append('\t"#%s",' % rgb_to_hex(theme.bg))
    buffer.append('\t"#%s",' % rgb_to_hex(theme.fg))
    buffer.append("};")
    buffer.append("unsigned int defaultbg = 256;");
    buffer.append("unsigned int defaultfg = 257;");
    buffer.append("static unsigned int defaultcs = 257;");
    return "\n".join(buffer)

GENERATE_LOOKUP = {
    'base8': generate_base8_theme,
    'kitty': generate_kitty_theme,
    'ghostty': generate_ghostty_theme,
    'wezterm': generate_wezterm_theme,
    'alacritty': generate_alacritty_theme,
    'foot': generate_foot_theme,
    'xresources': generate_xresources_theme,
    'st': generate_st_theme,
}

def preview_theme(name, palette, fg=None, bg=None):
    def color_str(index, text, background=True):
        rgb_color = bg if index is None else palette[index]
        index = index or 0
        if 16 <= index <= 231:
            idx = index - 16
            r_idx = (idx // 36) % 6
            g_idx = (idx // 6) % 6
            b_idx = idx % 6
            dark = r_idx < 4 and g_idx < 4 and b_idx < 4
        
        elif 232 <= index <= 255:
            grey_level = index - 232
            dark = grey_level < 11
        else:
            dark = index % 8 == 0
        if background:
            if dark:
                return Block(Style(fg=fg, bg=rgb_color).apply(text), width=len(text))
            else:
                return Block(Style(fg=rgb_color, bg=bg, reverse=True).apply(text), width=len(text))
        else:
            return Block(Style(fg=rgb_color, bg=bg).apply(text), width=len(text))

    def grey_block(from_idx, to_idx, vertical=True):
        def ansi_greyscale_index(grey_idx):
            if grey_idx == -1:
                return None
            elif grey_idx == 24:
                return 231
            else:
                return 232 + grey_idx

        step = 1 if from_idx < to_idx else -1
        r = range(from_idx, to_idx + step, step)
        
        blocks = []
        for i in r:
            ansi_color = ansi_greyscale_index(i)
            
            brightness = i / 24
            brightness_char = "%x" % min(15, int(brightness * 16))
            
            blocks.append(color_str(ansi_color, f" {brightness_char} "))
        
        return Block.vertical(*blocks) if vertical else Block.horizontal(*blocks)

    def color_slices_block(depth=3, vertical=True, final=False, black=False, reverse=False, background=True):
        def color_slice_block(*colors):
            blocks = []
            for i in range(0 if black else 1, 6):
                indexes = [0, 0, 0]
                for r_enable, g_enable, b_enable in colors:
                    indexes[0] += i if r_enable else 0
                    indexes[1] += i if g_enable else 0
                    indexes[2] += i if b_enable else 0
                for i in range(len(indexes)):
                    indexes[i] //= len(colors)
                r, g, b = indexes;
                index = 16 + r * 36 + g * 6 + b
                brightness = (r + g + b) / 15
                brightness_char = "%x" % int(brightness * 16)
                blocks.append(color_str(index, " " + brightness_char + " ", background=background))
            for i in range(1, 5 + final):
                indexes = [0, 0, 0]
                for r_enable, g_enable, b_enable in colors:
                    indexes[0] += 5 if r_enable else i
                    indexes[1] += 5 if g_enable else i
                    indexes[2] += 5 if b_enable else i
                for i in range(len(indexes)):
                    indexes[i] //= len(indexes)
                r, g, b = indexes;
                index = 16 + r * 36 + g * 6 + b
                brightness = (r + g + b) / 16
                brightness_char = "%x" % int(brightness * 16)
                blocks.append(color_str(index, " " + brightness_char + " ", background=background))
            if reverse:
                blocks.reverse()
            return Block.vertical(blocks)

        colors = [
            (True, False, False),
            (True, True, False),
            (False, True, False),
            (False, True, True),
            (False, False, True),
            (True, False, True),
        ]
        
        slices = []
        for i in range(len(colors)):
            current = colors[i]
            next_color = colors[(i + 1) % len(colors)]
            for step in range(depth):
                args = [current] * (depth - step) + [next_color] * step
                slices.append(color_slice_block(*args))
        if vertical:
            return Block.vertical(slices)
        else:
            return Block.horizontal(slices)

    Block.horizontal(
        Block.vertical(
            Block.horizontal(
                Block.vertical(grey_block(24, 6, vertical=True)),
                Block.vertical(
                    color_slices_block(reverse=True, vertical=False, final=True, depth=3),
                    color_slices_block(reverse=False, vertical=False, depth=3, black=False, background=False),
                    gap=0
                ),
                Block.vertical(grey_block(24, 6, vertical=True)),
                gap=0
            ),
            Block.horizontal(
                grey_block(5, 0, vertical=False),
                color_str(None, name[:24].center(24)),
                grey_block(0, 5, vertical=False),
                gap=0
            ),
            gap=0
        ),
        Block.vertical(
            Block(""),
            (Block.horizontal(
                 color_str(i, " %x " % i, background=True),
                 color_str(i + 8, " %x " % (i + 8), background=True)
            ) for i in range(8)),
            Block(""),
            Block(""),
            (Block.horizontal(
                 color_str(i, " %x " % i, background=False),
                 color_str(i + 8, " %x " % (i + 8), background=False)
            ) for i in range(8)),
            Block(""),
        ), gap=3
    ).print()


BASELINE_BASE_16 = [
    "000000", "cc0403", "19cb00", "cecb00",
    "0d73cc", "cb1ed1", "0dcdcd", "dddddd",
    "767676", "f2201f", "23fd00", "fffd00",
    "1a8fff", "fd28ff", "14ffff", "ffffff",
]

BASELINE_RGB = [
    (0 if r == 0 else 55 + r * 40,
     0 if g == 0 else 55 + g * 40,
     0 if b == 0 else 55 + b * 40)
    for r in range(6)
    for g in range(6)
    for b in range(6)
]

BASELINE_GREYSCALE = [
    (8 + i * 10, 8 + i * 10, 8 + i * 10)
    for i in range(24)
]

BASELINE_THEME = Theme("Default",
    [hex_to_rgb(c) for c in BASELINE_BASE_16] + BASELINE_RGB + BASELINE_GREYSCALE)

def main():
    parser = ArgumentParser()
    parser.add_argument("filenames", nargs="*")
    parser.add_argument("--generate", choices=GENERATE_LOOKUP)
    parser.add_argument("--output", type=str)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--adjust-lightness", type=int)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--colorspace", choices=COLORSPACES, default=DEFAULT_COLORSPACE)
    parser.add_argument("--test", action="store_true")
    ns = parser.parse_args()

    global to_colorspace, from_colorspace
    to_colorspace, from_colorspace = COLORSPACES[ns.colorspace]

    themes = list(map(parse_theme, ns.filenames))

    if ns.baseline:
        themes.append(BASELINE_THEME)

    if ns.adjust_lightness is not None:
        for theme in themes:
            theme.fg = adjust_lightness(theme.fg, ns.adjust_lightness)
            theme.bg = adjust_lightness(theme.bg, ns.adjust_lightness)
            for i in range(min(16, len(theme.palette))):
                theme[i] = adjust_lightness(theme[i], ns.adjust_lightness)

    for theme in themes:
        if theme != BASELINE_THEME:
            generate_base16_extras(theme)
            base8 = theme[:8]
            base8[0] = theme.bg
            base8[7] = theme.fg
            theme.palette = generate_256_palette(theme[:16], bg=theme.bg, fg=theme.fg)
    
    if ns.generate:
        if ns.output is not None:
            parent = ns.output or "."
            os.makedirs(parent, exist_ok=True)
            for theme in themes:
                fname = os.path.join(ns.output or ".", theme.name + "." + ns.generate + ".txt")
                with open(fname, "w+") as f:
                    f.write(GENERATE_LOOKUP[ns.generate](theme))
                    print("generated", fname)
        else:
            if len(themes) == 0:
                print("No theme selected", file=sys.stderr)
                exit(1)
            if len(themes) > 1:
                print("Can only apply a generate theme unless --output is specified", file=sys.stderr)
                exit(1)
            print(GENERATE_LOOKUP[ns.generate](themes[0]))
    elif ns.apply:
        if len(themes) == 0:
            print("No theme selected", file=sys.stderr)
            exit(1)
        else:
            if len(themes) > 1:
                print("Can only apply a single theme", file=sys.stderr)
                exit(1)
            apply_theme(themes[0])
    else:
        if themes:
            for i, theme in enumerate(themes):
                preview_theme(
                    theme.name,
                    theme.palette,
                    fg=theme.fg,
                    bg=theme.bg,
                )
                if i != len(themes) - 1:
                    print()
        else:
            preview_theme("Active Theme", list(range(256)))

if __name__ == "__main__":
    main()
