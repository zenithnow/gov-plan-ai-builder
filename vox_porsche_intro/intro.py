"""VOX 스타일 포르쉐 718 박스터 광고 인트로.

버드뷰(탑다운)로 스카이블루 박스터가 빠르게 들어오다가 드리프트에 들어가는 순간 슬로우모션이 걸리고,
타이어 자국과 종이 컷아웃 같은 연기를 남기며 멈춘다. 그 사이 'PORSCHE / 718 / BOXSTER' 타이포가
슬롯에서 튀어 오르듯 스프링 모션으로 올라온다.

이미지 소스 (Kling 생성, assets/)
  car_raw.png  : 마젠타 배경 위 탑다운 박스터  → 자동 키잉해서 car_cutout.png 로 캐시
  bg_raw.png   : 스카이블루 아스팔트/종이 질감 배경

사용
  python intro.py                       # → output/porsche_718_intro.mp4
  python intro.py --preview 0.4,1.2,2.5 # 지정 시각(초)의 정지 프레임만 PNG 로 저장
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
ASSETS = HERE / "assets"
FONT = HERE / "fonts" / "Archivo-Variable.ttf"

W, H, FPS = 1920, 1080, 30
DURATION = 6.0

# 팔레트 (sRGB)
INK = np.array([16, 40, 61], np.float32)          # 네이비 잉크: 글자·타이어 자국·그림자
PAPER_WHITE = np.array([250, 252, 250], np.float32)
HIGHLIGHT = np.array([255, 212, 71], np.float32)  # VOX 형광펜 노랑
SMOKE = np.array([246, 251, 255], np.float32)
BG_TINT = np.array([180, 213, 224], np.float32)   # 배경 기준색: 차체 도장색(90,162,183)을 흰색 쪽으로 55% 섞은 톤

# ── 동작 설계 (물리 시간 s 기준, 단위 px · 초) ─────────────────────────────
CAR_LEN = 540                    # 화면 위 차 길이
V0 = 2200.0                      # 진입 속도 px/s
START = np.array([-520.0, 760.0])
ARC_START_X = 1000.0             # 직선 → 드리프트 원호로 바뀌는 지점
RADIUS = 400.0                   # 드리프트 원호 반경
STOP_ANGLE = math.radians(55)    # 원호를 이만큼 돌고 정지 (차는 옆으로 미끄러진 채 멈춤)
BETA_MAX = math.radians(42)      # 최대 드리프트(슬립) 각

# 시간 재매핑: 영상 시간 t → 재생 속도 배율 r(t)  (1 = 실시간, 0.11 = 슬로우모션)
SLOWMO = [(0.00, 1.00), (0.575, 1.00), (0.775, 0.11), (2.70, 0.11), (3.90, 0.35)]  # 드리프트 진입 직전에 슬로우가 걸리도록 맞춘 값

# 타이포 타이밍 (영상 시간)
T_718, T_BOXSTER, T_BAR, T_PORSCHE = 1.25, 1.60, 2.05, 2.35


def smoothstep(e0: float, e1: float, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def spring(tau: float, zeta: float = 0.42, freq: float = 2.4) -> float:
    """감쇠 스프링 계단 응답: 0 → 1 로 가면서 한 번 크게 넘쳤다가 자리를 잡는다."""
    if tau <= 0:
        return 0.0
    w = 2 * math.pi * freq
    wd = w * math.sqrt(1 - zeta**2)
    return 1 - math.exp(-zeta * w * tau) * (math.cos(wd * tau) + zeta * w / wd * math.sin(wd * tau))


def rate(t: float) -> float:
    for (t0, r0), (t1, r1) in zip(SLOWMO, SLOWMO[1:]):
        if t <= t1:
            return r0 + (r1 - r0) * float(smoothstep(t0, t1, t))
    return SLOWMO[-1][1]


# ── 차량 궤적 ────────────────────────────────────────────────────────────
L_STRAIGHT = ARC_START_X - START[0]
ARC_LEN = RADIUS * STOP_ANGLE
DECEL = V0**2 / (2 * ARC_LEN)
S_ARC = L_STRAIGHT / V0
S_STOP = S_ARC + V0 / DECEL
CENTER = np.array([ARC_START_X, START[1] - RADIUS])


@dataclass
class Pose:
    pos: np.ndarray   # 차 중심 (화면 px)
    yaw: float        # 차 앞머리 방향 (라디안, +x 기준 시계방향 +)
    speed: float      # px/s (물리)
    beta: float       # 슬립 각
    phi: float        # 원호 진행 각


def pose_at(s: float) -> Pose:
    s = min(max(s, 0.0), S_STOP)
    if s <= S_ARC:
        u = V0 * s
        return Pose(START + [u, 0.0], 0.0, V0, 0.0, 0.0)
    d = s - S_ARC
    u = V0 * d - 0.5 * DECEL * d * d
    phi = u / RADIUS
    p = phi / STOP_ANGLE
    beta = BETA_MAX * float(smoothstep(0.0, 0.42, p)) * (1 - 0.15 * float(smoothstep(0.6, 1.0, p)))
    pos = CENTER + RADIUS * np.array([math.sin(phi), math.cos(phi)])
    return Pose(pos, -phi - beta, V0 - DECEL * d, beta, phi)


def wheel_points(pose: Pose, car_w: float) -> dict[str, np.ndarray]:
    """718 박스터 비율(전장 4.38m, 휠베이스 2.475m)로 잡은 네 바퀴 위치."""
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    fwd, side = np.array([c, s]), np.array([-s, c])
    pts = {}
    for name, ax, lat in (("rl", -0.27, -0.40), ("rr", -0.27, 0.40), ("fl", 0.295, -0.40), ("fr", 0.295, 0.40)):
        pts[name] = pose.pos + fwd * ax * CAR_LEN + side * lat * car_w
    return pts


# ── 에셋 준비 ────────────────────────────────────────────────────────────
def key_car(raw: np.ndarray) -> np.ndarray:
    """마젠타/크림슨 배경 키잉 → RGBA.

    Kling 이 순수 마젠타 대신 그라데이션 진 크림슨을 깔아서, 밝기와 무관한 Lab a*(초록↔빨강) 채널과
    밝기로 나눈 붉은 정도 (R-G)/(R+G+B) 를 함께 써서 뺀다.
    배경 a* ≈ +46~50, 하늘색 차체·검은 실내 a* ≤ +8 이라 경계가 깨끗하다.
    """
    lab = cv2.cvtColor(raw, cv2.COLOR_RGB2LAB).astype(np.float32)
    a_star = lab[..., 1] - 128
    # 차 아래 그림자는 (36,5,16) 처럼 어두워서 a* 가 15 정도로 눌린다 → 밝기로 나눈 붉은 정도로 한 번 더 거른다
    rgb_f = raw.astype(np.float32)
    redness = (rgb_f[..., 0] - rgb_f[..., 1]) / (rgb_f.sum(axis=2) + 12)
    alpha = 1 - np.maximum(np.asarray(smoothstep(18, 32, a_star), np.float32),
                           np.asarray(smoothstep(0.20, 0.32, redness), np.float32))
    # 가장 큰 덩어리(차)만 남기고, 윤곽 안쪽 구멍(빨간 테일램프 등)은 메움
    solid = (alpha > 0.5).astype(np.uint8)
    solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(solid, 8)
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    body = (labels == keep).astype(np.uint8)
    contours, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(body)
    cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
    inner = cv2.erode(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(np.float32)
    alpha = np.where(filled > 0, np.maximum(alpha, inner), 0).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.7)
    # 디스필: 가장자리 띠에서만 붉은 반사(a*>0)를 0 쪽으로 눌러줌
    band = (filled > 0) & (inner < 0.5)
    lab_d = lab.copy()
    lab_d[..., 1] = np.where(band, 128 + np.minimum(a_star, 2), lab[..., 1])
    rgb = cv2.cvtColor(np.clip(lab_d, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    out = np.dstack([rgb, np.clip(alpha * 255, 0, 255).astype(np.uint8)])
    ys, xs = np.nonzero(alpha > 0.02)
    return out[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def load_car() -> tuple[np.ndarray, np.ndarray, float]:
    """(차 RGBA premultiplied float, 스티커 외곽 알파, 화면상 차폭) 반환. 차 앞머리는 +x."""
    cache = ASSETS / "car_cutout.png"
    if not cache.exists():
        Image.fromarray(key_car(np.asarray(Image.open(ASSETS / "car_raw.png").convert("RGB")))).save(cache)
    car = np.asarray(Image.open(cache).convert("RGBA"))
    k = CAR_LEN / car.shape[1]
    car = cv2.resize(car, (CAR_LEN, round(car.shape[0] * k)), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    pad = 24
    car = cv2.copyMakeBorder(car, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    a = car[..., 3]
    # VOX 콜라주 느낌: 흰 스티커 테두리
    border = cv2.dilate((a > 0.4).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    border = cv2.GaussianBlur(border.astype(np.float32), (0, 0), 1.2)
    rgb = car[..., :3] * a[..., None] + (PAPER_WHITE / 255) * (border - a)[..., None].clip(0)
    alpha = np.maximum(a, border)
    premul = np.dstack([rgb, alpha]).astype(np.float32)
    car_w = float((a > 0.5).any(axis=1).sum())
    return premul, border, car_w


def load_background() -> np.ndarray:
    bg = np.asarray(Image.open(ASSETS / "bg_raw.png").convert("RGB")).astype(np.float32)
    h, w = bg.shape[:2]
    k = max(W / w, H / h)
    bg = cv2.resize(bg, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
    y0, x0 = (bg.shape[0] - H) // 2, (bg.shape[1] - W) // 2
    bg = bg[y0:y0 + H, x0:x0 + W]
    # Kling 배경의 종이 질감(밝기 변화)만 살리고 색은 차체 하늘색 계열로 통일
    lum = bg @ np.array([0.299, 0.587, 0.114], np.float32)
    rel = (lum - lum.mean()) / lum.mean()
    return BG_TINT * (1 + 0.6 * rel)[..., None]


def paper_textures(n: int, rng: np.random.Generator) -> list[np.ndarray]:
    out = []
    for _ in range(n):
        g = np.zeros((H, W), np.float32)
        for scale, weight in ((2, 0.4), (6, 0.35), (32, 0.25)):
            small = rng.standard_normal((H // scale, W // scale)).astype(np.float32)
            g += weight * cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)
        # 종이 섬유 결
        fib = rng.standard_normal((H // 3, W // 40)).astype(np.float32)
        g += 0.25 * cv2.resize(fib, (W, H), interpolation=cv2.INTER_CUBIC)
        g /= g.std() + 1e-6
        out.append(g[..., None])
    return out


# ── 타이포 ──────────────────────────────────────────────────────────────
def font(size: int, weight: int, width: int) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(str(FONT), size)
    f.set_variation_by_axes([weight, width])
    return f


@dataclass
class Glyph:
    img: np.ndarray   # premultiplied RGBA float
    x: int
    y: int           # 최종 위치 (좌상단)
    t0: float        # 등장 시작 시각
    clip_y: int      # 이 y 아래는 가려짐 (글자가 슬롯에서 올라오는 느낌)
    rise: float      # 시작할 때 아래로 내려가 있는 거리
    rot0: float      # 시작 회전각 (도)
    zeta: float = 0.42
    freq: float = 2.4


GLYPH_PAD = 24


def render_glyph(ch: str, f: ImageFont.FreeTypeFont, fill, shadow=None, shadow_off=(0, 0)) -> np.ndarray:
    """글자 하나를 premultiplied RGBA 로. 글리프 bbox 좌상단이 (GLYPH_PAD, GLYPH_PAD) 에 온다."""
    l, t, r, b = f.getbbox(ch)
    im = Image.new("RGBA", (r - l + 2 * GLYPH_PAD, b - t + 2 * GLYPH_PAD), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if shadow is not None:
        d.text((GLYPH_PAD - l + shadow_off[0], GLYPH_PAD - t + shadow_off[1]), ch, font=f, fill=shadow)
    d.text((GLYPH_PAD - l, GLYPH_PAD - t), ch, font=f, fill=fill)
    a = np.asarray(im).astype(np.float32) / 255
    a[..., :3] *= a[..., 3:4]
    return a


def layout_text() -> tuple[list[Glyph], dict]:
    ink = tuple(int(v) for v in INK) + (255,)
    x0 = 132
    glyphs: list[Glyph] = []

    def line(text, f, cap_top, t0, stagger, fill, tracking=0.0, shadow=None, shadow_off=(0, 0), rot=6.0, zeta=0.42, freq=2.4):
        """cap_top: 대문자 윗선의 화면 y. 반환: (끝 x, 대문자 윗선 y, 기준선 y)"""
        _, cap_t, _, cap_b = f.getbbox("H")
        origin_y = cap_top - cap_t
        baseline = origin_y + cap_b
        x = float(x0)
        for i, ch in enumerate(text):
            if ch != " ":
                l, t, _, _ = f.getbbox(ch)
                glyphs.append(Glyph(
                    img=render_glyph(ch, f, fill, shadow, shadow_off),
                    x=int(round(x + l - GLYPH_PAD)), y=int(round(origin_y + t - GLYPH_PAD)),
                    t0=t0 + i * stagger, clip_y=int(baseline + 4 + shadow_off[1]),
                    rise=(cap_b - cap_t) * 1.25 + shadow_off[1], rot0=rot * (1 if i % 2 else -1), zeta=zeta, freq=freq))
            x += f.getlength(ch) + tracking * f.size
        return x, cap_top, baseline

    f_p = font(46, 650, 125)
    f_7 = font(300, 900, 100)
    f_b = font(118, 900, 125)
    meta = {"x0": x0}
    meta["seven"] = line("718", f_7, 380, T_718, 0.09, (250, 252, 250, 255), shadow=ink, shadow_off=(12, 12), rot=7, zeta=0.62, freq=2.1)
    meta["boxster"] = line("BOXSTER", f_b, 660, T_BOXSTER, 0.045, ink, tracking=0.02, rot=5, zeta=0.42, freq=2.6)
    meta["porsche"] = line("PORSCHE", f_p, 318, T_PORSCHE, 0.035, ink, tracking=0.38, rot=0, zeta=0.7, freq=2.2)
    return glyphs, meta


def highlighter_mask(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    """가장자리가 살짝 거친 형광펜 띠."""
    m = np.zeros((h, w), np.float32)
    edge = cv2.resize(rng.standard_normal((1, w // 24 + 2)).astype(np.float32), (w, 1), interpolation=cv2.INTER_CUBIC)[0]
    top = (h * 0.08 + edge * 3).astype(int)
    bot = (h * 0.95 + np.roll(edge, 7) * 3).astype(int)
    for x in range(w):
        m[max(0, top[x]):min(h, bot[x]), x] = 1
    m *= 0.92 + 0.08 * cv2.resize(rng.random((h // 6 + 1, w // 30 + 1)).astype(np.float32), (w, h))
    return cv2.GaussianBlur(m, (0, 0), 0.8)


# ── 합성 유틸 ────────────────────────────────────────────────────────────
def over(dst: np.ndarray, premul: np.ndarray, x: int, y: int) -> None:
    """dst(H,W,3 float 0~1) 위에 premultiplied RGBA 를 (x,y) 좌상단에 얹는다 (화면 밖은 잘라냄)."""
    h, w = premul.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, dst.shape[1]), min(y + h, dst.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    p = premul[y0 - y:y1 - y, x0 - x:x1 - x]
    region = dst[y0:y1, x0:x1]
    region *= 1 - p[..., 3:4]
    region += p[..., :3]


def car_matrix(sprite_shape, pos: np.ndarray, yaw: float) -> np.ndarray:
    h, w = sprite_shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), -math.degrees(yaw), 1.0)
    m[:, 2] += pos - [w / 2, h / 2]
    return m


class Renderer:
    def __init__(self, seed: int = 11):
        self.rng = np.random.default_rng(seed)
        self.bg = load_background() / 255
        self.car, self.border, self.car_w = load_car()
        self.shadow = cv2.GaussianBlur(self.border, (0, 0), 9)
        self.papers = paper_textures(4, self.rng)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / math.sqrt(2)
        self.vignette = (1 - 0.16 * r**2.2)[..., None]
        self.marks = np.zeros((H, W), np.float32)
        self.mark_tex = 0.72 + 0.28 * np.clip(paper_textures(1, self.rng)[0][..., 0] * 0.5 + 0.5, 0, 1)
        self.smoke: list[tuple] = []
        self.glyphs, self.meta = layout_text()
        xe_b, top_b, bot_b = self.meta["boxster"]
        self.bar = highlighter_mask(int(xe_b - self.meta["x0"] + 44), int((bot_b - top_b) * 0.78), self.rng)
        self.s_prev = 0.0
        self.prev_wheels = wheel_points(pose_at(0.0), self.car_w)

    # 물리 시간 진행: 타이어 자국·연기 입자 갱신
    def advance(self, s_to: float) -> None:
        steps = max(1, int(math.ceil((s_to - self.s_prev) * V0 / 3)))
        for k in range(1, steps + 1):
            s = self.s_prev + (s_to - self.s_prev) * k / steps
            pose = pose_at(s)
            wheels = wheel_points(pose, self.car_w)
            slip = math.sin(pose.beta) / math.sin(BETA_MAX)
            if slip > 0.05:
                for name in ("rl", "rr"):
                    p, q = wheels[name], self.prev_wheels[name]
                    self._stroke(q, p, 30, min(1.0, 0.25 + slip * 1.5))
                    if self.rng.random() < 0.22 * slip:
                        outward = (p - CENTER) / (np.linalg.norm(p - CENTER) + 1e-6)
                        vel = outward * self.rng.uniform(160, 320) + self.rng.normal(0, 70, 2)
                        self.smoke.append((p.copy(), vel, s, self.rng.uniform(0.22, 0.38), self.rng.uniform(18, 30), self.rng.uniform(70, 115)))
            self.prev_wheels = wheels
        self.s_prev = s_to

    def _stroke(self, a: np.ndarray, b: np.ndarray, width: int, strength: float) -> None:
        x0, y0 = np.floor(np.minimum(a, b) - width).astype(int)
        x1, y1 = np.ceil(np.maximum(a, b) + width).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if x0 >= x1 or y0 >= y1:
            return
        tmp = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.line(tmp, tuple(np.round(a - [x0, y0]).astype(int)), tuple(np.round(b - [x0, y0]).astype(int)), int(255 * strength), width, cv2.LINE_AA)
        np.maximum(self.marks[y0:y1, x0:x1], tmp.astype(np.float32) / 255, out=self.marks[y0:y1, x0:x1])

    def smoke_layer(self, s: float) -> np.ndarray:
        q = 4
        dens = np.zeros((H // q, W // q), np.float32)
        alive = []
        for p0, vel, born, life, r0, r1 in self.smoke:
            age = s - born
            if age < 0:
                continue
            if age > life:
                continue
            alive.append((p0, vel, born, life, r0, r1))
            f = age / life
            pos = p0 + vel * life * (1 - math.exp(-3 * f)) / 3
            rad = r0 + (r1 - r0) * (1 - math.exp(-4 * f))
            a = 0.9 * (1 - f) ** 1.3 * float(smoothstep(0.0, 0.06, f))
            tmp = np.zeros_like(dens)
            cv2.circle(tmp, (int(pos[0] / q), int(pos[1] / q)), max(1, int(rad / q)), a, -1, cv2.LINE_AA)
            dens += tmp
        self.smoke = [p for p in self.smoke if s - p[2] <= p[3]]
        dens = cv2.GaussianBlur(dens, (0, 0), 3.0)
        # 종이 오려 붙인 연기: 농도를 3단계로 끊어서 평평한 층으로
        layers = sum(float(w) * smoothstep(t - 0.04, t + 0.04, dens) for t, w in ((0.20, 0.26), (0.60, 0.20), (1.1, 0.18)))
        return cv2.resize(np.asarray(layers, np.float32), (W, H), interpolation=cv2.INTER_LINEAR)[..., None]

    def world(self, s: float, s_shutter: float, play_rate: float = 1.0) -> np.ndarray:
        img = self.bg.copy()
        # 타이어 자국
        ma = (cv2.GaussianBlur(self.marks, (0, 0), 0.9) * self.mark_tex * 0.92)[..., None]
        img = img * (1 - ma) + (INK / 255 * 0.45) * ma
        # 연기 (차 아래)
        sm = self.smoke_layer(s)
        img = img * (1 - sm) + (SMOKE / 255) * sm
        # 속도선
        pose = pose_at(s)
        speed_vis = pose.speed / V0 * float(smoothstep(0.2, 0.7, play_rate))
        if speed_vis > 0.05:
            lines = np.zeros((H, W), np.uint8)
            c, sn = math.cos(pose.yaw), math.sin(pose.yaw)
            fwd, side = np.array([c, sn]), np.array([-sn, c])
            rng = np.random.default_rng(int(s * 1000))
            for lat in np.sort(rng.uniform(-0.6, 0.6, 6)):
                start = pose.pos - fwd * (CAR_LEN * 0.55 + rng.uniform(0, 140)) + side * lat * self.car_w * 1.2
                end = start - fwd * rng.uniform(180, 700) * speed_vis
                cv2.line(lines, tuple(start.astype(int)), tuple(end.astype(int)), 255, int(rng.integers(2, 6)), cv2.LINE_AA)
            la = (lines.astype(np.float32) / 255 * 0.55 * speed_vis)[..., None]
            img = img * (1 - la) + la
        # 차 그림자 + 차 (셔터 구간을 나눠 평균 → 모션블러)
        disp = abs(s - s_shutter) * pose.speed
        n = int(min(14, max(1, math.ceil(disp / 5))))
        acc = np.zeros((H, W, 4), np.float32)
        sh = np.zeros((H, W), np.float32)
        for i in range(n):
            si = s_shutter + (s - s_shutter) * (i + 0.5) / n if n > 1 else s
            pi = pose_at(si)
            m = car_matrix(self.car.shape, pi.pos, pi.yaw)
            acc += cv2.warpAffine(self.car, m, (W, H), flags=cv2.INTER_LINEAR)
            ms = m.copy()
            ms[:, 2] += [14, 20]
            sh += cv2.warpAffine(self.shadow, ms, (W, H), flags=cv2.INTER_LINEAR)
        acc /= n
        sh = (sh / n * 0.42)[..., None]
        img = img * (1 - sh) + (INK / 255 * 0.6) * sh
        img = img * (1 - acc[..., 3:4]) + acc[..., :3]
        return img

    def text(self, img: np.ndarray, t: float) -> np.ndarray:
        boil = np.random.default_rng(int(t * 12))  # 글자는 초당 12번만 미세하게 흔들림 (손작업 느낌)
        # 형광펜 띠: BOXSTER 뒤를 왼→오 로 칠함
        _, top_b, bot_b = self.meta["boxster"]
        if t > T_BAR:
            prog = float(smoothstep(0, 1, (t - T_BAR) / 0.38))
            bh, bw = self.bar.shape
            cut = 1 - np.clip((np.arange(bw) - prog * bw) / 10, 0, 1)
            m = (self.bar * cut[None, :])[..., None]
            by = int(bot_b - bh * 0.92)
            bx = self.meta["x0"] - 22
            region = img[by:by + bh, bx:bx + bw]
            region[:] = region * (1 - m) + (HIGHLIGHT / 255) * m
        # PORSCHE 옆 가는 선
        xe_p, top_p, bot_p = self.meta["porsche"]
        if t > T_PORSCHE + 0.3:
            ln = int(240 * float(smoothstep(0, 1, (t - T_PORSCHE - 0.3) / 0.5)))
            y = (top_p + bot_p) // 2
            img[y - 2:y + 2, int(xe_p) + 10:int(xe_p) + 10 + ln] = INK / 255
        for g in self.glyphs:
            tau = t - g.t0
            if tau <= 0:
                continue
            k = spring(tau, g.zeta, g.freq)
            dy = g.rise * (1 - k)
            ang = g.rot0 * (1 - k)
            gi = g.img
            if abs(ang) > 0.05:
                h, w = gi.shape[:2]
                m = cv2.getRotationMatrix2D((w / 2, h), ang, 1.0)
                gi = cv2.warpAffine(gi, m, (w, h), flags=cv2.INTER_LINEAR)
            jx, jy = boil.uniform(-0.8, 0.8, 2)
            x, y = int(round(g.x + jx)), int(round(g.y + dy + jy))
            # 슬롯 아래(clip_y)로 내려가 있는 부분은 숨김
            visible = max(0, min(gi.shape[0], g.clip_y - y))
            if visible > 0:
                over(img, gi[:visible], x, y)
        return img

    def frame(self, t: float, s: float, s_shutter: float, cam: tuple[float, float, float]) -> np.ndarray:
        world = self.world(s, s_shutter, rate(t))
        zoom, sx, sy = cam
        cx, cy = 1120.0, 560.0
        m = np.array([[zoom, 0, cx - zoom * cx + sx], [0, zoom, cy - zoom * cy + sy]], np.float32)
        img = cv2.warpAffine(world, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        img = self.text(img, t)
        paper = self.papers[int(t * 12) % len(self.papers)]
        img = img * (1 + 0.045 * paper) * self.vignette
        return np.clip(img * 255, 0, 255).astype(np.uint8)


def camera(t: float, rng: np.random.Generator) -> tuple[float, float, float]:
    shake = 5.0 * (1 - float(smoothstep(0.42, 0.78, t)))
    punch = 0.035 * math.exp(-((t - 0.78) / 0.18) ** 2)       # 슬로우모션 걸리는 순간 살짝 밀고 들어감
    zoom = 1.02 + 0.045 * float(smoothstep(0.6, 4.5, t)) + punch
    return zoom, rng.uniform(-shake, shake), rng.uniform(-shake, shake)


def timeline(total: int, sub: int = 8) -> list[tuple[float, float]]:
    """각 영상 프레임의 (물리 시간 s, 셔터 열린 시점 s) — 180도 셔터."""
    out, s = [], 0.0
    dt = 1 / FPS
    for i in range(total):
        t = i * dt
        s_open = s
        for k in range(sub):
            s += rate(t + dt * k / sub) * dt / sub
            if k == sub // 2 - 1:
                s_open = s
        out.append((min(s, S_STOP), min(s_open, S_STOP)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "output" / "porsche_718_intro.mp4")
    ap.add_argument("--preview", type=str, default="", help="쉼표로 구분한 시각(초)들의 정지 프레임만 저장")
    ap.add_argument("--crf", type=int, default=16)
    args = ap.parse_args()

    total = int(DURATION * FPS)
    tl = timeline(total)
    rnd = Renderer()
    cam_rng = np.random.default_rng(3)
    cams = [camera(i / FPS, cam_rng) for i in range(total)]
    wanted = sorted({min(total - 1, round(float(v) * FPS)) for v in args.preview.split(",") if v.strip()})

    enc = None
    if not wanted:
        if not shutil.which("ffmpeg"):
            raise SystemExit("ffmpeg 가 필요합니다.")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        enc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
             "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(args.out)],
            stdin=subprocess.PIPE)

    last = wanted[-1] if wanted else total - 1
    for i in range(last + 1):
        s, s_open = tl[i]
        rnd.advance(s)
        if enc is None and i not in wanted:
            continue
        img = rnd.frame(i / FPS, s, s_open, cams[i])
        if enc is not None:
            enc.stdin.write(img.tobytes())
        else:
            p = args.out.parent / f"preview_{i / FPS:05.2f}s.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(p)
            print("저장:", p)
    if enc is not None:
        enc.stdin.close()
        if enc.wait() != 0:
            raise SystemExit("ffmpeg 인코딩 실패")
        print(f"영상 저장: {args.out}  ({total}프레임, {DURATION:.1f}초, 정지 시점 영상시간≈{next((i for i, (s, _) in enumerate(tl) if s >= S_STOP), total) / FPS:.2f}초)")


if __name__ == "__main__":
    main()
