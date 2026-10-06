"""VOX 스타일 '콜럼버스의 항해' 인트로.

아날로그 에디토리얼 콜라주: 인쇄물을 손으로 오려 테이블 위 옛 해도에 배치하고 사진으로 찍은 느낌.

  0.0s  옛 해도를 정수리에서 내려다본 버드뷰. 종이 구름이 떠 있고, 팔로스에서 잉크 항로가 그려진다.
  1.7s  카메라가 해도면에서 30° 기울어지며 카나리아 제도로 다가간다 (틸트시프트 심도).
  3.3s  배 컷아웃이 해도 '속'에서 솟아올랐다가(오버슈트) 살짝 가라앉고, 흔들흔들 뜬다.
  4.3s  배가 항로선을 따라 서쪽으로 항해, 지나간 자리에 점선 항로가 찍힌다. 구름은 무역풍 따라 서쪽으로.
  5.2s  타이틀 '1492' 컷아웃 + 'THE FIRST VOYAGE OF COLUMBUS' 종이띠가 손으로 놓이듯 붙는다.
  9.4s  과나하니 도착 태그.

사용
  python cutouts.py                 # (처음 한 번) Kling 이미지 → 종이 컷아웃 캐시
  python intro.py                   # → output/columbus_1492_intro.mp4
  python intro.py --preview 0.5,3.6 # 지정 시각 정지 프레임만
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

from cutouts import ASSETS, hand_cut, smoothstep

HERE = Path(__file__).parent
FONTS = HERE / "fonts"

W, H, FPS = 1920, 1080, 24
DURATION = 11.0
FOCAL = 2640.0                  # 가로 화각 약 40°
TILT = math.radians(30)         # 해도면에서 기울어지는 각도
ON_TWOS = True                  # 배·구름·라벨은 초당 12장으로 끊어서 (스톱모션)

MAP_W, MAP_H = 5440, 3072
INK = np.array([112, 34, 22], np.float32) / 255          # 항로 잉크 (바랜 적갈색)
TABLE = (40, 31, 25)                                       # 해도 밖 책상색
LIGHT = np.array([0.26, -0.20])                            # 높이 1 당 그림자가 지는 방향 (북서쪽 조명)

# 항로 (해도 픽셀 좌표). 해도는 생성 이미지라 실제 지리와는 다르고, 그림 위 위치로 잡았다.
PALOS = (3545, 1925)
CANARIES = (3060, 2470)
LANDFALL_ISLE = (820, 2245)
ROUTE = [PALOS, (3420, 2060), (3300, 2230), (3170, 2390), CANARIES,
         (2600, 2492), (2050, 2452), (1500, 2362), (1010, 2262)]
CANARY_INDEX = 4

SHIP_LEN = 560.0                # 해도 위 배 길이 (해도 px)
SHIP_LEAN = math.radians(20)    # 배가 뒤로 기댄 각 (팝업북 받침대처럼)
SHIP_UP = 3.6                   # 배가 해도에서 솟아오르기 시작하는 시각

# 구름: (해도 x, y, 높이 z, 폭, 스프라이트 번호, 좌우반전)
CLOUDS = [
    (1450, 820, 600, 900, 0, False),    # 북서 대양 (버드뷰에서 보임)
    (3950, 1350, 520, 820, 2, True),    # 이베리아 위
    (2350, 1820, 420, 760, 1, False),   # 항로 북쪽 (기울어진 뒤 먼 배경)
    (4500, 2700, 560, 860, 3, False),   # 아프리카 위
    (3150, 3060, 600, 700, 2, False),   # 카나리아 남쪽 → 출발 때 화면 아래 전경
    (1850, 3040, 600, 700, 0, True),    # 대서양 한가운데 남쪽 → 항해 중 전경
    (650, 1640, 460, 760, 3, True),     # 카리브 북쪽 (도착 때 배경)
    (1250, 1930, 380, 620, 1, True),    # 도착 직전 항로 북쪽
]
WIND = -22.0                    # 구름 이동 (해도 px/s, 음수 = 서쪽: 콜럼버스를 밀어준 무역풍)


def smoother(x):
    x = np.clip(x, 0, 1)
    return x * x * x * (x * (x * 6 - 15) + 10)


def spring(tau: float, zeta: float, freq: float) -> float:
    if tau <= 0:
        return 0.0
    w = 2 * math.pi * freq
    wd = w * math.sqrt(1 - zeta**2)
    return 1 - math.exp(-zeta * w * tau) * (math.cos(wd * tau) + zeta * w / wd * math.sin(wd * tau))


def to_world(mx, my, z=0.0) -> np.ndarray:
    return np.array([mx - MAP_W / 2, MAP_H / 2 - my, z], np.float64)


# ── 항로 ────────────────────────────────────────────────────────────────
def catmull_rom(points: list[tuple[float, float]], per_seg: int = 40) -> np.ndarray:
    p = np.array(points, np.float64)
    p = np.vstack([p[0] * 2 - p[1], p, p[-1] * 2 - p[-2]])
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for t in np.linspace(0, 1, per_seg, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(p[-2])
    return np.array(out)


class Route:
    def __init__(self):
        self.pts = catmull_rom(ROUTE)
        seg = np.linalg.norm(np.diff(self.pts, axis=0), axis=1)
        self.s = np.concatenate([[0], np.cumsum(seg)])
        idx = np.argmin(np.linalg.norm(self.pts - np.array(CANARIES), axis=1))
        self.s_canary = self.s[idx]
        self.length = self.s[-1]
        rng = np.random.default_rng(7)
        self.wob = rng.normal(0, 1, len(self.pts))  # 펜 압력 흔들림

    def at(self, s: float) -> tuple[np.ndarray, np.ndarray]:
        s = float(np.clip(s, 0, self.length))
        i = int(np.clip(np.searchsorted(self.s, s) - 1, 0, len(self.pts) - 2))
        f = (s - self.s[i]) / max(1e-6, self.s[i + 1] - self.s[i])
        p = self.pts[i] * (1 - f) + self.pts[i + 1] * f
        j0, j1 = max(0, i - 6), min(len(self.pts) - 1, i + 7)
        tan = self.pts[j1] - self.pts[j0]
        return p, tan / (np.linalg.norm(tan) + 1e-9)


# ── 카메라 ──────────────────────────────────────────────────────────────
@dataclass
class Camera:
    target: np.ndarray   # 화면 중앙에 오는 해도 위 점 (월드 xy)
    pitch: float         # 0 = 정수리 버드뷰
    scale: float         # 중앙에서 해도 1px 이 화면 몇 px 인지

    def __post_init__(self):
        a = self.pitch
        self.D = FOCAL / self.scale
        t = np.array([self.target[0], self.target[1], 0.0])
        self.pos = t + self.D * np.array([0.0, -math.sin(a), math.cos(a)])
        self.fwd = np.array([0.0, math.sin(a), -math.cos(a)])
        self.right = np.array([1.0, 0.0, 0.0])
        self.up = np.cross(self.right, self.fwd)

    def project(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = np.atleast_2d(P) - self.pos
        x, y, z = d @ self.right, d @ self.up, d @ self.fwd
        return np.stack([W / 2 + FOCAL * x / z, H / 2 - FOCAL * y / z], 1), z


def homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    return cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))


def warp_rgba(sprite: np.ndarray, Hm: np.ndarray, dst_pts: np.ndarray):
    """premultiplied RGBA 스프라이트를 화면에 투영. 화면에 걸친 bbox 패치만 계산해서 반환."""
    x0, y0 = np.floor(dst_pts.min(0)).astype(int) - 2
    x1, y1 = np.ceil(dst_pts.max(0)).astype(int) + 2
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x1 <= x0 or y1 <= y0:
        return None
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
    patch = cv2.warpPerspective(sprite, T @ Hm, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return patch, x0, y0


def over(img: np.ndarray, warped) -> None:
    if warped is None:
        return
    patch, x0, y0 = warped
    h, w = patch.shape[:2]
    reg = img[y0:y0 + h, x0:x0 + w]
    a = patch[..., 3:4] if patch.ndim == 3 else patch[..., None]
    reg *= 1 - a
    reg += patch[..., :3]


def premul(rgba_u8: np.ndarray) -> np.ndarray:
    f = rgba_u8.astype(np.float32) / 255
    f[..., :3] *= f[..., 3:4]
    return f


def load_rgba(name: str, max_w: int | None = None) -> np.ndarray:
    im = Image.open(ASSETS / name).convert("RGBA")
    if max_w and im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    return np.asarray(im)


# ── 타이포·라벨 (인쇄된 종이) ───────────────────────────────────────────
def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS / name), size)


def ink_text(canvas: np.ndarray, text: str, f, xy, color=(38, 28, 22), rng=None, spacing=0.0) -> None:
    """종이 위에 활판 인쇄처럼 글자를 찍는다 (곱하기 합성 + 잉크 얼룩)."""
    h, w = canvas.shape[:2]
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=f, fill=255)
        x += f.getlength(ch) + spacing * f.size
    a = np.asarray(m).astype(np.float32) / 255
    a = cv2.GaussianBlur(a, (0, 0), 0.6)
    if rng is not None:
        a *= np.clip(0.82 + 0.25 * cv2.resize(rng.random((h // 6 + 1, w // 6 + 1)).astype(np.float32), (w, h)), 0, 1)
    rgb = canvas[..., :3].astype(np.float32)
    ink = np.array(color, np.float32)
    canvas[..., :3] = np.clip(rgb * (1 - a[..., None]) + (rgb * ink / 255) * a[..., None], 0, 255).astype(np.uint8)


def text_width(text: str, f, spacing=0.0) -> float:
    return sum(f.getlength(ch) + spacing * f.size for ch in text)


def make_strip_label(strip: np.ndarray, text: str, f, pad_x: int, height: int, rng, spacing=0.0) -> np.ndarray:
    tw = text_width(text, f, spacing)
    w = int(tw + 2 * pad_x)
    lab = cv2.resize(strip, (w, height), interpolation=cv2.INTER_AREA).copy()
    _, t, _, b = f.getbbox("H")
    ink_text(lab, text, f, (pad_x, (height - (b - t)) / 2 - t), rng=rng, spacing=spacing)
    return lab


def make_cut_text(text: str, f, rng) -> np.ndarray:
    """글자를 인쇄한 종이를 글자 모양대로 가위로 오린 컷아웃."""
    l, t, r, b = f.getbbox(text)
    pad = 60
    w, h = r - l + 2 * pad, b - t + 2 * pad
    raw = np.zeros((h, w, 3), np.uint8)
    raw[:] = (0, 177, 64)
    im = Image.fromarray(raw)
    ImageDraw.Draw(im).text((pad - l, pad - t), text, font=f, fill=(40, 30, 24))
    raw = np.asarray(im).copy()
    body = (np.abs(raw.astype(int) - [0, 177, 64]).sum(2) > 60).astype(np.float32)
    cut = hand_cut(raw, body, margin=18, bridge=41, rng=rng)
    # 잉크를 조금 바랜 인쇄 느낌으로
    cut[..., :3] = np.clip(cut[..., :3].astype(np.float32) * (0.97 + 0.03 * rng.random(cut.shape[:2]))[..., None], 0, 255).astype(np.uint8)
    return cut


# ── 렌더러 ──────────────────────────────────────────────────────────────
class Scene:
    def __init__(self, seed: int = 1492):
        self.rng = np.random.default_rng(seed)
        mp = np.asarray(Image.open(ASSETS / "map_raw.jpg").convert("RGB"))
        assert mp.shape[1] == MAP_W and mp.shape[0] == MAP_H, mp.shape
        self.mips = [mp, cv2.resize(mp, (MAP_W // 2, MAP_H // 2), interpolation=cv2.INTER_AREA),
                     cv2.resize(mp, (MAP_W // 4, MAP_H // 4), interpolation=cv2.INTER_AREA)]
        self.route = Route()

        ship = load_rgba("cut_ship.png", max_w=900)
        self.ship = premul(ship)
        self.ship_k = SHIP_LEN / ship.shape[1]                         # 해도 px / 스프라이트 px
        self.ship_vw = ship.shape[0] - 14                             # 흘수선 (스프라이트 행)
        uu, vv = np.meshgrid(np.arange(ship.shape[1], dtype=np.float32), np.arange(ship.shape[0], dtype=np.float32))
        self.ship_uv = (uu, vv)

        self.clouds = [premul(load_rgba(f"cut_cloud_{i}.png", max_w=700)) for i in range(4)]
        strips = [load_rgba(f"cut_strip_{i}.png") for i in range(3)]

        # 타이틀
        self.title_1492 = premul(make_cut_text("1492", font("OldStandardTT-Bold.ttf", 200), self.rng))
        self.title_strip = premul(make_strip_label(strips[0], "THE FIRST VOYAGE OF COLUMBUS", font("IMFeENsc28P.ttf", 50),
                                                   38, 92, self.rng, spacing=0.06))
        # 해도 위에 놓이는 날짜 태그 (해도 px 단위 크기로 만든 뒤 3D 로 배치)
        f_tag = font("CourierPrime-Bold.ttf", 44)
        self.tags = [
            dict(img=premul(make_strip_label(strips[1], "PALOS · 3 AUG 1492", f_tag, 30, 84, self.rng)),
                 anchor=PALOS, offset=(-60, -150), t0=1.05, rot=-3.0),
            dict(img=premul(make_strip_label(strips[2], "CANARIAS · 6 SEP 1492", f_tag, 30, 84, self.rng)),
                 anchor=CANARIES, offset=(330, 30), t0=3.9, rot=2.0),
            dict(img=premul(make_strip_label(strips[1], "GUANAHANI · 12 OCT 1492", f_tag, 30, 84, self.rng)),
                 anchor=LANDFALL_ISLE, offset=(-40, 205), t0=9.7, rot=-2.5),
        ]
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / math.sqrt(2)
        lamp = 1.06 - 0.16 * np.clip((xx / W) * 0.6 + (yy / H) * 0.4, 0, 1)   # 왼쪽 위 스탠드 조명
        self.light = ((1 - 0.30 * r ** 2.4) * lamp)[..., None].astype(np.float32)
        self.yy = yy
        self.grains = []
        for _ in range(4):
            g = self.rng.standard_normal((H, W)).astype(np.float32)
            g = 0.6 * g + 0.4 * cv2.resize(self.rng.standard_normal((H // 3, W // 3)).astype(np.float32), (W, H))
            self.grains.append(g[..., None])

    # ── 타임라인 ──
    def camera(self, t: float) -> Camera:
        start = np.array([0.0, 0.0])
        canary = to_world(*CANARIES)[:2] + [-120.0, 200.0]   # 배가 화면 아래쪽에 서도록 (위는 타이틀 자리)
        s_top0, s_top1, s_near = 0.362, 0.392, 0.92
        if t < 1.7:
            k = smoother(t / 1.7)
            return Camera(start, 0.0, s_top0 + (s_top1 - s_top0) * k)
        if t < 3.9:
            k = smoother((t - 1.7) / 2.2)
            tgt = start + (canary - start) * k
            sc = math.exp(math.log(s_top1) + (math.log(s_near) - math.log(s_top1)) * k)
            return Camera(tgt, TILT * k, sc)
        # 배를 느슨하게 따라감
        ship_xy = self.ship_xy(t)
        follow = ship_xy + [-200.0, 340.0]
        k = smoother((t - 3.9) / 1.4)
        tgt = canary + (follow - canary) * k
        # 해도 밖(책상)이 화면에 나오지 않도록 좌우 한계
        tgt[0] = np.clip(tgt[0], 1380 - MAP_W / 2, MAP_W / 2 - 1380)
        sc = s_near - 0.07 * smoother((t - 4.0) / 6.0)
        return Camera(tgt, TILT, sc)

    def ship_dist(self, t: float) -> float:
        k = smoother((t - 4.5) / 5.2)
        return self.route.s_canary + (self.route.length - self.route.s_canary) * float(k)

    def ship_xy(self, t: float) -> np.ndarray:
        p, _ = self.route.at(self.ship_dist(t))
        return to_world(*p)[:2]

    # ── 그리기 ──
    def draw_map(self, cam: Camera) -> np.ndarray:
        q = 4 if cam.scale * 4 <= 1.05 else 2 if cam.scale * 2 <= 1.05 else 1
        tex = self.mips[{1: 0, 2: 1, 4: 2}[q]]
        corners = np.array([[0, 0], [MAP_W, 0], [MAP_W, MAP_H], [0, MAP_H]], np.float64)
        dst, _ = cam.project(np.array([to_world(x, y) for x, y in corners]))
        Hm = homography(corners / q, dst)
        img = cv2.warpPerspective(tex, Hm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=TABLE)
        return img.astype(np.float32) / 255

    def draw_route(self, img: np.ndarray, cam: Camera, upto: float, t: float) -> None:
        r = self.route
        layer = np.zeros((H, W), np.uint8)
        n = np.searchsorted(r.s, upto)
        if n < 2:
            return
        world = np.c_[r.pts[:n, 0] - MAP_W / 2, MAP_H / 2 - r.pts[:n, 1], np.zeros(n)]
        scr, depth = cam.project(world)
        dash, gap = 46.0, 28.0
        for i in range(n - 1):
            mid = (r.s[i] + r.s[i + 1]) / 2
            if (mid % (dash + gap)) > dash:
                continue
            th = max(1, int(round(FOCAL / depth[i] * 9.0 * (1 + 0.18 * r.wob[i]))))
            cv2.line(layer, tuple(np.round(scr[i]).astype(int)), tuple(np.round(scr[i + 1]).astype(int)), 255, th, cv2.LINE_AA)
        # 출발점 동그라미, 카나리아 X 표
        for (mx, my), kind in ((PALOS, "o"), (CANARIES, "x")):
            if (kind == "o" and t > 1.0) or (kind == "x" and upto >= r.s_canary - 1):
                c, d = cam.project(to_world(mx, my))
                rad = FOCAL / d[0] * 26
                cc = tuple(np.round(c[0]).astype(int))
                th = max(1, int(FOCAL / d[0] * 7))
                if kind == "o":
                    cv2.circle(layer, cc, int(rad), 255, th, cv2.LINE_AA)
                else:
                    for sx in (-1, 1):
                        cv2.line(layer, (int(cc[0] - rad * 0.8), int(cc[1] - sx * rad * 0.8)),
                                 (int(cc[0] + rad * 0.8), int(cc[1] + sx * rad * 0.8)), 255, th, cv2.LINE_AA)
        a = (layer.astype(np.float32) / 255 * 0.86)[..., None]
        img[:] = img * (1 - a) + INK * a

    def cloud_quads(self, t: float):
        out = []
        for i, (mx, my, z, w, si, flip) in enumerate(CLOUDS):
            spr = self.clouds[si][:, ::-1] if flip else self.clouds[si]
            h = w * spr.shape[0] / spr.shape[1]
            cx = mx + WIND * t + 18 * math.sin(t * 0.5 + i)
            cy = my + 10 * math.cos(t * 0.4 + i * 1.7)
            pts = np.array([to_world(cx - w / 2, cy - h / 2, z), to_world(cx + w / 2, cy - h / 2, z),
                            to_world(cx + w / 2, cy + h / 2, z), to_world(cx - w / 2, cy + h / 2, z)])
            out.append((spr, pts, z))
        return out

    def ship_state(self, t: float):
        """배 스프라이트 → 월드 변환 (원점, 가로축, 세로축) 과 흘수선 아래로 잠긴 양."""
        p, tan = self.route.at(self.ship_dist(t))
        A = to_world(*p)
        bow = np.array([tan[0], -tan[1], 0.0])            # 해도 y 는 아래로, 월드 y 는 위로
        bow /= np.linalg.norm(bow)
        ex = -bow                                           # 스프라이트 +u = 선미 쪽 (뱃머리가 왼쪽 그림)
        north = np.array([-ex[1], ex[0], 0.0])
        if north[1] < 0:
            north = -north
        ev = math.cos(SHIP_LEAN) * np.array([0, 0, 1.0]) + math.sin(SHIP_LEAN) * north
        # 흔들흔들: 앞뒤로 기우뚱(평면 내 회전) + 위아래 출렁
        rho = math.radians(3.6) * math.sin(2 * math.pi * t / 1.9 + 0.5) + math.radians(1.3) * math.sin(2 * math.pi * t / 0.83)
        ex, ev = math.cos(rho) * ex + math.sin(rho) * ev, -math.sin(rho) * ex + math.cos(rho) * ev
        full = self.ship_vw * self.ship_k                   # 완전히 잠기려면 내려가야 하는 거리
        emerge = spring(t - SHIP_UP, 0.42, 1.25)            # 솟아올랐다가(넘침) 살짝 가라앉고 자리 잡음
        bob = 0.05 * full * math.sin(2 * math.pi * t / 1.45) * float(smoothstep(SHIP_UP, SHIP_UP + 0.9, t))
        lift = -full * 1.03 * (1 - emerge) + bob
        return A, ex, ev, lift

    def ship_world(self, t: float, u: np.ndarray, v: np.ndarray):
        A, ex, ev, lift = self.ship_state(t)
        k = self.ship_k
        uc = self.ship.shape[1] * 0.5
        return (A[None] + np.outer(u - uc, ex) * k + np.outer((self.ship_vw - v) * k + lift, ev)), ex, ev, A, lift

    def render(self, t: float) -> np.ndarray:
        cam = self.camera(t)
        ta = math.floor(t * 12) / 12 if ON_TWOS else t       # 콜라주 요소용 스톱모션 시간
        img = self.draw_map(cam)

        # 항로: 팔로스→카나리아 는 1.0~2.8s 에 그려지고, 그 뒤로는 배 꼬리까지
        r = self.route
        if t < SHIP_UP:
            upto = r.s_canary * float(smoother((t - 1.0) / 1.8))
        else:
            upto = max(r.s_canary, self.ship_dist(ta) - SHIP_LEN * 0.15)
        self.draw_route(img, cam, upto, t)

        # 그림자 (구름·배·태그) — 반 해상도에서 투영 후 흐림
        sh = np.zeros((H // 2, W // 2), np.float32)
        half = np.diag([0.5, 0.5, 1.0])
        clouds = self.cloud_quads(ta)
        for spr, pts, z in clouds:
            shadow_pts = pts.copy()
            shadow_pts[:, :2] += np.outer(pts[:, 2], LIGHT)
            shadow_pts[:, 2] = 0
            dst, _ = cam.project(shadow_pts)
            src = np.array([[0, 0], [spr.shape[1], 0], [spr.shape[1], spr.shape[0]], [0, spr.shape[0]]], np.float64)
            a = cv2.warpPerspective(spr[..., 3], half @ homography(src, dst), (W // 2, H // 2), flags=cv2.INTER_LINEAR)
            sh = np.maximum(sh, cv2.GaussianBlur(a, (0, 0), 2 + z * cam.scale * 0.012) * 0.30)

        # 배 (흘수선 아래는 해도 속에 잠겨서 안 보임)
        uu, vv = self.ship_uv
        hs, ws = self.ship.shape[:2]
        corners_uv = np.array([[0, 0], [ws, 0], [ws, hs], [0, hs]], np.float64)
        Pc, ex, ev, A, lift = self.ship_world(ta, corners_uv[:, 0], corners_uv[:, 1])
        z_map = (ex[2] * (uu - ws * 0.5) * self.ship_k + ev[2] * ((self.ship_vw - vv) * self.ship_k + lift))
        visible = np.clip(z_map / 2.0 + 0.5, 0, 1)
        ship_spr = self.ship * visible[..., None]
        ship_vis = t > SHIP_UP - 0.05
        if ship_vis:
            shadow_pts = Pc.copy()
            shadow_pts[:, :2] += np.outer(np.maximum(Pc[:, 2], 0), LIGHT)
            shadow_pts[:, 2] = 0
            dst, _ = cam.project(shadow_pts)
            a = cv2.warpPerspective(ship_spr[..., 3], half @ homography(corners_uv, dst), (W // 2, H // 2), flags=cv2.INTER_LINEAR)
            sh = np.maximum(sh, cv2.GaussianBlur(a, (0, 0), 2.2) * 0.42)
        img *= 1 - cv2.resize(sh, (W, H))[..., None] * np.array([0.95, 1.0, 1.05], np.float32)

        # 해도 위에 놓이는 날짜 태그 (3D, 해도면에 붙음)
        for tag in self.tags:
            tau = ta - tag["t0"]
            if tau < 0:
                continue
            drop = 1 - spring(tau, 0.55, 2.2)           # 위에서 툭 놓임
            spr = tag["img"]
            k = 0.95
            w, h = spr.shape[1] * k, spr.shape[0] * k
            ax, ay = tag["anchor"]
            cx, cy = ax + tag["offset"][0], ay + tag["offset"][1]
            rot = math.radians(tag["rot"] + 6 * drop)
            z = 2 + 90 * drop
            c, s = math.cos(rot), math.sin(rot)
            local = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
            mp = local @ np.array([[c, s], [-s, c]]) + [cx, cy]
            pts = np.array([to_world(x, y, z) for x, y in mp])
            src = np.array([[0, 0], [spr.shape[1], 0], [spr.shape[1], spr.shape[0]], [0, spr.shape[0]]], np.float64)
            # 지시선
            a_scr, _ = cam.project(to_world(ax, ay))
            t_scr, _ = cam.project(to_world(cx, cy + h / 2 * (1 if tag["offset"][1] < 0 else -1), z))
            line = np.zeros((H, W), np.uint8)
            cv2.line(line, tuple(a_scr[0].astype(int)), tuple(t_scr[0].astype(int)), 255, 2, cv2.LINE_AA)
            la = (line.astype(np.float32) / 255 * 0.7)[..., None]
            img[:] = img * (1 - la) + INK * la
            sp = pts.copy(); sp[:, :2] += [10, -14]; sp[:, 2] = 0
            dst_s, _ = cam.project(sp)
            sa = cv2.GaussianBlur(cv2.warpPerspective(spr[..., 3], homography(src, dst_s), (W, H)), (0, 0), 3 + 10 * drop)
            img *= 1 - (sa * 0.35)[..., None]
            dst, _ = cam.project(pts)
            over(img, warp_rgba(spr, homography(src, dst), dst))

        # 배가 솟아오르는 해도 틈(칼집) 그림자
        if ship_vis and lift > -self.ship_vw * self.ship_k * 0.98:
            half_len = SHIP_LEN * 0.36
            along = np.array([ex[0], ex[1], 0.0]) * half_len
            p0, _ = cam.project(A - along)
            p1, _ = cam.project(A + along)
            slit = np.zeros((H, W), np.uint8)
            cv2.line(slit, tuple(p0[0].astype(int)), tuple(p1[0].astype(int)), 255, max(2, int(cam.scale * 7)), cv2.LINE_AA)
            sa = cv2.GaussianBlur(slit.astype(np.float32) / 255, (0, 0), 2.5)[..., None] * 0.55
            img *= 1 - sa

        # 깊이 순서대로: 배보다 먼 구름 → 배 → 가까운 구름
        _, ship_depth = cam.project(A + ev * 150)
        items = []
        for spr, pts, z in clouds:
            _, d = cam.project(pts.mean(0, keepdims=True))
            items.append((d[0], "cloud", spr, pts))
        if ship_vis:
            items.append((ship_depth[0], "ship", ship_spr, Pc))
        for _, kind, spr, pts in sorted(items, key=lambda x: -x[0]):
            dst, depth = cam.project(pts)
            if (depth < 50).any():
                continue
            src = np.array([[0, 0], [spr.shape[1], 0], [spr.shape[1], spr.shape[0]], [0, spr.shape[0]]], np.float64)
            over(img, warp_rgba(spr, homography(src, dst), dst))

        # 틸트시프트 심도: 기울어질수록, 초점선(배)에서 멀수록 흐림
        tilt_k = cam.pitch / TILT
        if tilt_k > 0.02:
            focus_y = cam.project(A)[0][0, 1] if ship_vis else H / 2
            blur = cv2.GaussianBlur(img, (0, 0), 3.2)
            m = (smoothstep(170, 560, np.abs(self.yy - focus_y)) * tilt_k)[..., None]
            img = img * (1 - m) + blur * m

        # 화면에 붙는 타이틀 (손으로 놓는 스톱모션)
        self.draw_titles(img, ta)

        # 사진으로 찍은 느낌: 스탠드 조명·비네팅, 노출 깜빡임, 필름 그레인, 카메라 미세 흔들림
        step = int(t * 12)
        frng = np.random.default_rng(step)
        img = img * self.light * (1 + frng.uniform(-0.012, 0.012))
        img = img * (1 + 0.035 * self.grains[step % 4])
        img = np.clip(img * np.array([1.03, 1.0, 0.95], np.float32) + 0.012, 0, 1)
        jx, jy = frng.uniform(-0.8, 0.8, 2)
        m = cv2.getRotationMatrix2D((W / 2, H / 2), frng.uniform(-0.05, 0.05), 1.0)
        m[:, 2] += [jx, jy]
        img = cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return (img * 255).astype(np.uint8)

    def draw_titles(self, img: np.ndarray, t: float) -> None:
        for spr, (x, y), rot, t0 in ((self.title_1492, (96, 36), -2.5, 5.3),
                                     (self.title_strip, (110, 246), 1.4, 5.65)):
            tau = t - t0
            if tau < 0:
                continue
            drop = 1 - spring(tau, 0.5, 2.6)
            sc = 1 + 0.10 * drop
            ang = rot + 7 * drop
            h, w = spr.shape[:2]
            m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
            m[:, 2] += [x, y]
            # 그림자: 들려 있을수록 멀리·흐리게
            off = 8 + 26 * drop
            ms = m.copy(); ms[:, 2] += [off, off * 1.2]
            sa = cv2.warpAffine(spr[..., 3], ms, (W, H))
            sa = cv2.GaussianBlur(sa, (0, 0), 4 + 10 * drop) * 0.45
            img *= 1 - sa[..., None]
            p = cv2.warpAffine(spr, m, (W, H))
            img *= 1 - p[..., 3:4]
            img += p[..., :3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "output" / "columbus_1492_intro.mp4")
    ap.add_argument("--preview", type=str, default="")
    ap.add_argument("--crf", type=int, default=20)
    args = ap.parse_args()

    scene = Scene()
    total = int(DURATION * FPS)
    if args.preview:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        for v in args.preview.split(","):
            t = float(v)
            p = args.out.parent / f"preview_{t:05.2f}s.png"
            Image.fromarray(scene.render(t)).save(p)
            print("저장:", p)
        return
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg 가 필요합니다.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(args.out)], stdin=subprocess.PIPE)
    for i in range(total):
        enc.stdin.write(scene.render(i / FPS).tobytes())
    enc.stdin.close()
    if enc.wait() != 0:
        raise SystemExit("ffmpeg 인코딩 실패")
    print(f"영상 저장: {args.out}  ({total}프레임, {DURATION}초)")


if __name__ == "__main__":
    main()
