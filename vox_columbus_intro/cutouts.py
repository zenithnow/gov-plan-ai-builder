"""Kling 이미지 → 손으로 오린 종이 컷아웃 (RGBA).

크로마 그린 위에 생성한 요소를 키잉하고, 가위로 대충 오린 듯한 크림색 종이 테두리를 붙인다.
결과는 assets/cut_*.png 로 캐시한다.

  python cutouts.py    # 캐시 다시 만들기
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).parent
ASSETS = HERE / "assets"
PAPER = np.array([238, 228, 205], np.float32)  # 책 종이 크림색


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, np.float32) - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def key_green(raw: np.ndarray) -> np.ndarray:
    """초록 배경 알파. Lab a*(초록↔빨강) 기준.

    Kling 이 이미지마다 다른 초록(인쇄물 같은 탁한 초록 a*≈-15 ~ 형광 초록 a*≈-50)을 깔아서,
    테두리 픽셀로 배경 a* 를 추정한 뒤 그 값과 0 사이에서 문턱을 잡는다.
    """
    lab = cv2.cvtColor(raw, cv2.COLOR_RGB2LAB).astype(np.float32)
    a_star = lab[..., 1] - 128
    border = np.concatenate([a_star[:8].ravel(), a_star[:, :8].ravel(), a_star[:, -8:].ravel()])
    bg = float(np.percentile(border, 20))
    return smoothstep(bg * 0.72, bg * 0.38, a_star)


def despill(raw: np.ndarray) -> np.ndarray:
    rgb = raw.astype(np.float32)
    g_lim = np.maximum(rgb[..., 0], rgb[..., 2]) * 1.02
    rgb[..., 1] = np.minimum(rgb[..., 1], g_lim)
    return rgb


def components(alpha: np.ndarray, min_frac: float) -> list[np.ndarray]:
    solid = (alpha > 0.5).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(solid, 8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_frac * alpha.size:
            out.append(labels == i)
    return out


def paper_texture(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    g = np.zeros((h, w), np.float32)
    for scale, wt in ((2, 0.5), (8, 0.3), (40, 0.2)):
        g += wt * cv2.resize(rng.standard_normal((max(1, h // scale), max(1, w // scale))).astype(np.float32), (w, h))
    g /= g.std() + 1e-6
    return PAPER * (1 + 0.025 * g)[..., None]


def hand_cut(raw: np.ndarray, body: np.ndarray, margin: int, bridge: int, rng: np.random.Generator) -> np.ndarray:
    """body(오릴 대상 마스크) 주위를 margin 만큼 띄워 가위로 오린 다각형 윤곽을 만든다.

    bridge: 이 크기보다 좁은 틈(밧줄 사이 등)은 종이째로 남긴다 — 가위로 못 파내는 부분.
    반환: premultiplied 가 아닌 RGBA uint8.
    """
    alpha = key_green(raw)
    pad = margin * 3 + bridge
    raw = cv2.copyMakeBorder(raw, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    alpha = cv2.copyMakeBorder(alpha * body, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    mask = (alpha > 0.4).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge | 1, bridge | 1))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(closed)
    cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
    # 여백: 위치마다 굵기가 조금씩 다른 테두리
    dist = cv2.distanceTransform(1 - filled, cv2.DIST_L2, 5)
    h, w = filled.shape
    wobble = cv2.resize(rng.random((h // 60 + 2, w // 60 + 2)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    grown = (dist <= margin * (0.6 + 0.8 * wobble)).astype(np.uint8)
    # 가위질: 윤곽을 짧은 직선 구간들로 단순화
    contours, _ = cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cut = np.zeros_like(grown)
    for c in contours:
        poly = cv2.approxPolyDP(c, max(2.0, margin * 0.35), True)
        cv2.fillPoly(cut, [poly], 1)
    cut = cv2.GaussianBlur(cut.astype(np.float32), (0, 0), 0.7)
    paper = paper_texture(h, w, rng)
    rgb = despill(raw)
    a = alpha[..., None]
    out = rgb * a + paper * (1 - a)
    ys, xs = np.nonzero(cut > 0.02)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    rgba = np.dstack([np.clip(out, 0, 255), cut * 255])[y0:y1, x0:x1]
    return rgba.astype(np.uint8)


def build() -> None:
    rng = np.random.default_rng(1492)

    # 배: 물결(바다) 줄을 흘수선 아래로 잘라내고, 배와 이어진 덩어리만 남김
    raw = np.asarray(Image.open(ASSETS / "ship_raw.png").convert("RGB"))
    h = raw.shape[0]
    cut_row = int(h * SHIP_CUT_ROW)
    a = key_green(raw)
    a[cut_row:] = 0
    body = max(components(a, 0.01), key=lambda m: m.sum()).astype(np.float32)
    body = cv2.dilate(body, np.ones((3, 3), np.uint8))
    ship = hand_cut(raw[:cut_row + 40], body[:cut_row + 40], margin=16, bridge=45, rng=rng)
    Image.fromarray(ship).save(ASSETS / "cut_ship.png")

    # 구름 4개
    raw = np.asarray(Image.open(ASSETS / "clouds_raw.png").convert("RGB"))
    a = key_green(raw)
    for i, m in enumerate(sorted(components(a, 0.01), key=lambda m: (np.nonzero(m)[0].mean() > raw.shape[0] / 2, np.nonzero(m)[1].mean()))):
        m = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
        Image.fromarray(hand_cut(raw, m, margin=12, bridge=25, rng=rng)).save(ASSETS / f"cut_cloud_{i}.png")

    # 찢어진 종이띠 3개: 이미 찢긴 가장자리가 있으니 테두리 없이 키잉만
    raw = np.asarray(Image.open(ASSETS / "strips_raw.png").convert("RGB"))
    # 종이띠 아래 깔린 어두운 그림자는 초록 기가 약해 남으므로 밝기로 한 번 더 거른다
    lum = cv2.cvtColor(raw, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    a = key_green(raw) * smoothstep(105, 150, lum)
    for i, m in enumerate(sorted(components(a, 0.01), key=lambda m: np.nonzero(m)[0].mean())):
        alpha = cv2.GaussianBlur((a * m).astype(np.float32), (0, 0), 0.6)
        ys, xs = np.nonzero(alpha > 0.02)
        rgba = np.dstack([despill(raw), alpha * 255])[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        Image.fromarray(np.clip(rgba, 0, 255).astype(np.uint8)).save(ASSETS / f"cut_strip_{i}.png")


SHIP_CUT_ROW = 0.875  # 이 높이(비율) 아래는 물결 그림 → 잘라내고, 이 선을 흘수선으로 씀


if __name__ == "__main__":
    build()
    print("saved:", sorted(p.name for p in ASSETS.glob("cut_*.png")))
