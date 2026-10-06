"""VOX 스타일 종이 스톱모션 루프 생성기.

Kling 으로 만든 N장의 포즈 이미지(예: 카약 패들링 5단계)를 순서대로 반복 재생하면서
종이 스톱모션 특유의 느낌을 더해 mp4/gif 로 내보낸다.

처리 순서
  1. 정렬   : 2~N번 프레임을 1번 프레임에 맞춰 ECC 정렬 (생성 이미지 간 미세한 구도 흔들림 제거)
  2. 배경 고정: 1번 프레임과 달라진 영역(팔·패들·물보라)만 남기고 나머지는 1번 배경으로 통일
  3. 타이밍 : 24fps 기준 한 장을 --hold 프레임씩 유지 (3 = 초당 8장, 전형적인 스톱모션 박자)
  4. 촬영감 : 장이 바뀔 때마다 미세한 카메라 흔들림·노출 깜빡임·종이 그레인을 새로 뽑음
  5. 카메라 : 전체 길이에 걸쳐 천천히 밀고 들어가는 push-in + 비네팅

사용 예
  python paper_motion.py --frames frames --out output/kayak_loop.mp4 --gif output/kayak_loop.gif
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def load_frames(frames_dir: Path, order: list[int] | None) -> tuple[list[np.ndarray], list[Path]]:
    paths = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise SystemExit(f"{frames_dir} 에 이미지가 없습니다.")
    frames = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
    h, w = frames[0].shape[:2]
    frames = [f if f.shape[:2] == (h, w) else cv2.resize(f, (w, h), interpolation=cv2.INTER_LANCZOS4) for f in frames]
    if order:
        frames = [frames[i - 1] for i in order]
        paths = [paths[i - 1] for i in order]
    return frames, paths


def align_to(ref: np.ndarray, img: np.ndarray, work_width: int = 960) -> np.ndarray:
    """img 를 ref 에 맞춰 회전+이동(유클리드) 정렬한다. 실패하면 원본을 돌려준다."""
    h, w = ref.shape[:2]
    k = work_width / w
    small = (work_width, round(h * k))
    ref_g = cv2.cvtColor(cv2.resize(ref, small, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float32)
    img_g = cv2.cvtColor(cv2.resize(img, small, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    try:
        _, warp = cv2.findTransformECC(ref_g, img_g, warp, cv2.MOTION_EUCLIDEAN, criteria, None, 5)
    except cv2.error:
        return img
    warp[:, 2] /= k
    return cv2.warpAffine(img, warp, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)


def match_color(ref: np.ndarray, img: np.ndarray, samples: int = 200_000, seed: int = 0) -> np.ndarray:
    """편집 생성 과정에서 생긴 전체 색감 차이를 채널별 gain/offset 으로 ref 에 맞춘다.

    움직인 영역은 이상치이므로 1차 피팅 후 잔차가 큰 픽셀을 빼고 다시 피팅한다.
    """
    rng = np.random.default_rng(seed)
    a = ref.reshape(-1, 3).astype(np.float32)
    b = img.reshape(-1, 3).astype(np.float32)
    idx = rng.choice(len(a), size=min(samples, len(a)), replace=False)
    a_s, b_s = a[idx], b[idx]
    out = np.empty_like(b)
    for c in range(3):
        gain, offset = np.polyfit(b_s[:, c], a_s[:, c], 1)
        keep = np.abs(b_s[:, c] * gain + offset - a_s[:, c]) < 20
        if keep.sum() > 1000:
            gain, offset = np.polyfit(b_s[keep, c], a_s[keep, c], 1)
        out[:, c] = b[:, c] * gain + offset
    return np.clip(out, 0, 255).reshape(img.shape).astype(np.uint8)


def change_mask(ref: np.ndarray, img: np.ndarray, threshold: float, grow: int, feather: int) -> np.ndarray:
    """ref 와 img 가 실제로 달라진 영역만 1에 가까운 부드러운 마스크 (H, W, 1) float32."""
    a = cv2.GaussianBlur(ref, (0, 0), 3).astype(np.float32)
    b = cv2.GaussianBlur(img, (0, 0), 3).astype(np.float32)
    diff = np.abs(a - b).max(axis=2)
    mask = (diff > threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    if grow > 0:
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    soft = mask.astype(np.float32)
    if feather > 0:
        soft = cv2.GaussianBlur(soft, (0, 0), feather)
    return np.clip(soft, 0, 1)[..., None]


def paper_grain(h: int, w: int, rng: np.random.Generator, strength: float) -> np.ndarray:
    """여러 스케일의 노이즈를 섞은 종이 그레인 (곱셈용, 평균 1.0)."""
    grain = np.zeros((h, w), np.float32)
    for scale, weight in ((2, 0.45), (6, 0.35), (24, 0.2)):
        small = rng.standard_normal((max(1, h // scale), max(1, w // scale))).astype(np.float32)
        grain += weight * cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    grain /= grain.std() + 1e-6
    return (1.0 + strength * grain)[..., None]


def vignette(h: int, w: int, amount: float) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2) / math.sqrt(2)
    return (1.0 - amount * r**2)[..., None]


def place(img: np.ndarray, out_w: int, out_h: int, zoom: float, angle: float, tx: float, ty: float) -> np.ndarray:
    """입력 이미지를 출력 화면을 꽉 채우도록(cover) 배치하면서 줌·회전·이동을 적용한다."""
    h, w = img.shape[:2]
    scale = max(out_w / w, out_h / h) * zoom
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[0, 2] += out_w / 2 - w / 2 + tx
    m[1, 2] += out_h / 2 - h / 2 + ty
    return cv2.warpAffine(img, m, (out_w, out_h), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)


def open_encoder(out: Path, w: int, h: int, fps: int, crf: int) -> subprocess.Popen:
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg 가 필요합니다. (macOS: brew install ffmpeg / Windows: winget install ffmpeg)")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(out),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def export_gif(mp4: Path, gif: Path, fps: float, width: int) -> None:
    gif.parent.mkdir(parents=True, exist_ok=True)
    vf = f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf", vf, "-loop", "0", str(gif)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="VOX 스타일 종이 스톱모션 루프 생성기")
    ap.add_argument("--frames", type=Path, default=Path(__file__).parent / "frames", help="포즈 이미지 폴더 (파일명 순서대로 재생)")
    ap.add_argument("--order", type=str, default="", help="재생 순서 직접 지정, 예: 1,2,3,4,5")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "output" / "kayak_loop.mp4")
    ap.add_argument("--gif", type=Path, default=None, help="gif 미리보기도 저장할 경로")
    ap.add_argument("--gif-width", type=int, default=720)
    ap.add_argument("--size", type=str, default="1920x1080", help="출력 해상도 WxH")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--hold", type=int, default=3, help="한 장을 유지하는 프레임 수 (24fps 에서 3 = 초당 8장)")
    ap.add_argument("--seconds", type=float, default=8.0, help="영상 길이(초)")
    ap.add_argument("--no-align", action="store_true", help="프레임 정렬 끄기")
    ap.add_argument("--no-color-match", action="store_true", help="색감 맞춤 끄기")
    ap.add_argument("--no-lock-bg", action="store_true", help="배경 고정 끄기 (각 이미지를 그대로 사용)")
    ap.add_argument("--lock-threshold", type=float, default=30.0, help="배경 고정 시 '움직인 영역'으로 볼 색 차이 (0~255)")
    ap.add_argument("--lock-grow", type=int, default=18, help="움직인 영역 마스크 확장(px)")
    ap.add_argument("--lock-feather", type=int, default=10, help="마스크 경계 블러(px)")
    ap.add_argument("--jitter", type=float, default=3.0, help="장마다 카메라 흔들림(px, 1080p 기준)")
    ap.add_argument("--rot-jitter", type=float, default=0.2, help="장마다 회전 흔들림(도)")
    ap.add_argument("--flicker", type=float, default=0.015, help="장마다 노출 깜빡임 비율")
    ap.add_argument("--grain", type=float, default=0.03, help="종이 그레인 세기")
    ap.add_argument("--vignette", type=float, default=0.18)
    ap.add_argument("--zoom-start", type=float, default=1.04, help="push-in 시작 배율 (흔들림 여백 포함, 1.02 이상 권장)")
    ap.add_argument("--zoom-end", type=float, default=1.10, help="push-in 끝 배율")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--save-processed", type=Path, default=None, help="정렬·배경고정을 마친 포즈 이미지를 저장할 폴더")
    args = ap.parse_args()

    out_w, out_h = (int(v) for v in args.size.lower().split("x"))
    order = [int(v) for v in args.order.split(",") if v.strip()] or None
    frames, paths = load_frames(args.frames, order)
    print(f"포즈 {len(frames)}장: {', '.join(p.name for p in paths)}")

    base = frames[0]
    drawings = [base]
    for i, f in enumerate(frames[1:], start=2):
        g = f if args.no_align else align_to(base, f)
        if not args.no_color_match:
            g = match_color(base, g, seed=args.seed)
        if not args.no_lock_bg:
            m = change_mask(base, g, args.lock_threshold, args.lock_grow, args.lock_feather)
            g = (base * (1 - m) + g * m).astype(np.uint8)
            print(f"  {i}번: 움직인 영역 {m.mean() * 100:.1f}%")
        drawings.append(g)

    if args.save_processed:
        args.save_processed.mkdir(parents=True, exist_ok=True)
        for i, d in enumerate(drawings, start=1):
            Image.fromarray(d).save(args.save_processed / f"processed_{i:02d}.png")

    rng = np.random.default_rng(args.seed)
    grains = [paper_grain(out_h, out_w, rng, args.grain) for _ in range(4)]
    vig = vignette(out_h, out_w, args.vignette)
    px = out_h / 1080  # 흔들림 크기를 해상도에 맞춰 보정

    total = round(args.seconds * args.fps)
    enc = open_encoder(args.out, out_w, out_h, args.fps, args.crf)
    shot = None
    for t in range(total):
        step = t // args.hold
        if t % args.hold == 0:
            # 새 '컷'을 찍을 때마다 카메라·조명·종이 상태를 새로 뽑는다
            shot = {
                "img": drawings[step % len(drawings)],
                "dx": rng.uniform(-1, 1) * args.jitter * px,
                "dy": rng.uniform(-1, 1) * args.jitter * px,
                "rot": rng.uniform(-1, 1) * args.rot_jitter,
                "gain": 1 + rng.uniform(-1, 1) * args.flicker,
                "grain": grains[rng.integers(len(grains))],
            }
        p = t / max(1, total - 1)
        ease = p * p * (3 - 2 * p)
        zoom = args.zoom_start + (args.zoom_end - args.zoom_start) * ease
        frame = place(shot["img"], out_w, out_h, zoom, shot["rot"], shot["dx"], shot["dy"]).astype(np.float32)
        frame *= shot["grain"] * vig * shot["gain"]
        enc.stdin.write(np.clip(frame, 0, 255).astype(np.uint8).tobytes())
    enc.stdin.close()
    if enc.wait() != 0:
        raise SystemExit("ffmpeg 인코딩 실패")
    print(f"영상 저장: {args.out}  ({total}프레임, {args.seconds:.1f}초, 초당 {args.fps / args.hold:.1f}장)")

    if args.gif:
        export_gif(args.out, args.gif, args.fps / args.hold, args.gif_width)
        print(f"GIF 저장: {args.gif}")


if __name__ == "__main__":
    main()
