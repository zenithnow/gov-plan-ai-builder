# VOX 스타일 종이 스톱모션 - 카약 패들링 루프

Kling으로 찢어진 종이 콜라주 스타일의 카약 패들링 포즈 5장을 만들고, 파이썬으로 순서대로 반복 재생해
종이 스톱모션처럼 움직이는 영상을 만든다.

```
frame_01 → 02 → 03 → 04 → 05 → 01 → …   (24fps, 한 장당 3프레임 = 초당 8장)
```

결과물: `output/kayak_loop.mp4` (1920x1080, 8초, push-in) · `output/kayak_loop.gif` (720px, 2.5초 무한 루프)

## 빠른 실행

```bash
cd vox_paper_kayak
pip install -r requirements.txt          # ffmpeg 도 필요
python paper_motion.py                   # → output/kayak_loop.mp4

# 끊김 없는 GIF 루프 (동작 4회, 카메라 이동 없음)
python paper_motion.py --cycles 4 --zoom-start 1.05 --zoom-end 1.05 \
    --out output/gif_src.mp4 --gif output/kayak_loop.gif
```

`frames/` 의 5장은 이미 들어 있다. Kling 원본을 다시 받으려면 `python fetch_frames.py` (URL은 생성 후 24시간 유효).

## 5단계 패들링 사이클

측면 뷰, 인물은 화면 왼쪽을 보고 앉아 있다. 화면 앞쪽 블레이드 하나로 한 번의 스트로크를 보여준다.

| 파일 | 포즈 | 설명 |
|------|------|------|
| frame_01 | 캐치 | 몸을 앞으로 숙이고 블레이드를 앞쪽 물에 꽂음 |
| frame_02 | 입수 | 블레이드가 앞쪽 물에 들어가며 물보라 |
| frame_03 | 파워 | 패들을 세워 엉덩이 옆에서 당김, 물보라 |
| frame_04 | 피니시 | 블레이드가 엉덩이 뒤까지 빠짐 (Kling 기준 프레임) |
| frame_05 | 엑시트 | 블레이드가 뒤에서 들려 나오며 물방울 → 다시 01로 |

## Kling 생성 방법 (배경 일관성의 핵심)

5장을 따로 생성하면 매번 배경이 달라져 루프가 깨진다. 그래서

1. **기준 프레임 1장**을 먼저 만들고 (`kling-image-v3_0_omni`, text_to_image, 16:9, 2k)
2. 나머지 4장은 **모두 같은 기준 프레임에서** `kling-image-o1` image_to_image 로 "팔과 패들만 바꿔라" 편집

연쇄 편집(1→2→3…)이 아니라 항상 1번에서 편집해야 오차가 누적되지 않는다.

> 실제 생성에서는 프롬프트의 "left to right" 와 달리 인물이 왼쪽을 보고 나왔고, 포즈도 프롬프트 번호와
> 다르게 나왔다. 그래서 생성 결과를 보고 동작 순서대로 파일 이름을 다시 붙였다 (`kling_frames.json` 의
> `kling_order` = 아래 프롬프트 번호). 프롬프트를 재사용할 땐 결과를 보고 순서를 정하는 게 빠르다.

<details>
<summary>기준 프레임 프롬프트 (text_to_image)</summary>

```
Handcrafted layered torn-paper collage illustration, side view of a young woman paddling a sit-in kayak
on a calm lake, the kayak heading from left to right. Every single element, background and foreground,
is cut from textured paper with rough deckled TORN edges showing white paper fibers along every edge,
subtle paper grain, fine fiber line texture, and soft realistic drop shadows between stacked paper layers,
creating a 3D paper diorama depth. Minimalist modern paper-art style, like a still from a VOX explainer
animation. Pastel palette: mustard yellow, peach orange, coral pink, dusty blue, pale teal, cream and
off-white, warm chocolate brown. Character: woman with long wavy chocolate-brown hair flowing behind her,
simple minimalist face in profile with closed eyes, rosy cheek and red lips, white long-sleeve shirt,
mustard-yellow life vest. She sits in a coral-pink paper kayak holding a double-bladed paddle with
mustard-yellow blades in both hands. Pose: torso rotated forward, right arm extended forward, the
near-side blade just entering the water ahead of her near the front of the kayak, the other blade raised
high above her left shoulder. Background: layered torn-paper mountains in peach, cream and dusty pink,
a pale yellow paper sun, cream paper sky with a few torn-paper clouds; lake made of overlapping
torn-paper wave strips in dusty blue, teal and white, small torn-paper ripples around the kayak.
The whole kayak and the entire paddle fully visible and centered in the frame with generous empty space
around them, nothing cropped. Soft diffused light, high detail, no text.
```
</details>

<details>
<summary>포즈 편집 프롬프트 (image_to_image, 图片1 = 기준 프레임)</summary>

공통 앞부분:

```
Edit 图片1. Keep everything exactly the same as 图片1: same camera angle, framing and composition,
same background (torn-paper mountains, sun, clouds, lake waves), same woman, same hair, clothes and
life vest, same kayak in the exact same position and size, same torn-paper textures with white fibrous
torn edges, same colors and lighting. Change ONLY her arms and the double-bladed paddle to this pose:
```

포즈별 뒷부분:

- **2** power stroke on the near side — the near-side mustard-yellow blade is fully submerged in the water right beside her hip, pulling backward, with a small splash of torn-paper water droplets; the paddle shaft is at a steep diagonal, her lower arm bent at the elbow pulling back, her upper hand pushing forward at chin height, torso unwinding upright. The far blade is up in the air in front of her face.
- **3** stroke exit and transition — the near-side mustard-yellow blade is lifting out of the water behind her hip with torn-paper water droplets dripping from it, the paddle shaft is almost horizontal held across her chest at shoulder height, both elbows bent, the far blade lowering forward toward the water on the far side of the kayak.
- **4** catch on the far side — torso rotated forward, her left arm reaching forward, the far-side mustard-yellow blade entering the water ahead of her on the far side of the kayak, partly hidden behind the kayak hull; the near-side blade is raised high in the air above her right shoulder, right hand up near her head.
- **5** power stroke on the far side — the far-side mustard-yellow blade is in the water beside her hip on the far side of the kayak (mostly hidden behind the hull), pulling backward with a small torn-paper splash visible above the hull; the near-side blade is high in the air in front of her, swinging forward, ready for the next stroke; paddle shaft at a steep diagonal.

공통 끝: `Paddle keeps the same length and colors.`
</details>

포즈 하나가 어색하면 그 장만 같은 기준 프레임으로 다시 편집 생성해서 `frames/` 의 해당 파일만 교체하면 된다.

## paper_motion.py 가 하는 일

1. **정렬** – 2~5번을 1번에 아핀 정렬. Kling 편집본은 원본보다 1% 남짓 확대돼 나와서(배경이 1080p 기준 약 10px 밀림)
   옵티컬 플로우 + RANSAC 으로 배경만 보고 확대·이동을 되돌린다 (정렬 후 배경 오차 중앙값 1px 미만)
2. **색감 맞춤** – 편집 때 틀어진 전체 톤을 1번에 맞춤
3. **배경 고정** – 1번과 실제로 달라진 영역(팔·패들·물보라)만 남기고 나머지는 1번 배경으로 통일
4. **스톱모션 타이밍** – 한 장을 `--hold` 프레임씩 유지
5. **촬영감** – 장이 바뀔 때마다 카메라 흔들림·노출 깜빡임·종이 그레인을 새로 뽑음 (매 컷 다시 찍은 느낌)
6. **카메라** – 천천히 밀고 들어가는 push-in + 비네팅

## 자주 쓰는 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--hold` | 3 | 한 장 유지 프레임 수. 2 = 부드럽게(초당 12장), 4 = 더 뚝뚝 끊기게(초당 6장) |
| `--seconds` | 8 | 영상 길이 |
| `--order` | 파일명 순 | 예: `1,2,3,4,5` / 왕복은 `1,2,3,4,5,4,3,2` |
| `--jitter` / `--rot-jitter` | 3px / 0.2° | 컷마다 흔들림. 0이면 고정 카메라 |
| `--grain` / `--flicker` | 0.03 / 0.015 | 종이 그레인, 노출 깜빡임 |
| `--zoom-start` / `--zoom-end` | 1.04 / 1.10 | push-in. 같게 두면 카메라 이동 없음 |
| `--no-lock-bg` | - | 배경 고정 끄기 (배경도 컷마다 살짝 변하는 '보일링' 느낌) |
| `--lock-threshold` | 30 | 배경 고정 시 움직임으로 볼 색 차이. 패들 끝이 잘리면 낮추고, 배경이 깜빡이면 높임 |
| `--lock-grow` | 36 | 움직인 영역 확장(px, 입력 해상도 기준). 머리카락 뒤 그림자가 잔상으로 남으면 키움 |
| `--cycles` | - | 길이를 동작 N회로 딱 맞춤 (끊김 없는 루프용) |
| `--size` | 1920x1080 | 쇼츠용은 `1080x1920` (가운데를 잘라 씀) |
| `--save-processed DIR` | - | 정렬·배경고정 결과 5장을 저장해 확인 |
