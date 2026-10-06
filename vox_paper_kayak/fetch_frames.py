"""kling_frames.json 에 적힌 Kling 결과 이미지를 frames/ 폴더로 내려받는다."""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent


def main() -> None:
    manifest = json.loads((HERE / "kling_frames.json").read_text(encoding="utf-8"))
    out_dir = HERE / "frames"
    out_dir.mkdir(exist_ok=True)
    failed = 0
    for item in manifest["frames"]:
        dest = out_dir / item["file"]
        req = urllib.request.Request(item["url"], headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                dest.write_bytes(resp.read())
            print(f"OK   {dest.name}  ({dest.stat().st_size / 1e6:.1f} MB)  {item['pose']}")
        except Exception as e:  # 만료(403/404) 등
            failed += 1
            print(f"FAIL {dest.name}: {e}")
    if failed:
        sys.exit(f"{failed}개 실패 - URL이 만료됐다면 Kling 웹의 내 작업 목록에서 직접 받아 frames/ 에 넣으세요.")


if __name__ == "__main__":
    main()
