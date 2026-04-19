"""
개조식 문체 교정기 (style_refiner)
서술형 문장 → "— ~함/임" 개조식 변환 + 불필요한 미사여구 제거
"""
import json
import re
from typing import Any

TOOL_DEFINITION = {
    "name": "style_refiner",
    "description": (
        "사업계획서 초안의 문체를 정부지원사업 심사 기준에 맞는 개조식으로 교정합니다. "
        "서술형 문장을 '— ~함/임' 형태로 변환하고, 추상적 미사여구를 제거합니다."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "교정할 원문 텍스트",
            },
            "section": {
                "type": "string",
                "description": "섹션 구분 (problem / solution / growth)",
                "enum": ["problem", "solution", "growth"],
            },
        },
        "required": ["text", "section"],
    },
}

# 제거 대상 미사여구 패턴
VAGUE_PATTERNS = [
    r"혁신적인\s*",
    r"획기적인\s*",
    r"차세대\s*",
    r"세계 최고\s*",
    r"최첨단\s*",
    r"탁월한\s*",
    r"놀라운\s*",
    r"압도적인\s*",
]

# 서술형 종결어미 → 개조식 변환 패턴
ENDING_PATTERNS = [
    (r"합니다\s*\.?\s*$",   "함"),
    (r"합니다\.\s*$",       "함"),
    (r"됩니다\s*\.?\s*$",   "됨"),
    (r"있습니다\s*\.?\s*$", "있음"),
    (r"입니다\s*\.?\s*$",   "임"),
    (r"입니다\.\s*$",       "임"),
    (r"했습니다\s*\.?\s*$", "했음"),
    (r"한다\s*\.?\s*$",     "함"),
    (r"된다\s*\.?\s*$",     "됨"),
    (r"있다\s*\.?\s*$",     "있음"),
    (r"이다\s*\.?\s*$",     "임"),
]


def _refine_line(line: str) -> str:
    """단일 줄 교정"""
    # 미사여구 제거
    for pat in VAGUE_PATTERNS:
        line = re.sub(pat, "", line)

    # 불릿 포인트 라인만 종결어미 변환
    stripped = line.lstrip()
    if stripped.startswith(("-", "•", "*", "·")):
        for pat, replacement in ENDING_PATTERNS:
            line, count = re.subn(pat, replacement, line, flags=re.MULTILINE)
            if count:
                break

    return line.rstrip()


def run(inputs: dict[str, Any]) -> dict:
    text: str = inputs.get("text", "")
    section: str = inputs.get("section", "problem")

    lines = text.split("\n")
    refined_lines = [_refine_line(line) for line in lines]
    refined_text = "\n".join(refined_lines)

    # 변경된 줄 수 계산
    changed = sum(1 for a, b in zip(lines, refined_lines) if a != b)

    return {
        "refined_text": refined_text,
        "section": section,
        "lines_changed": changed,
        "total_lines": len(lines),
    }


def handle_tool_call(tool_input: dict) -> str:
    return json.dumps(run(tool_input), ensure_ascii=False, indent=2)
