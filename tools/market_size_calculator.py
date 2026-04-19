"""
시장 규모 계산 도구 (market_size_calculator)
TAM > SAM > SOM 자동 계산 및 텍스트 테이블 반환
"""
import json
from typing import Any

TOOL_DEFINITION = {
    "name": "market_size_calculator",
    "description": (
        "타겟 시장의 TAM-SAM-SOM을 계산합니다. "
        "전체 시장 규모(TAM)를 입력하면 업종 비율 또는 기본값(100:20:5)으로 SAM/SOM을 산출합니다."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "market_name": {"type": "string", "description": "시장/산업 분야명 (예: 국내 산업안전 솔루션 시장)"},
            "tam_billion_krw": {"type": "number", "description": "TAM 추정 규모 (억 원 단위)"},
            "sam_ratio": {"type": "number", "description": "SAM/TAM 비율 0~1 (기본값 0.2)", "default": 0.2},
            "som_ratio": {"type": "number", "description": "SOM/SAM 비율 0~1 (기본값 0.25)", "default": 0.25},
            "growth_rate_pct": {"type": "number", "description": "연평균 성장률(%) (기본값 12)", "default": 12},
            "target_year": {"type": "integer", "description": "목표 연도 (기본값 3년 후)", "default": 3},
        },
        "required": ["market_name", "tam_billion_krw"],
    },
}


def run(inputs: dict[str, Any]) -> dict:
    market_name = inputs["market_name"]
    tam = inputs["tam_billion_krw"]
    sam_ratio = inputs.get("sam_ratio", 0.2)
    som_ratio = inputs.get("som_ratio", 0.25)
    growth_rate = inputs.get("growth_rate_pct", 12) / 100
    target_year = inputs.get("target_year", 3)

    sam = tam * sam_ratio
    som = sam * som_ratio

    # 성장 후 규모
    tam_future = round(tam * ((1 + growth_rate) ** target_year), 1)
    sam_future = round(sam * ((1 + growth_rate) ** target_year), 1)
    som_future = round(som * ((1 + growth_rate) ** target_year), 1)

    table = (
        f"| 구분 | 현재 규모 | {target_year}년 후 예측 | 정의 |\n"
        f"|---|---|---|---|\n"
        f"| **TAM** (전체 시장) | {tam:,.0f}억 원 | {tam_future:,.0f}억 원 | {market_name} 전체 |\n"
        f"| **SAM** (유효 시장) | {sam:,.0f}억 원 | {sam_future:,.0f}억 원 | 실제 공략 가능 세그먼트 ({int(sam_ratio*100)}%) |\n"
        f"| **SOM** (수익 시장) | {som:,.0f}억 원 | {som_future:,.0f}억 원 | 초기 목표 점유율 ({int(som_ratio*100)}%) |\n"
    )

    return {
        "market_name": market_name,
        "tam": tam, "sam": round(sam, 1), "som": round(som, 1),
        "tam_future": tam_future, "sam_future": sam_future, "som_future": som_future,
        "growth_rate_pct": inputs.get("growth_rate_pct", 12),
        "target_year": target_year,
        "markdown_table": table,
    }


def handle_tool_call(tool_input: dict) -> str:
    result = run(tool_input)
    return json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8").decode("utf-8")
