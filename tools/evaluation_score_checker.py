"""
평가 지표 매칭 검기 — 공고문 배점 기준 vs 초안 키워드 밀도 비교
"""
import json
import re
from typing import Any


# Tool Use 정의 (Anthropic API에 전달)
TOOL_DEFINITION = {
    "name": "evaluation_score_checker",
    "description": (
        "공고문 배점 기준과 생성된 초안을 비교하여 항목별 충족도를 분석합니다. "
        "누락 항목이 있으면 recall_needed=true와 함께 해당 에이전트명을 반환합니다."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "announcement_criteria": {
                "type": "array",
                "description": "공고문에서 추출된 평가 항목 리스트",
                "items": {
                    "type": "object",
                    "properties": {
                        "item": {"type": "string", "description": "평가 항목명 (예: 기술성)"},
                        "score": {"type": "integer", "description": "배점"},
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "해당 항목의 핵심 키워드 목록",
                        },
                    },
                    "required": ["item", "score", "keywords"],
                },
            },
            "draft_text": {
                "type": "string",
                "description": "검토 대상 초안 전문",
            },
        },
        "required": ["announcement_criteria", "draft_text"],
    },
}


def run(inputs: dict[str, Any]) -> dict:
    """
    배점 항목별 키워드 밀도를 계산하여 점수표와 Recall 신호를 반환.
    """
    # Claude가 키 이름을 다르게 전달하는 경우 폴백 처리
    criteria: list[dict] = (
        inputs.get("announcement_criteria")
        or inputs.get("criteria")
        or inputs.get("evaluation_criteria")
        or []
    )
    draft_raw = (
        inputs.get("draft_text")
        or inputs.get("draft")
        or inputs.get("text")
        or ""
    )
    draft: str = draft_raw.lower()

    results = []
    recall_items = []
    total_score = 0
    achieved_score = 0

    for criterion in criteria:
        item_name = criterion["item"]
        max_score = criterion["score"]
        keywords = [kw.lower() for kw in criterion["keywords"]]

        # 키워드 출현 횟수 기반 밀도 계산
        hit_count = sum(1 for kw in keywords if re.search(re.escape(kw), draft))
        density = hit_count / len(keywords) if keywords else 0

        # 밀도에 따른 점수 추정 (단순 비례)
        estimated = round(max_score * density)
        coverage_pct = round(density * 100)

        results.append({
            "item": item_name,
            "max_score": max_score,
            "estimated_score": estimated,
            "coverage_pct": coverage_pct,
            "missing_keywords": [kw for kw in criterion["keywords"] if kw.lower() not in draft],
        })

        total_score += max_score
        achieved_score += estimated

        if density < 0.5:
            recall_items.append(item_name)

    return {
        "score_table": results,
        "total_max": total_score,
        "total_estimated": achieved_score,
        "overall_pct": round(achieved_score / total_score * 100) if total_score else 0,
        "recall_needed": len(recall_items) > 0,
        "recall_items": recall_items,
    }


def handle_tool_call(tool_input: dict) -> str:
    """Claude Tool Use 응답 처리 진입점 → JSON 문자열 반환"""
    result = run(tool_input)
    return json.dumps(result, ensure_ascii=False, indent=2)
