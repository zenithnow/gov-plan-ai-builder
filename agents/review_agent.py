"""
Agent 4: 최종 검토 (The Final Reviewer)
— style_refiner 3회 호출 → evaluation_score_checker 재실행 → 최종 통합 문서 생성
"""
import json
import anthropic
from shared_context import SharedContext
from tools import ALL_TOOLS, TOOL_HANDLERS

SYSTEM_PROMPT = """너는 정부지원사업 사업계획서 최종 편집자이다.

[필수 수행 순서]
1. style_refiner 툴을 3번 호출하라 (section: problem, solution, growth 각각).
2. evaluation_score_checker 툴을 1번 호출하여 최종 점수를 산출하라.
3. 교정된 3개 섹션을 하나의 완성된 마크다운 문서로 통합하라.

[통합 문서 구조]
# 사업계획서

## 1. 개요 및 문제인식
{problem 교정본}

## 2. 해결방안
{solution 교정본}

## 3. 시장 분석 및 성장 전략
{growth 교정본}

[주의사항]
- [통계데이터_확인필요], [수치데이터_확인필요] 태그는 그대로 유지하라
- 각 섹션 제목(##)은 변경하지 말라
- 개조식 종결어미를 최종 점검하고 서술형이 남아있으면 수정하라"""


def run(client: anthropic.Anthropic, context: SharedContext) -> SharedContext:
    """Agent 4 실행 — style_refiner × 3 + evaluation_score_checker Tool Use"""

    user_message = f"""
아래 3개 섹션 초안을 교정하고 최종 통합 문서를 작성하라.

[문제인식 초안]
{context.problem_draft}

[해결방안 초안]
{context.solution_draft}

[성장전략 초안]
{context.growth_draft}

[평가기준 (evaluation_score_checker용)]
{json.dumps(list(context.eval_score.get("score_table", [])) or
    [{"item": "기술성", "score": 40, "keywords": ["기술", "AI", "알고리즘", "특허"]},
     {"item": "사업성", "score": 30, "keywords": ["시장", "수익", "BM", "고객"]},
     {"item": "팀역량", "score": 20, "keywords": ["팀", "대표", "경력", "전문가"]},
     {"item": "사회적가치", "score": 10, "keywords": ["사회", "환경", "일자리", "공익"]}],
    ensure_ascii=False)}
""".strip()

    messages = [{"role": "user", "content": user_message}]
    refined = {"problem": context.problem_draft,
               "solution": context.solution_draft,
               "growth": context.growth_draft}
    final_eval = {}
    final_text = ""

    # Tool Use 루프
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=6000,
        system=SYSTEM_PROMPT,
        tools=ALL_TOOLS,
        messages=messages,
    )

    while response.stop_reason == "tool_use":
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            if block.name == "style_refiner":
                raw = TOOL_HANDLERS["style_refiner"](block.input)
                result = json.loads(raw)
                section = result.get("section", "problem")
                refined[section] = result.get("refined_text", refined.get(section, ""))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"Refined {section}: {result['lines_changed']}/{result['total_lines']} lines changed.",
                })

            elif block.name == "evaluation_score_checker":
                raw = TOOL_HANDLERS["evaluation_score_checker"](block.input)
                final_eval = json.loads(raw)
                summary = (
                    f"Score: {final_eval.get('total_estimated',0)}/{final_eval.get('total_max',0)} "
                    f"({final_eval.get('overall_pct',0)}%). "
                    f"Recall needed: {final_eval.get('recall_needed', False)}"
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": summary,
                })

        if not tool_results:
            break

        messages = messages + [
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": tool_results},
        ]
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=6000,
            system=SYSTEM_PROMPT,
            tools=ALL_TOOLS,
            messages=messages,
        )

    for block in response.content:
        if hasattr(block, "text"):
            final_text += block.text

    context.refined_sections = refined
    context.final_doc = final_text
    context.final_eval = final_eval

    return context
