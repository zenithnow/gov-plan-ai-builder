"""
Agent 2: 해결 방안 (The Solution Architect)
— Bridge Logic: 문제 키워드와 1:1 대응하는 솔루션 강제
"""
import re
import anthropic
from shared_context import SharedContext

SYSTEM_PROMPT = """너는 정부지원사업 사업계획서 전문 솔루션 설계자이다.

[Bridge Logic 규칙 — 반드시 준수]
Shared Context에 저장된 3가지 핵심 키워드를 각 소제목으로 사용하여 해결 방안을 작성하라.
- 사회적 문제 키워드 → "2-1. [키워드] 해결을 위한 핵심 기술"
- 기술적 한계 키워드 → "2-2. [키워드] 극복을 위한 구현 방안"
- 시장 기회 키워드 → "2-3. [키워드] 기반 차별화 전략"

각 소제목 아래에는 반드시:
1. 기존 방식의 한계 (1~2줄)
2. 제안 솔루션의 구체적 구현 방법 (수치 포함)
3. 기존 대비 개선 효과 (예: "기존 대비 OO% 향상")

[문체 규칙]
- 개조식 종결어미 사용: "— ~함", "— ~임", "— ~함으로써"
- 구현 불가 추상 표현 금지 (예: "혁신적인", "획기적인")
- 통계 수치가 없는 자리는 반드시 [수치데이터_확인필요] 태그 삽입

[출력 형식]
## 2. 해결방안 (Bridge)
### 2-1. [사회적 문제 키워드] 해결을 위한 핵심 기술
...
### 2-2. [기술적 한계 키워드] 극복을 위한 구현 방안
...
### 2-3. [시장 기회 키워드] 기반 차별화 전략
..."""


def _validate_bridge(draft: str, keywords: dict) -> list[str]:
    """Bridge Logic 검증 — 키워드가 제목에 포함됐는지 확인, 누락 목록 반환"""
    missing = []
    for key, kw in keywords.items():
        if kw and kw.lower() not in draft.lower():
            missing.append(f"{key}: '{kw}'")
    return missing


def run(client: anthropic.Anthropic, context: SharedContext, max_retries: int = 2) -> SharedContext:
    """Agent 2 실행 — Bridge Logic 미충족 시 자동 재호출"""
    kw = context.core_keywords
    user_message = f"""
{context.to_prompt_context()}

[사용자 원본 아이디어]
{context.user_idea}

위 Shared Context의 3가지 키워드를 소제목으로 반드시 활용하여 해결 방안 섹션을 작성하라.
""".strip()

    for attempt in range(max_retries + 1):
        extra = ""
        if attempt > 0:
            extra = f"\n\n[재호출 {attempt}회차] 이전 초안에서 다음 키워드가 소제목에 누락됨: {', '.join(missing)}. 반드시 소제목에 포함하라."

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message + extra}],
        )
        draft = response.content[0].text

        missing = _validate_bridge(draft, kw)
        if not missing:
            context.solution_draft = draft
            context.bridge_validated = True
            break
        # 마지막 시도면 그냥 저장
        if attempt == max_retries:
            context.solution_draft = draft
            context.bridge_validated = False

    return context
