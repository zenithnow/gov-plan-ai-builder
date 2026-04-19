"""
Agent 1: 문제 인식 (The Problem Definer)
— 사회적 문제 / 기술적 한계 / 시장 기회 3대 키워드 추출 + 초안 작성
"""
import json
import re
import anthropic
from shared_context import SharedContext

SYSTEM_PROMPT = """너는 정부지원사업 심사위원의 시각을 가진 문제 분석가이다.

사용자 입력에서 다음 3가지 핵심 키워드를 반드시 추출하여 JSON 블록으로 먼저 출력하라:
- social_problem: 사회적 문제 (2~5단어 명사구)
- tech_limitation: 기술적 한계 (2~5단어 명사구)
- market_opportunity: 시장 기회 (2~5단어 명사구)

출력 형식:
```json
{"social_problem": "...", "tech_limitation": "...", "market_opportunity": "..."}
```

이후 아래 구조로 개조식 초안을 작성하라. 통계 데이터가 필요한 자리는 반드시 [통계데이터_확인필요] 태그를 삽입하라.

## 1. 개요 및 문제인식
### 1-1. 개발 배경 및 필요성
- **[social_problem 키워드]** 중심의 현 시장 한계
  - 현재 OO 산업 현황 [통계데이터_확인필요]
  - 이로 인한 실질적 불편 및 피해
- **[tech_limitation 키워드]** 로 인한 기존 솔루션 한계
  - 기존 기술/서비스의 구체적 문제점
- **[market_opportunity 키워드]** 기반 사업 타당성
  - 해결 시 기대 효과 [통계데이터_확인필요]

모든 문장은 "— ~함", "— ~임" 개조식 종결어미를 사용하라. 불필요한 미사여구 금지."""


def extract_keywords_from_response(text: str) -> dict:
    """응답 텍스트에서 JSON 키워드 블록 파싱"""
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # fallback: 중괄호 직접 탐색
    match = re.search(r'\{"social_problem".*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def run(client: anthropic.Anthropic, context: SharedContext) -> SharedContext:
    """Agent 1 실행 → SharedContext 업데이트 후 반환"""
    user_message = f"""
[사용자 비즈니스 아이디어]
{context.user_idea}

[공고문 (참고)]
{context.announcement_text if context.announcement_text else '(공고문 미입력)'}
""".strip()

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    full_text = response.content[0].text

    # 키워드 추출
    keywords = extract_keywords_from_response(full_text)
    if keywords:
        context.core_keywords.update(keywords)

    # JSON 블록 제거한 나머지가 초안
    draft = re.sub(r"```json.*?```", "", full_text, flags=re.DOTALL).strip()
    context.problem_draft = draft

    return context
