"""
Agent 3: 성장 전략 (The Growth Strategist)
— market_size_calculator 툴 호출 → TAM-SAM-SOM + BM + 로드맵 작성
"""
import json
import anthropic
from shared_context import SharedContext
from tools import ALL_TOOLS, TOOL_HANDLERS

SYSTEM_PROMPT = """너는 정부지원사업 사업계획서 전문 경영 전략 컨설턴트이다.

[필수 수행 순서]
1. market_size_calculator 툴을 반드시 호출하여 TAM-SAM-SOM 수치를 계산하라.
2. 계산된 수치를 기반으로 아래 구조의 성장 전략 섹션을 작성하라.

[출력 형식]
## 3. 시장 분석 및 성장 전략

### 3-1. 시장 규모 (TAM-SAM-SOM)
{market_size_calculator 툴 결과 테이블 그대로 삽입}
- TAM 산출 근거: [근거 기술]
- 연평균 성장률(CAGR): OO% [수치데이터_확인필요]

### 3-2. 비즈니스 모델 (BM)
- **수익 구조:** (SaaS 구독 / 라이선스 / 성과 기반 등 명시)
  - 1차 수익원: 설명
  - 2차 수익원: 설명
- **단가 및 목표 매출:**
  - 초기 단가: OO만 원/월
  - 1년차 목표 고객 수: OO개사 → 매출 OO억 원
  - 3년차 목표: OO억 원

### 3-3. 시장 진입 전략 (GTM)
- **1단계 (0~6개월):** 파일럿 고객 확보 전략
- **2단계 (6~18개월):** 채널 확대 및 파트너십
- **3단계 (18개월~):** 스케일업 및 해외 진출

[문체 규칙]
- 개조식 종결어미: "— ~함", "— ~임"
- 모든 수치는 툴 결과 또는 [수치데이터_확인필요] 태그 사용
- 추상 표현 금지"""


def run(client: anthropic.Anthropic, context: SharedContext) -> SharedContext:
    """Agent 3 실행 — market_size_calculator Tool Use 포함"""

    user_message = f"""
{context.to_prompt_context()}

[해결 방안 요약]
{context.solution_draft[:600] if context.solution_draft else '(미작성)'}

[사용자 원본 아이디어]
{context.user_idea}

market_size_calculator 툴을 호출하여 시장 규모를 계산한 후 성장 전략 섹션을 작성하라.
아이디어의 업종과 규모에 맞는 TAM 수치를 추정하여 툴에 입력하라.
""".strip()

    messages = [{"role": "user", "content": user_message}]

    # 1차 호출 — 툴 사용 유도
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3000,
        system=SYSTEM_PROMPT,
        tools=ALL_TOOLS,
        messages=messages,
    )

    market_data = {}
    final_text = ""

    # Tool Use 처리 루프
    while response.stop_reason == "tool_use":
        tool_results = []
        for block in response.content:
            if block.type == "tool_use" and block.name == "market_size_calculator":
                raw = TOOL_HANDLERS["market_size_calculator"](block.input)
                market_data = json.loads(raw)
                # content는 영문/수치만 포함한 요약 전달 (한글 인코딩 오류 방지)
                summary = (
                    f"TAM: {market_data.get('tam',0)} bil KRW, "
                    f"SAM: {market_data.get('sam',0)} bil KRW, "
                    f"SOM: {market_data.get('som',0)} bil KRW, "
                    f"CAGR: {market_data.get('growth_rate_pct',12)}%, "
                    f"Target year: {market_data.get('target_year',3)}yr later "
                    f"TAM={market_data.get('tam_future',0)}, SAM={market_data.get('sam_future',0)}, SOM={market_data.get('som_future',0)} bil KRW"
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": summary,
                })

        if not tool_results:
            break

        # 툴 결과를 포함하여 재호출
        messages = messages + [
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": tool_results},
        ]
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=3000,
            system=SYSTEM_PROMPT,
            tools=ALL_TOOLS,
            messages=messages,
        )

    # 최종 텍스트 추출
    for block in response.content:
        if hasattr(block, "text"):
            final_text += block.text

    context.market_data = market_data
    context.growth_draft = final_text

    return context
