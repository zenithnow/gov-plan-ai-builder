"""
Shared Context Memory — 에이전트 간 상태 공유 저장소
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SharedContext:
    # Agent 1 출력
    core_keywords: dict = field(default_factory=lambda: {
        "social_problem": "",
        "tech_limitation": "",
        "market_opportunity": "",
    })
    problem_draft: str = ""

    # Agent 2 출력
    solution_draft: str = ""
    bridge_validated: bool = False

    # Agent 3 출력
    market_data: dict = field(default_factory=dict)
    growth_draft: str = ""

    # 검토 결과
    eval_score: dict = field(default_factory=dict)
    recall_needed: bool = False
    recall_target: Optional[str] = None   # "problem" | "solution" | "growth"

    # Agent 4 출력
    refined_sections: dict = field(default_factory=dict)
    final_doc: str = ""
    final_eval: dict = field(default_factory=dict)

    # 원문
    user_idea: str = ""
    announcement_text: str = ""

    def to_prompt_context(self) -> str:
        """에이전트 프롬프트에 주입할 컨텍스트 문자열 생성"""
        return f"""
[Shared Context]
- 사회적 문제 키워드: {self.core_keywords['social_problem']}
- 기술적 한계 키워드: {self.core_keywords['tech_limitation']}
- 시장 기회 키워드: {self.core_keywords['market_opportunity']}
- 문제 인식 초안: {self.problem_draft[:500] if self.problem_draft else '(미작성)'}
""".strip()
