"""
정부지원사업 AI 빌더 — Streamlit 웹앱
Step 1: Agent 1 (문제 인식) + Evaluation Score Checker
Step 2: Agent 2 (해결 방안) + Bridge Logic 검증
Step 3: Agent 3 (성장 전략) + market_size_calculator Tool Use
"""
import os
import sys
import json
import streamlit as st
import anthropic
from dotenv import load_dotenv

# Windows 환경에서 stdout/stderr 인코딩을 UTF-8로 강제 설정
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

from shared_context import SharedContext
from agents import problem_agent, solution_agent, growth_agent, review_agent


def get_api_key() -> str:
    """로컬: .env / 클라우드: st.secrets 순서로 API Key 로드"""
    # st.secrets (Streamlit Cloud)
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    # 환경 변수 (.env 또는 시스템)
    return os.getenv("ANTHROPIC_API_KEY", "")
from tools import ALL_TOOLS, TOOL_HANDLERS

# ── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="정부지원사업 AI 빌더",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* 카드 */
.card {
    background: #ffffff;
    color: #1f2937;
    border-radius: 12px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.10);
    border-left: 4px solid #1a56db;
}
.card p, .card span, .card div, .card strong, .card h4 { color: #1f2937; }
.card-warn    { border-left-color: #d97706; }
.card-success { border-left-color: #059669; }
.card-error   { border-left-color: #dc2626; }
.card-gray    { border-left-color: #9ca3af; background: #f9fafb; }

/* 배지 */
.step-badge {
    display: inline-block;
    background: #1a56db;
    color: #ffffff;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.76rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.badge-done    { background: #059669; }
.badge-active  { background: #1a56db; }
.badge-pending { background: #9ca3af; }

/* 점수 게이지 */
.score-bar-outer {
    background: #e5e7eb;
    border-radius: 999px;
    height: 10px;
    margin: 3px 0 1px 0;
}
.score-bar-inner { height: 10px; border-radius: 999px; }

/* 키워드 태그 */
.kw-tag {
    display: inline-block;
    background: #eff6ff;
    color: #1e40af;
    border: 1px solid #bfdbfe;
    border-radius: 6px;
    padding: 3px 11px;
    margin: 3px 4px;
    font-size: 0.82rem;
    font-weight: 700;
}

/* 테이블 */
table { width: 100%; border-collapse: collapse; }
th, td { padding: 6px 10px; color: #1f2937; }
thead tr { border-bottom: 2px solid #e5e7eb; }
tbody tr { border-bottom: 1px solid #f3f4f6; }
</style>
""", unsafe_allow_html=True)


# ── 세션 상태 ─────────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "context": None,
        # 0=대기, 1=s1실행, 2=s1완료, 3=s2실행, 4=s2완료, 5=s3실행, 6=s3완료, 7=s4실행, 8=s4완료
        "step": 0,
        "api_key": get_api_key(),
        "eval_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ── 사이드바 ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏛️ AI 사업계획서 빌더")
    st.markdown("---")

    st.markdown("**⚙️ API 설정**")
    api_key_input = st.text_input(
        "Anthropic API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="sk-ant-...",
    )
    if api_key_input:
        st.session_state.api_key = api_key_input

    st.markdown("---")
    st.markdown("**📋 구현 단계**")

    s = st.session_state.step
    sidebar_steps = [
        ("1", "문제 인식 Agent",  "done" if s >= 2 else ("active" if s == 1 else "pending")),
        ("2", "해결 방안 Agent",  "done" if s >= 4 else ("active" if s == 3 else "pending")),
        ("3", "성장 전략 Agent",  "done" if s >= 6 else ("active" if s == 5 else "pending")),
        ("4", "최종 검토 & 출력", "done" if s >= 8 else ("active" if s == 7 else "pending")),
    ]
    for num, label, status in sidebar_steps:
        color = {"active": "#1a56db", "done": "#059669", "pending": "#9ca3af"}[status]
        icon  = {"active": "🔵", "done": "✅", "pending": "⚪"}[status]
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin:5px 0'>"
            f"<span style='font-weight:700;color:{color};font-size:0.85rem'>Step {num}</span>"
            f"<span style='font-size:0.85rem;color:#374151'>{icon} {label}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    step_label = {0:"대기 중", 1:"Step 1 실행 중", 2:"Step 1 완료",
                  3:"Step 2 실행 중", 4:"Step 2 완료",
                  5:"Step 3 실행 중", 6:"Step 3 완료",
                  7:"Step 4 실행 중", 8:"완료"}.get(s, "")
    st.caption(f"현재: **{step_label}**")


# ── 메인 헤더 ─────────────────────────────────────────────────────────────────
st.markdown("# 🏛️ 정부지원사업 AI 빌더")
st.markdown("비즈니스 아이디어와 공고문을 입력하면 AI가 사업계획서 초안을 단계별로 생성합니다.")
st.markdown("---")

col_input, col_output = st.columns([1, 1.4], gap="large")


# ── 좌측 입력 패널 ────────────────────────────────────────────────────────────
with col_input:
    st.markdown("### 📝 입력")

    user_idea = st.text_area(
        "비즈니스 아이디어 *",
        placeholder="예) 중소기업 현장 작업자를 위한 AI 기반 안전사고 예측 솔루션...",
        height=160,
    )
    announcement_text = st.text_area(
        "공고문 (선택)",
        placeholder="예) 기술성 40점, 사업성 30점, 팀역량 20점...",
        height=90,
    )

    default_criteria = [
        {"item": "기술성",    "score": 40, "keywords": ["기술", "알고리즘", "AI", "특허", "혁신"]},
        {"item": "사업성",    "score": 30, "keywords": ["시장", "수익", "BM", "고객", "성장"]},
        {"item": "팀역량",    "score": 20, "keywords": ["팀", "대표", "경력", "전문가", "역량"]},
        {"item": "사회적가치","score": 10, "keywords": ["사회", "환경", "일자리", "공익", "지속가능"]},
    ]
    with st.expander("📊 평가 항목 편집"):
        criteria_json = st.text_area(
            "JSON",
            value=json.dumps(default_criteria, ensure_ascii=False, indent=2),
            height=200,
            label_visibility="collapsed",
        )

    run_disabled = not st.session_state.api_key or not user_idea.strip()

    if st.button("🚀 Step 1 — 문제 인식 분석", type="primary",
                 disabled=run_disabled, use_container_width=True):
        st.session_state.step = 1
        st.session_state.context = SharedContext(
            user_idea=user_idea,
            announcement_text=announcement_text,
        )
        st.session_state.eval_result = None
        st.rerun()

    if run_disabled and not st.session_state.api_key:
        st.warning("사이드바에서 API Key를 입력해 주세요.")
    elif run_disabled:
        st.info("비즈니스 아이디어를 입력하면 버튼이 활성화됩니다.")


# ── 우측 출력 패널 ────────────────────────────────────────────────────────────
with col_output:
    st.markdown("### 📄 분석 결과")

    # ━━━ Step 1 실행 중 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if st.session_state.step == 1:
        ctx: SharedContext = st.session_state.context
        with st.spinner("Agent 1 실행 중... 문제 인식 분석 중입니다."):
            try:
                client = anthropic.Anthropic(api_key=st.session_state.api_key)
                ctx = problem_agent.run(client, ctx)

                try:
                    criteria = json.loads(criteria_json)
                except Exception:
                    criteria = default_criteria

                eval_resp = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1000,
                    tools=ALL_TOOLS,
                    messages=[{
                        "role": "user",
                        "content": (
                            "evaluation_score_checker 툴로 아래 초안을 분석해줘.\n\n"
                            f"[초안]\n{ctx.problem_draft}\n\n"
                            f"[평가기준]\n{json.dumps(criteria, ensure_ascii=False)}"
                        ),
                    }],
                )

                eval_result = None
                for block in eval_resp.content:
                    if block.type == "tool_use" and block.name == "evaluation_score_checker":
                        eval_result = json.loads(
                            TOOL_HANDLERS["evaluation_score_checker"](block.input)
                        )
                        break

                ctx.eval_score = eval_result or {}
                if eval_result and eval_result.get("recall_needed"):
                    ctx.recall_needed = True
                    ctx.recall_target = "problem"

                st.session_state.context = ctx
                st.session_state.eval_result = eval_result
                st.session_state.step = 2
                st.rerun()

            except anthropic.AuthenticationError:
                st.error("❌ API Key가 올바르지 않습니다.")
                st.session_state.step = 0
            except Exception as e:
                st.error(f"❌ 오류: {e}")
                st.session_state.step = 0

    # ━━━ Step 1 완료 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elif st.session_state.step == 2:
        ctx: SharedContext = st.session_state.context
        kw = ctx.core_keywords

        # 키워드 카드
        if any(kw.values()):
            st.markdown(
                '<div class="card">'
                '<span class="step-badge badge-done">✅ Step 1 완료</span>'
                "<h4 style='margin:0.3rem 0 0.8rem;color:#1f2937'>추출된 핵심 키워드</h4>"
                f'<span class="kw-tag">🔴 사회적 문제: {kw.get("social_problem","—")}</span>'
                f'<span class="kw-tag">🔵 기술적 한계: {kw.get("tech_limitation","—")}</span>'
                f'<span class="kw-tag">🟢 시장 기회: {kw.get("market_opportunity","—")}</span>'
                "</div>",
                unsafe_allow_html=True,
            )

        if ctx.recall_needed:
            st.markdown(
                '<div class="card card-warn">'
                '<span style="color:#92400e">⚠️ <strong>Recall 신호</strong> — 일부 평가 항목 키워드 밀도 부족. Step 2에서 자동 보완됩니다.</span>'
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("#### 📋 문제 인식 초안")
        st.markdown(ctx.problem_draft.replace("[통계데이터_확인필요]", "**`[통계데이터_확인필요]`**"))

        # 평가 점수
        if st.session_state.eval_result:
            ev = st.session_state.eval_result
            overall = ev.get("overall_pct", 0)
            score_color = "#059669" if overall >= 70 else "#d97706" if overall >= 40 else "#dc2626"
            card_cls = "card-success" if overall >= 70 else "card-warn" if overall >= 40 else "card-error"

            rows_html = ""
            for row in ev.get("score_table", []):
                pct = row["coverage_pct"]
                bar_color = "#059669" if pct >= 70 else "#d97706" if pct >= 40 else "#dc2626"
                missing_str = ", ".join(row["missing_keywords"][:3]) or "없음"
                rows_html += (
                    f"<tr>"
                    f"<td style='padding:5px 8px;font-weight:600;color:#1f2937'>{row['item']}</td>"
                    f"<td style='padding:5px 8px;text-align:center;color:#1f2937'>{row['estimated_score']}/{row['max_score']}점</td>"
                    f"<td style='padding:5px 8px;min-width:110px'>"
                    f"<div class='score-bar-outer'><div class='score-bar-inner' style='width:{pct}%;background:{bar_color}'></div></div>"
                    f"<span style='font-size:0.74rem;color:#6b7280'>{pct}%</span></td>"
                    f"<td style='padding:5px 8px;font-size:0.78rem;color:#6b7280'>{missing_str}</td>"
                    f"</tr>"
                )

            recall_html = ""
            if ev.get("recall_needed"):
                recall_html = (
                    "<div style='margin-top:0.8rem;color:#92400e'>"
                    "⚠️ <strong>Recall 필요 항목:</strong> " + ", ".join(ev.get("recall_items", [])) + "</div>"
                )

            st.markdown(
                f'<div class="card {card_cls}">'
                f"<h4 style='margin:0 0 0.6rem;color:#1f2937'>📊 평가 지표 매칭 결과</h4>"
                f"<div style='font-size:1.5rem;font-weight:800;color:{score_color}'>"
                f"{ev['total_estimated']}/{ev['total_max']}점 ({overall}%)</div>"
                f"<table style='margin-top:0.8rem'>"
                f"<thead><tr style='border-bottom:2px solid #e5e7eb'>"
                f"<th style='text-align:left;padding:5px 8px;color:#374151;font-size:0.8rem'>항목</th>"
                f"<th style='padding:5px 8px;color:#374151;font-size:0.8rem'>추정점수</th>"
                f"<th style='padding:5px 8px;color:#374151;font-size:0.8rem'>커버리지</th>"
                f"<th style='text-align:left;padding:5px 8px;color:#374151;font-size:0.8rem'>누락키워드</th>"
                f"</tr></thead><tbody>{rows_html}</tbody></table>"
                + recall_html + "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        if st.button("⚡ Step 2 — 해결 방안 작성 (Bridge Logic)", type="primary", use_container_width=True):
            st.session_state.step = 3
            st.rerun()

        with st.expander("🔍 SharedContext"):
            st.json({"core_keywords": ctx.core_keywords, "recall_needed": ctx.recall_needed})

    # ━━━ Step 2 실행 중 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elif st.session_state.step == 3:
        ctx: SharedContext = st.session_state.context
        with st.spinner("Agent 2 실행 중... Bridge Logic 검증 포함"):
            try:
                client = anthropic.Anthropic(api_key=st.session_state.api_key)
                ctx = solution_agent.run(client, ctx)
                st.session_state.context = ctx
                st.session_state.step = 4
                st.rerun()
            except anthropic.AuthenticationError:
                st.error("❌ API Key가 올바르지 않습니다.")
                st.session_state.step = 2
            except Exception as e:
                st.error(f"❌ 오류: {e}")
                st.session_state.step = 2

    # ━━━ Step 2 완료 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elif st.session_state.step == 4:
        ctx: SharedContext = st.session_state.context
        kw = ctx.core_keywords

        with st.expander("✅ Step 1 — 문제 인식 초안", expanded=False):
            st.markdown(
                f'<span class="kw-tag">🔴 {kw.get("social_problem","—")}</span>'
                f'<span class="kw-tag">🔵 {kw.get("tech_limitation","—")}</span>'
                f'<span class="kw-tag">🟢 {kw.get("market_opportunity","—")}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(ctx.problem_draft.replace("[통계데이터_확인필요]", "**`[통계데이터_확인필요]`**"))

        bridge_card = "card-success" if ctx.bridge_validated else "card-warn"
        bridge_msg  = ("✅ <strong>Bridge Logic 검증 통과</strong> — 3대 키워드 모두 솔루션 제목에 반영됨"
                       if ctx.bridge_validated else
                       "⚠️ <strong>Bridge Logic 부분 미충족</strong> — 내용 검토 권장")
        bridge_color = "#065f46" if ctx.bridge_validated else "#92400e"
        st.markdown(
            f'<div class="card {bridge_card}">'
            f'<span style="color:{bridge_color}">{bridge_msg}</span>'
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("#### 📋 해결 방안 초안")
        st.markdown(
            ctx.solution_draft
            .replace("[수치데이터_확인필요]", "**`[수치데이터_확인필요]`**")
            .replace("[통계데이터_확인필요]", "**`[통계데이터_확인필요]`**")
        )

        st.markdown("---")
        if st.button("⚡ Step 3 — 성장 전략 작성 (TAM-SAM-SOM)", type="primary", use_container_width=True):
            st.session_state.step = 5
            st.rerun()

        with st.expander("🔍 SharedContext"):
            st.json({"core_keywords": ctx.core_keywords, "bridge_validated": ctx.bridge_validated})

    # ━━━ Step 3 실행 중 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elif st.session_state.step == 5:
        ctx: SharedContext = st.session_state.context
        with st.spinner("Agent 3 실행 중... 시장 규모 계산 및 성장 전략 작성 중입니다."):
            try:
                client = anthropic.Anthropic(api_key=st.session_state.api_key)
                ctx = growth_agent.run(client, ctx)
                st.session_state.context = ctx
                st.session_state.step = 6
                st.rerun()
            except anthropic.AuthenticationError:
                st.error("❌ API Key가 올바르지 않습니다.")
                st.session_state.step = 4
            except Exception as e:
                st.error(f"❌ 오류: {e}")
                st.session_state.step = 4

    # ━━━ Step 3 완료 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elif st.session_state.step == 6:
        ctx: SharedContext = st.session_state.context
        kw = ctx.core_keywords

        # 이전 단계 요약 (접힘)
        with st.expander("✅ Step 1 — 문제 인식 초안", expanded=False):
            st.markdown(ctx.problem_draft.replace("[통계데이터_확인필요]", "**`[통계데이터_확인필요]`**"))

        with st.expander("✅ Step 2 — 해결 방안 초안", expanded=False):
            st.markdown(
                ctx.solution_draft
                .replace("[수치데이터_확인필요]", "**`[수치데이터_확인필요]`**")
                .replace("[통계데이터_확인필요]", "**`[통계데이터_확인필요]`**")
            )

        # TAM-SAM-SOM 수치 카드
        md = ctx.market_data
        if md:
            st.markdown(
                '<div class="card card-success">'
                '<span class="step-badge badge-done">✅ market_size_calculator 실행 완료</span>'
                f"<h4 style='margin:0.4rem 0 0.8rem;color:#1f2937'>📈 {md.get('market_name','시장 규모')}</h4>"
                "<div style='display:flex;gap:1.5rem;flex-wrap:wrap'>"
                f"<div><div style='font-size:0.75rem;color:#6b7280;font-weight:600'>TAM (전체)</div>"
                f"<div style='font-size:1.3rem;font-weight:800;color:#1a56db'>{md.get('tam',0):,.0f}억 원</div>"
                f"<div style='font-size:0.75rem;color:#6b7280'>{md.get('target_year',3)}년 후 {md.get('tam_future',0):,.0f}억 원</div></div>"
                f"<div><div style='font-size:0.75rem;color:#6b7280;font-weight:600'>SAM (유효)</div>"
                f"<div style='font-size:1.3rem;font-weight:800;color:#059669'>{md.get('sam',0):,.0f}억 원</div>"
                f"<div style='font-size:0.75rem;color:#6b7280'>{md.get('target_year',3)}년 후 {md.get('sam_future',0):,.0f}억 원</div></div>"
                f"<div><div style='font-size:0.75rem;color:#6b7280;font-weight:600'>SOM (수익)</div>"
                f"<div style='font-size:1.3rem;font-weight:800;color:#d97706'>{md.get('som',0):,.0f}억 원</div>"
                f"<div style='font-size:0.75rem;color:#6b7280'>{md.get('target_year',3)}년 후 {md.get('som_future',0):,.0f}억 원</div></div>"
                f"<div><div style='font-size:0.75rem;color:#6b7280;font-weight:600'>CAGR</div>"
                f"<div style='font-size:1.3rem;font-weight:800;color:#7c3aed'>{md.get('growth_rate_pct',12)}%</div></div>"
                "</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("#### 📋 성장 전략 초안")
        st.markdown(
            ctx.growth_draft
            .replace("[수치데이터_확인필요]", "**`[수치데이터_확인필요]`**")
            .replace("[통계데이터_확인필요]", "**`[통계데이터_확인필요]`**")
        )

        # 전체 문서 다운로드
        st.markdown("---")
        full_doc = (
            f"# 정부지원사업 사업계획서 초안\n\n"
            f"{ctx.problem_draft}\n\n"
            f"{ctx.solution_draft}\n\n"
            f"{ctx.growth_draft}"
        )
        st.download_button(
            "📥 전체 초안 마크다운 다운로드",
            data=full_doc.encode("utf-8"),
            file_name="business_plan_draft.md",
            mime="text/markdown",
            use_container_width=True,
        )

        st.markdown("---")
        if st.button("⚡ Step 4 — 최종 검토 & 완성본 출력", type="primary", use_container_width=True):
            st.session_state.step = 7
            st.rerun()

        with st.expander("🔍 SharedContext"):
            st.json({
                "core_keywords": ctx.core_keywords,
                "bridge_validated": ctx.bridge_validated,
                "market_data": ctx.market_data,
            })

    # ━━━ Step 4 실행 중 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elif st.session_state.step == 7:
        ctx: SharedContext = st.session_state.context
        with st.spinner("Agent 4 실행 중... 문체 교정 및 최종 검토 중입니다. (시간이 다소 걸립니다)"):
            try:
                client = anthropic.Anthropic(api_key=st.session_state.api_key)
                ctx = review_agent.run(client, ctx)
                st.session_state.context = ctx
                st.session_state.step = 8
                st.rerun()
            except anthropic.AuthenticationError:
                st.error("❌ API Key가 올바르지 않습니다.")
                st.session_state.step = 6
            except Exception as e:
                st.error(f"❌ 오류: {e}")
                st.session_state.step = 6

    # ━━━ Step 4 완료 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elif st.session_state.step == 8:
        ctx: SharedContext = st.session_state.context

        # 최종 평가 점수 카드
        fe = ctx.final_eval
        if fe:
            overall = fe.get("overall_pct", 0)
            score_color = "#059669" if overall >= 70 else "#d97706" if overall >= 40 else "#dc2626"
            card_cls = "card-success" if overall >= 70 else "card-warn" if overall >= 40 else "card-error"

            rows_html = ""
            for row in fe.get("score_table", []):
                pct = row["coverage_pct"]
                bar_color = "#059669" if pct >= 70 else "#d97706" if pct >= 40 else "#dc2626"
                missing_str = ", ".join(row["missing_keywords"][:3]) or "없음"
                rows_html += (
                    f"<tr>"
                    f"<td style='padding:5px 8px;font-weight:600;color:#1f2937'>{row['item']}</td>"
                    f"<td style='padding:5px 8px;text-align:center;color:#1f2937'>{row['estimated_score']}/{row['max_score']}점</td>"
                    f"<td style='padding:5px 8px;min-width:110px'>"
                    f"<div class='score-bar-outer'><div class='score-bar-inner' style='width:{pct}%;background:{bar_color}'></div></div>"
                    f"<span style='font-size:0.74rem;color:#6b7280'>{pct}%</span></td>"
                    f"<td style='padding:5px 8px;font-size:0.78rem;color:#6b7280'>{missing_str}</td>"
                    f"</tr>"
                )

            st.markdown(
                f'<div class="card {card_cls}">'
                f'<span class="step-badge badge-done">✅ 최종 평가 점수</span>'
                f"<div style='font-size:1.8rem;font-weight:800;color:{score_color};margin:0.4rem 0'>"
                f"{fe['total_estimated']}/{fe['total_max']}점 ({overall}%)</div>"
                f"<table style='margin-top:0.6rem'>"
                f"<thead><tr style='border-bottom:2px solid #e5e7eb'>"
                f"<th style='text-align:left;padding:5px 8px;color:#374151;font-size:0.8rem'>항목</th>"
                f"<th style='padding:5px 8px;color:#374151;font-size:0.8rem'>최종점수</th>"
                f"<th style='padding:5px 8px;color:#374151;font-size:0.8rem'>커버리지</th>"
                f"<th style='text-align:left;padding:5px 8px;color:#374151;font-size:0.8rem'>누락키워드</th>"
                f"</tr></thead><tbody>{rows_html}</tbody></table>"
                + ("" if not fe.get("recall_needed") else
                   "<div style='margin-top:0.7rem;color:#92400e'>⚠️ <strong>보완 권장:</strong> "
                   + ", ".join(fe.get("recall_items", [])) + "</div>")
                + "</div>",
                unsafe_allow_html=True,
            )

        # 문체 교정 통계
        rs = ctx.refined_sections
        if rs:
            st.markdown(
                '<div class="card">'
                '<span class="step-badge">style_refiner 적용 완료</span>'
                "<div style='display:flex;gap:1.5rem;margin-top:0.6rem;flex-wrap:wrap'>",
                unsafe_allow_html=True,
            )
            for sec, label in [("problem","문제인식"), ("solution","해결방안"), ("growth","성장전략")]:
                pass  # 아래에서 개별 렌더

            st.markdown(
                '<div class="card">'
                '<span class="step-badge">style_refiner 적용 완료</span>'
                "<p style='color:#374151;margin:0.5rem 0 0'>문체 교정이 완료되었습니다. "
                "아래 최종 문서를 확인하세요.</p></div>",
                unsafe_allow_html=True,
            )

        # 최종 통합 문서
        st.markdown("#### 📄 최종 사업계획서 초안")
        final_display = (
            ctx.final_doc
            .replace("[통계데이터_확인필요]", "**`[통계데이터_확인필요]`**")
            .replace("[수치데이터_확인필요]", "**`[수치데이터_확인필요]`**")
        )
        st.markdown(final_display)

        # 다운로드 버튼
        st.markdown("---")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "📥 최종본 다운로드 (.md)",
                data=ctx.final_doc.encode("utf-8"),
                file_name="business_plan_final.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col_dl2:
            full_raw = (
                f"# 사업계획서 초안 (전체)\n\n"
                f"## 문제인식\n{ctx.problem_draft}\n\n"
                f"## 해결방안\n{ctx.solution_draft}\n\n"
                f"## 성장전략\n{ctx.growth_draft}\n\n"
                f"## 최종본\n{ctx.final_doc}"
            )
            st.download_button(
                "📥 전체 단계 원본 (.md)",
                data=full_raw.encode("utf-8"),
                file_name="business_plan_all_steps.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with st.expander("🔍 SharedContext 전체"):
            st.json({
                "core_keywords": ctx.core_keywords,
                "bridge_validated": ctx.bridge_validated,
                "market_data": {k: v for k, v in ctx.market_data.items() if k != "markdown_table"},
                "final_eval_pct": fe.get("overall_pct") if fe else None,
            })

    # ━━━ 대기 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    else:
        st.markdown(
            '<div class="card" style="text-align:center;padding:3rem 2rem">'
            "<div style='font-size:3rem'>🏛️</div>"
            "<h3 style='color:#1f2937;margin:0.5rem 0'>분석 대기 중</h3>"
            "<p style='color:#6b7280;margin:0'>좌측에 비즈니스 아이디어를 입력하고<br>"
            "<strong>Step 1 실행</strong> 버튼을 클릭하세요.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
