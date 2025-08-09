# app.py — 팟캐스트 AI 코치 (화자 수 필수 입력 + 화자분리/진단/통계/타임라인/Gemini 통합)

import streamlit as st
import pandas as pd
import numpy as np
import tempfile, os, json, time as _time
from datetime import datetime

import plotly.express as px
from plotly import graph_objects as go

from faster_whisper import WhisperModel
import google.generativeai as genai

# --- NumPy 2.0 임시 호환 (일부 라이브러리의 np.NaN 참조 대비) ---
if not hasattr(np, "NaN"):
    np.NaN = np.nan

# HF login for diarization
try:
    from huggingface_hub import login as hf_login
except Exception:
    hf_login = None  # 미설치 시 무시

# ==================== Config & Secrets ====================
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
HF_TOKEN = st.secrets.get("HF_TOKEN")   # 없으면 화자분리 비활성

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

if HF_TOKEN and hf_login is not None:
    try:
        hf_login(HF_TOKEN)
    except Exception:
        pass  # 로그인 실패해도 앱은 계속 동작

st.set_page_config(page_title="팟캐스트 AI 코치", layout="wide")
st.title("🎧 팟캐스트 AI 코치 & 자기 성찰")

st.markdown("""
1) 오디오(WAV/MP3/M4A) 업로드 → 2) 주제 입력 → 3) **AI 분석 시작하기**  
Hugging Face 토큰이 있으면 **화자 분리**가 적용되어 개인별 참여도(발화 횟수/시간/비율)를 보여줍니다.
""")

# ==================== Utils ====================
def _sec_to_mmss(s: float) -> str:
    if s is None or s < 0:
        return "00:00"
    return _time.strftime("%M:%S", _time.gmtime(float(s)))

# ==================== Models (cache) ====================
@st.cache_resource
def load_whisper():
    return WhisperModel("small", device="cpu", compute_type="int8")
model_whisper = load_whisper()

@st.cache_resource
def load_diar_pipeline():
    """pyannote diarization pipeline (토큰 없으면 None 반환)"""
    if not HF_TOKEN:
        return None
    try:
        from pyannote.audio import Pipeline
        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
        return pipe
    except Exception as e:
        st.warning(f"화자 분리 파이프라인 로드 실패: {e}")
        return None
diar_pipeline = load_diar_pipeline()

# ==================== Core: STT + 기본 지표 ====================
def analyze_podcast_audio(file_path, progress_callback=None):
    if progress_callback: progress_callback(10, "STT 변환 중... (길이에 따라 수십 초 소요)")
    segments, info = model_whisper.transcribe(file_path, vad_filter=True, word_timestamps=False)

    if progress_callback: progress_callback(55, "텍스트 정리 및 기본 지표 계산...")

    rows = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in segments]
    df = pd.DataFrame(rows)

    if df.empty:
        total_audio = round(getattr(info, "duration", 0) / 60, 1) if info else 0.0
        return pd.DataFrame(columns=["start","end","text","duration","word_count"]), {
            "총 길이(분)": total_audio, "발화 총 시간(분)": 0.0, "말속도(WPM)": 0.0, "침묵 비율(%)": 100.0
        }

    df["duration"] = df["end"] - df["start"]
    df["word_count"] = df["text"].str.split().str.len()

    total_time = float(df["duration"].sum())
    total_audio = float(getattr(info, "duration", total_time))
    wpm = (df["word_count"].sum() / (total_time / 60)) if total_time > 0 else 0
    silence_ratio = max(0, 1 - (total_time / total_audio)) if total_audio > 0 else 0

    metrics = {
        "총 길이(분)": round(total_audio / 60, 1),
        "발화 총 시간(분)": round(total_time / 60, 1),
        "말속도(WPM)": round(wpm, 1),
        "침묵 비율(%)": round(silence_ratio * 100, 1),
    }
    return df, metrics

# ==================== Diarization + attach speakers ====================
def diarize_audio(file_path, progress_callback=None, num_speakers: int | None = None):
    if diar_pipeline is None:
        return pd.DataFrame(columns=["start","end","speaker"])
    if progress_callback: progress_callback(65, "화자 분리 중...")

    diar_kwargs = {}
    if num_speakers and num_speakers > 0:
        diar_kwargs["num_speakers"] = int(num_speakers)

    diar = diar_pipeline(file_path, **diar_kwargs)

    out = []
    for turn, _, spk in diar.itertracks(yield_label=True):
        out.append({"start": round(turn.start, 2), "end": round(turn.end, 2), "speaker": spk})
    return pd.DataFrame(out)

def attach_speaker(trans_df: pd.DataFrame, diar_df: pd.DataFrame):
    """각 Whisper 발화에 가장 많이 겹치는 화자 라벨 부여"""
    if diar_df.empty or trans_df.empty:
        trans_df["speaker"] = "S1"
        return trans_df
    speakers = []
    for _, u in trans_df.iterrows():
        overlap = diar_df.apply(lambda r: max(0, min(u.end, r.end) - max(u.start, r.start)), axis=1)
        if overlap.max() <= 0:
            speakers.append("UNK")
        else:
            speakers.append(diar_df.loc[int(overlap.idxmax()), "speaker"])
    trans_df["speaker"] = speakers
    return trans_df

def per_speaker_stats(df: pd.DataFrame):
    """화자별 발화횟수/시간/비율 + '총 N회 중 n회 = p%'"""
    total_turns = int(len(df))
    total_time = float(df["duration"].sum()) if total_turns else 0.0

    g = df.groupby("speaker", dropna=False).agg(
        turns=("text","count"),
        speak_time=("duration","sum")
    ).reset_index()

    g["턴 비율(%)"] = (g["turns"] / total_turns * 100) if total_turns > 0 else 0.0
    g["시간 비율(%)"] = (g["speak_time"] / total_time * 100) if total_time > 0 else 0.0
    g["턴 비율(%)"] = g["턴 비율(%)"].round(1)
    g["시간 비율(%)"] = g["시간 비율(%)"].round(1)
    if total_turns > 0:
        g["요약"] = g.apply(lambda r: f"총 {total_turns}회 중 {int(r.turns)}회 = {round(r['턴 비율(%)'],1)}%", axis=1)
    else:
        g["요약"] = "-"
    return g

# ==================== Gemini 평가 ====================
def gemini_podcast_analysis(df, topic_hint="", progress_callback=None):
    if df.empty:
        return {"structure":1,"delivery":1,"content":1,"creativity":1,"summary":"음성이 감지되지 않아 AI 평가를 생략했습니다."}
    if progress_callback: progress_callback(75, "AI 모델에 콘텐츠 평가 요청 중...")

    records = df["text"].tolist()
    prompt = f"""
너는 학생들의 발표 능력을 돕는 팟캐스트 코칭 AI야. 전사본을 분석해 아래 기준으로 JSON만 출력해.
주제: {topic_hint}
[구성 1~5] 논리/전환, [전달력 1~5] 발음·속도·억양, [내용 1~5] 정확성·근거·핵심성, [창의성 1~5] 독창성·사례/연출.
출력 ONLY JSON: {{"structure":int,"delivery":int,"content":int,"creativity":int,"summary":"2~3문장 코멘트"}}
전사: {records}
""".strip()

    for m in ["gemini-2.5-flash", "gemini-pro"]:
        try:
            mdl = genai.GenerativeModel(m)
            res = mdl.generate_content(prompt)
            txt = (res.text or "").strip()
            if txt.startswith("```"):
                txt = txt.strip("` \n")
                if txt.lower().startswith("json"):
                    txt = txt[4:].strip()
            if "{" in txt and "}" in txt:
                txt = txt[txt.find("{"): txt.rfind("}")+1]
            out = json.loads(txt)
            for k in ["structure","delivery","content","creativity","summary"]:
                out.setdefault(k, 1 if k!="summary" else "")
            if progress_callback: progress_callback(95, "AI 평가 완료. 결과 정리 중...")
            return out
        except Exception:
            continue
    st.error("⚠️ AI 응답을 해석할 수 없어 기본값으로 표시합니다.")
    return {"structure":1,"delivery":1,"content":1,"creativity":1,"summary":""}

# ==================== UI: 업로드/설정/실행 (폼 + 필수 입력 강제) ====================
uploaded_file = st.file_uploader("🎵 팟캐스트 파일 업로드 (WAV/MP3/M4A)", type=["wav","mp3","m4a"])
topic_hint = st.text_input("📌 팟캐스트 주제 (필수)", value="")

# 화자 분리 파이프라인 사용 가능 여부
diar_available = diar_pipeline is not None

# 폼: 필수값 없으면 분석 시작 안 함
with st.form("analyze_form", clear_on_submit=False):
    if diar_available:
        st.markdown("### 👥 화자 분리 설정")
        use_diar = st.checkbox("화자 분리 사용(권장)", value=True)
        num_speakers = st.number_input(
            "화자 수(필수)", min_value=1, max_value=6, step=1, value=2,
            help="대부분 2~3명으로 시작해 보세요."
        ) if use_diar else None
    else:
        use_diar = False
        num_speakers = None
        st.info("ℹ️ Hugging Face 토큰이 없어 화자 분리를 건너뜁니다. (단일 화자 처리)")

    submit = st.form_submit_button("🤖 AI 분석 시작하기")

    if submit:
        # 1) 공통 입력 확인
        if not uploaded_file:
            st.error("오디오 파일을 업로드해 주세요."); st.stop()
        if not topic_hint.strip():
            st.error("팟캐스트 주제를 입력해 주세요."); st.stop()
        # 2) 화자 분리 필수값 확인
        if diar_available and use_diar and (num_speakers is None):
            st.error("화자 수를 입력해 주세요."); st.stop()

        # ---- 실제 분석 시작 ----
        progress = st.progress(0); status = st.empty()
        def step(p,msg): progress.progress(int(p)); status.info(msg)

        step(5, "오디오 파일 준비 중...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getvalue()); tmp_path = tmp.name

        # 1) STT + 기본지표
        df, basic = analyze_podcast_audio(tmp_path, step)

        # 2) 화자 분리 → speaker 라벨 부착
        if diar_available and use_diar:
            diar_df = diarize_audio(tmp_path, step, num_speakers=int(num_speakers))
        else:
            diar_df = pd.DataFrame(columns=["start","end","speaker"])
        st.session_state.diar_df = diar_df
        df = attach_speaker(df, diar_df)

        # 3) Gemini 평가
        gem = gemini_podcast_analysis(df, topic_hint, step)

        try: os.remove(tmp_path)
        except: pass

        st.session_state.analysis_complete = True
        st.session_state.df = df
        st.session_state.metrics_basic = basic
        st.session_state.metrics_gemini = gem
        st.session_state.spk_stats = per_speaker_stats(df)

        step(100, "분석 완료! 아래에서 리포트를 확인하세요."); status.success("분석 완료!")

# ==================== 결과 탭 & 진단 ====================
if st.session_state.get("analysis_complete"):
    st.divider()
    st.header("📈 AI 코치의 분석 리포트")

    # 🔍 화자 분리 진단
    if 'diar_df' in st.session_state:
        dd = st.session_state.diar_df
        if dd.empty:
            st.info("ℹ️ 화자 분리 결과가 비어 있습니다. (토큰 미설정/모델 로드 실패/1인 화자 가능)")
        else:
            st.success(f"✅ 화자 분리 완료: 세그먼트 {len(dd)}개, 화자 {dd['speaker'].nunique()}명")
            with st.expander("화자 분리 원시 세그먼트 보기"):
                st.dataframe(dd.head(50), use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["📊 종합/화자별", "🗣️ 발화 타임라인", "🤔 자기 성찰"])

    # --- 탭1: 종합 & 화자별 ---
    with tab1:
        st.subheader("🎯 콘텐츠 평가 (Gemini)")
        st.info(f"**AI 총평:** {st.session_state.metrics_gemini['summary']}")
        score_df = pd.DataFrame({
            "항목": ["구성","전달력","내용 충실도","창의성"],
            "AI 점수": [
                st.session_state.metrics_gemini["structure"],
                st.session_state.metrics_gemini["delivery"],
                st.session_state.metrics_gemini["content"],
                st.session_state.metrics_gemini["creativity"],
            ],
        })
        fig_bar = px.bar(score_df, x="항목", y="AI 점수", range_y=[0,5.5], text="AI 점수", color="항목")
        fig_bar.update_traces(textposition="outside"); st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("⏱️ 제작 품질 지표")
        cols = st.columns(len(st.session_state.metrics_basic))
        for col, (label, value) in zip(cols, st.session_state.metrics_basic.items()):
            col.metric(label, value)

        # 🔥 화자별 참여도
        st.subheader("👥 화자별 참여도")
        spk = st.session_state.spk_stats.copy()
        if not spk.empty:
            spk_disp = spk.rename(columns={"speaker":"화자","turns":"발화 횟수","speak_time":"발화 시간(초)"})
            st.dataframe(spk_disp[["화자","발화 횟수","턴 비율(%)","발화 시간(초)","시간 비율(%)","요약"]],
                         use_container_width=True)
            st.plotly_chart(px.bar(spk, x="speaker", y="turns", text="턴 비율(%)",
                                   title="화자별 발화 횟수(총 N회 대비)"), use_container_width=True)
        else:
            st.caption("화자 분리 정보가 없습니다. (HF 토큰 미설정/모델 로드 실패/1인 화자)")

    # --- 탭2: 타임라인 (go.Bar + base, 화자별 색상) ---
    with tab2:
        st.subheader("🕒 발화 구간 타임라인")
        df2 = st.session_state.df.copy()
        if df2.empty:
            st.warning("오디오에서 음성이 감지되지 않아 타임라인을 표시할 수 없습니다.")
        else:
            df2 = df2.reset_index(drop=True)
            df2["utt_id"] = df2.index.map(lambda i: f"#{i+1}")
            df2["bar_len"] = (df2["end"] - df2["start"]).clip(lower=0.01)

            uniq = df2["speaker"].fillna("S1").unique().tolist()
            pal = px.colors.qualitative.Set2
            color_map = {s: pal[i % len(pal)] for i, s in enumerate(uniq)}
            colors = df2["speaker"].fillna("S1").map(color_map)

            total_audio = max(float(df2["end"].max()), 0.0)
            tick_vals = np.linspace(0.0, total_audio, 7)
            tick_text = [_sec_to_mmss(v) for v in tick_vals]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df2["bar_len"], y=df2["utt_id"], base=df2["start"], orientation="h",
                marker=dict(color=colors),
                hovertext=df2.apply(lambda r: f"[{r['speaker']}] {r['text']}", axis=1),
                hovertemplate="<b>%{y}</b><br>시작: %{base:.2f}s<br>길이: %{x:.2f}s<br><br>%{hovertext}"
            ))
            fig.update_layout(
                title="발화 구간 (화자별 색상)",
                barmode="stack", bargap=0.15,
                xaxis=dict(title="오디오 시간 (분:초)", tickmode="array", tickvals=tick_vals, ticktext=tick_text, rangemode="nonnegative"),
                yaxis=dict(title="발화 ID"),
                height=440, showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

    # --- 탭3: 자기 성찰 (저장 없음) ---
    with tab3:
        st.subheader("🤔 스스로 점검하기")
        st.slider("1. 구성", 1, 5, 3)
        st.slider("2. 전달력", 1, 5, 3)
        st.slider("3. 내용 충실도", 1, 5, 3)
        st.slider("4. 창의성", 1, 5, 3)
        st.text_area("메모 (저장되지 않습니다)")
