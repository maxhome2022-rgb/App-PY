import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
LAST_YEAR = 2025      # 수업 기준 기간의 끝
MIN_DAYS = 300        # 연간 최소 관측일 수

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="centered")
st.title("🌡️ 기온 예측기")
st.caption("서울 연평균기온에 회귀 직선을 맞추고, 선택한 해의 기온을 예상합니다.")


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_yearly(url: str):
    df = pd.read_csv(url, encoding="utf-8")
    df["날짜"] = pd.to_datetime(df["날짜"].astype(str).str.strip(), errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])
    df["연도"] = df["날짜"].dt.year
    df = df[df["연도"] <= LAST_YEAR]

    g = df.groupby("연도")["평균기온"].agg(관측일수="count", 연평균기온="mean").reset_index()
    excluded = g[g["관측일수"] < MIN_DAYS]
    yearly = g[g["관측일수"] >= MIN_DAYS].reset_index(drop=True)
    return yearly, excluded


try:
    yearly, excluded = load_yearly(DATA_URL)
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요: {e}")
    st.stop()

if len(yearly) < 3:
    st.error("회귀 직선을 만들기에 데이터가 부족해요.")
    st.stop()

x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)
slope, intercept = np.polyfit(x, y, 1)
r = float(np.corrcoef(x, y)[0, 1])

n_years = len(yearly)
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())

# 최근 20년(끝 연도 포함 20년 구간)만으로 구한 회귀
RECENT_YEARS = 20
recent = yearly[yearly["연도"] > end_year - RECENT_YEARS].reset_index(drop=True)
n_recent = len(recent)
if n_recent >= 3:
    xr = recent["연도"].to_numpy(dtype=float)
    yr = recent["연평균기온"].to_numpy(dtype=float)
    slope_r, intercept_r = np.polyfit(xr, yr, 1)
    r_recent = float(np.corrcoef(xr, yr)[0, 1])
else:
    slope_r = intercept_r = r_recent = None

# ── 슬라이더와 예상 기온 ──
year = st.slider("예상할 연도", min_value=1900, max_value=2100, value=2050, step=1)
pred = slope * year + intercept

st.markdown(
    f"""
    <div style="text-align:center; padding:1.2rem 0;">
        <div style="font-size:1.1rem; opacity:0.7;">{year}년 서울 예상 연평균기온</div>
        <div style="font-size:4.5rem; font-weight:800; line-height:1.2;">{pred:.2f} ℃</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── 회귀 정보 ──
c1, c2, c3 = st.columns(3)
c1.metric("직선을 만든 해의 개수", f"{n_years}개")
c2.metric("시작 연도", f"{start_year}년")
c3.metric("끝 연도", f"{end_year}년")

st.metric("상관계수 (r)", f"{r:.3f}")


def per100(v):
    return v * 100


def word(v):
    return "상승" if v >= 0 else "하강"


st.markdown(
    f"""
    <div style="text-align:center; padding:1rem 0 0.4rem;">
        <div style="font-size:1.1rem; opacity:0.7;">회귀 직선의 기울기 (전체 {start_year}~{end_year}년)</div>
        <div style="font-size:3.4rem; font-weight:800; line-height:1.2;">
            100년에 {per100(slope):+.2f} ℃ {word(slope)}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("기울기 비교: 전체 기간 vs 최근 20년")
col_all, col_rec = st.columns(2)
with col_all:
    with st.container(border=True):
        st.markdown("**전체 기간**")
        st.caption(f"{start_year}~{end_year}년 · {n_years}개 해")
        st.markdown(
            f"<div style='font-size:2.2rem; font-weight:800;'>{per100(slope):+.2f} ℃ / 100년</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"상관계수 r = {r:.3f}")
with col_rec:
    with st.container(border=True):
        st.markdown("**최근 20년**")
        if slope_r is None:
            st.caption("해의 개수가 부족해 계산할 수 없어요.")
        else:
            st.caption(f"{int(recent['연도'].min())}~{int(recent['연도'].max())}년 · {n_recent}개 해")
            st.markdown(
                f"<div style='font-size:2.2rem; font-weight:800;'>{per100(slope_r):+.2f} ℃ / 100년</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"상관계수 r = {r_recent:.3f}")

if slope_r is not None:
    diff = per100(slope_r) - per100(slope)
    if slope > 0 and slope_r > 0:
        ratio_txt = f" (전체 기간의 약 {slope_r / slope:.1f}배)"
    else:
        ratio_txt = ""
    st.info(
        f"최근 20년의 기울기는 전체 기간보다 100년당 {diff:+.2f} ℃ 차이가 나요{ratio_txt}. "
        "다만 20년은 짧은 구간이라 해마다의 들쭉날쭉함에 크게 흔들릴 수 있어요."
    )

st.caption(
    f"기준: {LAST_YEAR}년까지의 자료, 관측일 {MIN_DAYS}일 미만인 해는 제외. "
    f"회귀식: 기온 = {slope:.5f} × 연도 + {intercept:.3f}"
)
if len(excluded):
    ex = ", ".join(f"{int(a)}년({int(b)}일)" for a, b in zip(excluded["연도"], excluded["관측일수"]))
    st.caption(f"제외된 해: {ex}")

# ── 산점도 + 회귀 직선 ──
x_line = np.array([min(1900, start_year), 2100], dtype=float)
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=yearly["연도"], y=yearly["연평균기온"], mode="markers", name="연평균기온",
    marker=dict(size=8, color="#1f77b4"),
    hovertemplate="%{x}년<br>%{y:.2f} ℃<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=x_line, y=slope * x_line + intercept, mode="lines", name="회귀 직선",
    line=dict(color="#d62728", width=3),
    hovertemplate="%{x}년<br>예상 %{y:.2f} ℃<extra></extra>",
))
if slope_r is not None:
    x_rec = np.array([float(recent["연도"].min()), float(recent["연도"].max())])
    fig.add_trace(go.Scatter(
        x=x_rec, y=slope_r * x_rec + intercept_r, mode="lines", name="최근 20년 직선",
        line=dict(color="#2ca02c", width=4, dash="dash"),
        hovertemplate="%{x}년<br>%{y:.2f} ℃<extra></extra>",
    ))
fig.add_trace(go.Scatter(
    x=[year], y=[pred], mode="markers", name=f"{year}년 예상",
    marker=dict(size=16, color="#ff7f0e", symbol="star", line=dict(width=1, color="black")),
    hovertemplate="%{x}년<br>예상 %{y:.2f} ℃<extra></extra>",
))
fig.update_layout(
    title=f"서울 연평균기온 산점도와 회귀 직선 (r = {r:.3f})",
    xaxis_title="연도", yaxis_title="연평균기온 (℃)",
    xaxis=dict(range=[1900, 2100]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=80, b=10),
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("연도별 평균기온 데이터 보기"):
    st.dataframe(yearly.rename(columns={"연평균기온": "연평균기온(℃)"}), use_container_width=True)
