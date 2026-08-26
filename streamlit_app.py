from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st


st.set_page_config(
    page_title="경기도 장례시설 접근성 및 개선 시급지역 분석",
    page_icon="🗺️",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stMetric"] {
        min-height: 118px;
        padding: 1.1rem 1.25rem;
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        box-shadow: 0 7px 20px rgba(15, 23, 42, 0.09);
    }
    [data-testid="stMetricLabel"] {
        color: #4b5563;
        font-weight: 650;
    }
    [data-testid="stMetricValue"] {
        color: #111827;
        font-weight: 750;
    }
    [data-testid="stExpander"] {
        background: #f3f4f6;
        border: 1px solid #d1d5db;
        border-radius: 10px;
    }
    [data-testid="stExpander"] details summary {
        color: #1f2937;
        font-weight: 700;
    }
    .header-divider {
        height: 1px;
        margin: 1.35rem 0 1.7rem;
        background: #cbd5e1;
        border: 0;
    }
    .map-title-box {
        box-sizing: border-box;
        width: 100%;
        margin: 0.6rem 0 0.45rem;
        padding: 0.8rem 1.05rem;
        background: #111111;
        color: #ffffff;
        font-size: 1.35rem;
        font-weight: 750;
        line-height: 1.35;
    }
    .section-heading {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        margin: 0.4rem 0 0.9rem;
        color: #111827;
        font-size: 1.35rem;
        font-weight: 750;
        line-height: 1.35;
    }
    .section-heading::before {
        width: 0.72rem;
        height: 0.72rem;
        background: #111111;
        content: "";
        flex: 0 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
GRID_PATH = DATA_DIR / "gyeonggi_funeral_access_grid.geojson"
FACILITY_CANDIDATES = [
    DATA_DIR / "facilities.csv",
    DATA_DIR / "장례시설현황.csv",
]

METRICS = {
    "80세 이상 인구 분포": {
        "column": "pop80",
        "grade_column": "pop80_grade",
        # Colab의 viridis 역방향(낮은 구간 → 높은 구간)
        "palette": ["#cae11f", "#48c16e", "#21918c", "#365d8d", "#481d6f"],
        "labels": ["0명", "25명 미만", "50명 미만", "100명 미만", "100명 이상"],
        "alpha": 184,  # 72%
        "direction": "80세 이상 인구가 많을수록 장례서비스에 대한 잠재수요가 클 것으로 예상",
    },
    "수요 대비 장례시설 접근성": {
        "column": "access_per_1000",
        "grade_column": "competitive_grade",
        # Colab의 cool: 매우 불량(cyan) → 매우 양호(magenta)
        "palette": ["#14ebff", "#4ab5ff", "#807fff", "#b54aff", "#eb14ff"],
        "labels": ["매우 불량", "불량", "보통", "양호", "매우 양호"],
        "alpha": 191,  # 75%
        "direction": "장례시설 공급 상황과 시설 인근 서비스 잠재수요를 함께 고려",
    },
    "장례서비스 개선 시급성": {
        "column": "priority_score",
        "grade_column": "urgency_grade",
        # Colab의 afmhot 역방향(매우 낮음 → 매우 높음)
        "palette": ["#ffffd7", "#ffeb6b", "#ff8001", "#941400", "#280000"],
        "labels": ["매우 낮음", "낮음", "보통", "높음", "매우 높음"],
        "alpha": 184,  # 72%
        "direction": "80세 이상 인구가 많을수록, 수요 대비 접근성이 낮을수록 시급성이 높아짐",
    },
}

POPULATION_LABELS = ["0명", "25명 미만", "50명 미만", "100명 미만", "100명 이상"]
ACCESS_LABELS = ["매우 불량", "불량", "보통", "양호", "매우 양호"]
URGENCY_LABELS = ["매우 낮음", "낮음", "보통", "높음", "매우 높음"]
URGENCY_BINS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

SIGUNGU_BY_SGIS_PREFIX = {
    "31011": "수원시 장안구",
    "31012": "수원시 권선구",
    "31013": "수원시 팔달구",
    "31014": "수원시 영통구",
    "31021": "성남시 수정구",
    "31022": "성남시 중원구",
    "31023": "성남시 분당구",
    "31030": "의정부시",
    "31041": "안양시 만안구",
    "31042": "안양시 동안구",
    "31051": "부천시 원미구",
    "31052": "부천시 소사구",
    "31053": "부천시 오정구",
    "31060": "광명시",
    "31070": "평택시",
    "31080": "동두천시",
    "31091": "안산시 상록구",
    "31092": "안산시 단원구",
    "31101": "고양시 덕양구",
    "31103": "고양시 일산동구",
    "31104": "고양시 일산서구",
    "31110": "과천시",
    "31120": "구리시",
    "31130": "남양주시",
    "31140": "오산시",
    "31150": "시흥시",
    "31160": "군포시",
    "31170": "의왕시",
    "31180": "하남시",
    "31191": "용인시 처인구",
    "31192": "용인시 기흥구",
    "31193": "용인시 수지구",
    "31200": "파주시",
    "31210": "이천시",
    "31220": "안성시",
    "31230": "김포시",
    "31240": "화성시",
    "31250": "광주시",
    "31260": "양주시",
    "31270": "포천시",
    "31280": "여주시",
    "31550": "연천군",
    "31570": "가평군",
    "31580": "양평군",
}


def read_csv_flexible(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"{path.name} 인코딩을 읽을 수 없습니다.")


@st.cache_data(show_spinner="분석 결과를 불러오는 중입니다...")
def load_grid(path: str) -> gpd.GeoDataFrame:
    grid = gpd.read_file(path)
    if grid.crs is None:
        grid = grid.set_crs("EPSG:4326")
    else:
        grid = grid.to_crs("EPSG:4326")

    required = {
        "grid_id",
        "pop80",
        "nearest_km",
        "gravity_raw",
        "access_per_1000",
        "priority_score",
    }
    missing = required.difference(grid.columns)
    if missing:
        raise ValueError(f"격자 결과에 필요한 컬럼이 없습니다: {sorted(missing)}")

    for column in required - {"grid_id"}:
        grid[column] = pd.to_numeric(grid[column], errors="coerce")
    grid["grid_id"] = grid["grid_id"].astype(str)
    if "admin_dong" in grid.columns:
        grid["admin_dong"] = grid["admin_dong"].fillna("행정동 미확인").astype(str)
    else:
        grid["admin_dong"] = "행정동 경계 미연결"
    if "adm_cd" in grid.columns:
        grid["sigungu"] = (
            grid["adm_cd"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str[:5]
            .map(SIGUNGU_BY_SGIS_PREFIX)
            .fillna("시군구 미확인")
        )
    else:
        grid["sigungu"] = "시군구 코드 미연결"
    grid["pop80_grade"] = population_grade(grid["pop80"])
    grid["gravity_grade"] = relative_access_grade(grid["gravity_raw"])
    grid["competitive_grade"] = relative_access_grade(grid["access_per_1000"])
    grid["urgency_grade"] = urgency_grade(grid["priority_score"])
    return grid


@st.cache_data(show_spinner=False)
def load_facilities(path: str) -> pd.DataFrame:
    raw = read_csv_flexible(Path(path))
    aliases = {
        "facility_name": ["facility_name", "장례시설명"],
        "city": ["city", "시군명"],
        "facility_type": ["facility_type", "장례시설유형"],
        "rooms": ["rooms", "빈소수"],
        "capacity": ["capacity", "안치능력수"],
        "latitude": ["latitude", "WGS84위도"],
        "longitude": ["longitude", "WGS84경도"],
    }
    normalized = pd.DataFrame(index=raw.index)
    for target, candidates in aliases.items():
        source = next((name for name in candidates if name in raw.columns), None)
        normalized[target] = raw[source] if source else np.nan

    for column in ["rooms", "capacity", "latitude", "longitude"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized["facility_name"] = normalized["facility_name"].fillna("시설명 없음")
    normalized["city"] = normalized["city"].fillna("미분류")
    normalized["facility_type"] = normalized["facility_type"].fillna("미분류")
    return normalized.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)


def find_facility_file() -> Path | None:
    return next((path for path in FACILITY_CANDIDATES if path.exists()), None)


def population_grade(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").clip(lower=0)
    return pd.cut(
        numeric,
        bins=[-0.1, 0, 24, 49, 99, np.inf],
        labels=POPULATION_LABELS,
        include_lowest=True,
    ).astype("string")


def relative_access_grade(values: pd.Series) -> pd.Series:
    """경기도 전체 격자의 순위를 고정된 5분위 상대등급으로 변환합니다."""
    numeric = pd.to_numeric(values, errors="coerce")
    percentile = numeric.rank(method="average", pct=True)
    return pd.cut(
        percentile,
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=ACCESS_LABELS,
        include_lowest=True,
    ).astype("string")


def urgency_grade(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").clip(0, 1)
    return pd.cut(
        numeric,
        bins=URGENCY_BINS,
        labels=URGENCY_LABELS,
        include_lowest=True,
        right=True,
    ).astype("string")


def assign_category_colors(
    values: pd.Series, labels: list[str], palette: list[str], alpha: int
) -> list[list[int]]:
    rgb = [[int(color[i : i + 2], 16) for i in (1, 3, 5)] for color in palette]
    color_by_label = {label: [*color, alpha] for label, color in zip(labels, rgb)}
    return [color_by_label.get(str(value), [160, 160, 160, 80]) for value in values]


def category_legend_html(labels: list[str], palette: list[str]) -> str:
    items = [
        "<span style='display:inline-flex;align-items:center;margin-right:14px'>"
        "<i style='width:11px;height:11px;background:#dc2626;border-radius:50%;"
        "display:inline-block;margin-right:4px'></i>장례시설</span>"
    ]
    for label, color in zip(labels, palette):
        items.append(
            f"<span style='display:inline-flex;align-items:center;margin-right:14px'>"
            f"<i style='width:13px;height:13px;background:{color};display:inline-block;margin-right:4px'></i>"
            f"{label}</span>"
        )
    return "".join(items)


def format_metric(value: float, digits: int = 0) -> str:
    if pd.isna(value):
        return "–"
    return f"{value:,.{digits}f}"


def section_heading(title: str) -> None:
    st.markdown(f'<h3 class="section-heading">{title}</h3>', unsafe_allow_html=True)


st.title("경기도 장례시설 접근성 및 개선 시급지역 분석")
st.markdown(
    "본 대시보드는 2024년을 기준으로 경기도를 1km 격자로 구분하고, 격자별 80세 이상 "
    "인구를 장례시설의 잠재수요로 산정하여 지역별 접근성을 분석합니다. 초고령사회 진입에 "
    "따른 장례서비스 수요 증가에 대비해 시설 공급과 잠재수요의 공간적 불균형을 파악했습니다. "
    "이를 바탕으로 장례서비스 개선이 우선적으로 필요한 지역을 제시하고자 합니다."
)
with st.expander("접근성 분석 방법", expanded=False):
    st.markdown(
        "접근성은 장례시설의 공급량과 격자별 80세 이상 인구를 결합한 중력모델로 "
        "산출했습니다. 반감거리 5km의 거리감쇠 함수 $w(d)=2^{-d/5}$를 적용하여 가까운 "
        "시설일수록 크게 반영하고, 각 시설의 공급을 주변 잠재수요가 나누어 이용하는 것으로 "
        "계산했습니다. 접근성 값이 클수록 잠재수요에 비해 이용 가능한 장례시설이 많고 "
        "가까운 지역임을 의미합니다."
    )
st.markdown('<hr class="header-divider">', unsafe_allow_html=True)

facility_path = find_facility_file()
missing_files = [path for path in [GRID_PATH] if not path.exists()]
if facility_path is None:
    missing_files.append(FACILITY_CANDIDATES[0])

if missing_files:
    st.warning("대시보드를 실행하려면 분석 결과 파일을 data 폴더에 넣어주세요.")
    st.code(
        "gyeonggi_funeral_dashboard/data/\n"
        "├── gyeonggi_funeral_access_grid.geojson\n"
        "└── facilities.csv  # 또는 장례시설현황.csv",
        language="text",
    )
    st.write("현재 없는 파일:")
    for path in missing_files:
        st.write(f"- `{path.name}`")
    st.stop()

try:
    grid = load_grid(str(GRID_PATH))
    facilities = load_facilities(str(facility_path))
except Exception as exc:
    st.error(f"데이터를 불러오지 못했습니다: {exc}")
    st.stop()

st.sidebar.header("지도 설정")
selected_metric = st.sidebar.radio("지도에 표시할 지표", list(METRICS))
metric_config = METRICS[selected_metric]
st.sidebar.caption("장례시설은 모든 지도에 빨간 점으로 항상 표시됩니다.")
metric_column = metric_config["column"]
grade_column = metric_config["grade_column"]

total_pop80 = grid["pop80"].sum()
total_capacity = facilities["capacity"].fillna(0).sum()
weighted_access = np.average(
    grid["access_per_1000"].fillna(0),
    weights=grid["pop80"].fillna(0) + 1e-9,
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("80세 이상 인구", f"{total_pop80:,.0f}명")
kpi2.metric("장례시설", f"{len(facilities):,}개")
kpi3.metric("안치능력", f"{total_capacity:,.0f}구")
kpi4.metric("인구가중 평균 접근성", format_metric(weighted_access, 3))

display_grid = grid.copy()
display_grid["fill_color"] = assign_category_colors(
    display_grid[grade_column],
    metric_config["labels"],
    metric_config["palette"],
    metric_config["alpha"],
)
display_grid["pop80_label"] = display_grid["pop80"].round(0).fillna(0).astype(int)
display_grid["gravity_label"] = display_grid["gravity_raw"].round(3)
display_grid["nearest_label"] = display_grid["nearest_km"].round(2)
display_grid["access_label"] = display_grid["access_per_1000"].round(4)
display_grid["priority_label"] = display_grid["priority_score"].round(4)
display_grid["display_name"] = (
    display_grid["sigungu"]
    + " "
    + display_grid["admin_dong"]
    + " · 격자 "
    + display_grid["grid_id"]
)
display_grid["detail_1"] = (
    "80세 이상 " + display_grid["pop80_label"].astype(str) + "명"
    + " (" + display_grid["pop80_grade"].fillna("분류 없음") + ")"
)
display_grid["detail_2"] = (
    "주변 접근성 " + display_grid["gravity_label"].astype(str)
    + " (" + display_grid["gravity_grade"].fillna("분류 없음") + ")"
)
display_grid["detail_3"] = (
    "수요 대비 접근성 " + display_grid["access_label"].astype(str)
    + " (" + display_grid["competitive_grade"].fillna("분류 없음") + ")"
)
display_grid["detail_4"] = (
    "개선 시급성 " + display_grid["priority_label"].astype(str)
    + " (" + display_grid["urgency_grade"].fillna("분류 없음") + ")"
)

minx, miny, maxx, maxy = display_grid.total_bounds
view_state = pdk.ViewState(
    latitude=(miny + maxy) / 2,
    longitude=(minx + maxx) / 2,
    zoom=8.0,
    pitch=0,
)

layers: list[pdk.Layer] = [
    pdk.Layer(
        "GeoJsonLayer",
        data=display_grid.__geo_interface__,
        get_fill_color="properties.fill_color",
        get_line_color=[255, 255, 255, 60],
        line_width_min_pixels=0.25,
        pickable=True,
        stroked=True,
        filled=True,
    )
]

# 시설 레이어를 항상 마지막에 추가하여 격자 레이어 위에 표시
facility_layer_data = facilities.copy()
facility_layer_data["display_name"] = facility_layer_data["facility_name"]
facility_layer_data["detail_1"] = (
    facility_layer_data["city"] + " · " + facility_layer_data["facility_type"]
)
facility_layer_data["detail_2"] = (
    "빈소 "
    + facility_layer_data["rooms"].fillna(0).round(0).astype(int).astype(str)
    + "개"
)
facility_layer_data["detail_3"] = (
    "안치능력 "
    + facility_layer_data["capacity"].fillna(0).round(0).astype(int).astype(str)
    + "명"
)
facility_layer_data["detail_4"] = ""
layers.append(
    pdk.Layer(
        "ScatterplotLayer",
        data=facility_layer_data,
        get_position="[longitude, latitude]",
        get_radius=260,
        radius_min_pixels=3,
        radius_max_pixels=9,
        get_fill_color=[220, 38, 38, 230],
        get_line_color=[255, 255, 255, 230],
        line_width_min_pixels=1,
        pickable=True,
        stroked=True,
        parameters={"depthTest": False},
    )
)

tooltip = {
    "html": (
        "<b>{display_name}</b><br/>"
        "{detail_1}<br/>"
        "{detail_2}<br/>"
        "{detail_3}<br/>"
        "{detail_4}"
    ),
    "style": {"backgroundColor": "#172033", "color": "white"},
}

st.markdown(f'<div class="map-title-box">{selected_metric}</div>', unsafe_allow_html=True)
st.caption(metric_config["direction"])
st.pydeck_chart(
    pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="light",
        tooltip=tooltip,
    ),
    width="stretch",
    height=650,
)
legend = category_legend_html(metric_config["labels"], metric_config["palette"])
st.markdown(legend, unsafe_allow_html=True)
if metric_column == "access_per_1000":
    st.caption("접근성 등급은 경기도 전체 격자를 기준으로 나눈 5분위 상대등급입니다.")

st.divider()
chart_left, chart_right = st.columns([1, 1])

with chart_left:
    section_heading("잠재수요와 접근성")
    scatter_data = grid.loc[
        grid["pop80"] > 0,
        [
            "grid_id",
            "sigungu",
            "admin_dong",
            "pop80",
            "access_per_1000",
            "nearest_km",
            "priority_score",
            "urgency_grade",
        ],
    ].copy()
    scatter_data["location_label"] = (
        scatter_data["sigungu"] + " " + scatter_data["admin_dong"]
    )
    population_cut = scatter_data["pop80"].quantile(0.75)
    access_cut = scatter_data["access_per_1000"].quantile(0.25)
    top_priority = scatter_data.nlargest(20, "priority_score").copy()
    urgency_color_map = dict(
        zip(
            URGENCY_LABELS,
            ["#fde68a", "#fbbf24", "#fb923c", "#ef4444", "#7f1d1d"],
        )
    )
    fig = px.scatter(
        scatter_data,
        x="pop80",
        y="access_per_1000",
        color="urgency_grade",
        hover_name="location_label",
        hover_data={
            "grid_id": False,
            "sigungu": False,
            "admin_dong": False,
            "pop80": ":,.0f",
            "access_per_1000": ":.3f",
            "priority_score": ":.3f",
            "urgency_grade": False,
            "nearest_km": False,
        },
        category_orders={"urgency_grade": URGENCY_LABELS},
        color_discrete_map=urgency_color_map,
        opacity=0.42,
        log_x=True,
        render_mode="webgl",
        labels={
            "pop80": "80세 이상 인구(명, 로그 척도)",
            "access_per_1000": "수요 대비 장례시설 접근성",
            "priority_score": "개선 시급성 점수",
            "urgency_grade": "개선 시급성 등급",
        },
    )
    fig.update_traces(
        marker={"size": 6, "line": {"color": "rgba(17, 24, 39, 0.25)", "width": 0.3}}
    )
    fig.add_shape(
        type="rect",
        xref="x",
        yref="y",
        x0=population_cut,
        x1=scatter_data["pop80"].max() * 1.1,
        y0=max(0, scatter_data["access_per_1000"].min()),
        y1=access_cut,
        fillcolor="rgba(220, 38, 38, 0.10)",
        line_width=0,
        layer="below",
    )
    fig.add_vline(x=population_cut, line_dash="dash", line_color="#6b7280")
    fig.add_hline(y=access_cut, line_dash="dash", line_color="#6b7280")
    fig.add_annotation(
        x=scatter_data["pop80"].quantile(0.90),
        y=scatter_data["access_per_1000"].quantile(0.07),
        text="개선 우선 검토영역",
        showarrow=False,
        font={"color": "#991b1b", "size": 12},
        bgcolor="rgba(255, 255, 255, 0.78)",
    )
    fig.add_scatter(
        x=top_priority["pop80"],
        y=top_priority["access_per_1000"],
        mode="markers",
        name="시급성 상위 20개",
        marker={
            "size": 11,
            "color": "rgba(255, 255, 255, 0)",
            "line": {"color": "#dc2626", "width": 2.2},
        },
        customdata=top_priority[["location_label", "priority_score"]],
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "80세 이상 인구 %{x:,.0f}명<br>"
            "수요 대비 접근성 %{y:.3f}<br>"
            "개선 시급성 %{customdata[1]:.3f}"
            "<extra>시급성 상위 20개</extra>"
        ),
    )
    fig.update_xaxes(
        tickvals=[5, 10, 25, 50, 100, 250, 500, 1000, 2000],
        ticktext=["5", "10", "25", "50", "100", "250", "500", "1,000", "2,000"],
        gridcolor="#e5e7eb",
    )
    fig.update_yaxes(gridcolor="#e5e7eb", rangemode="tozero")
    fig.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#ffffff",
        legend_title_text="개선 시급성 등급",
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "점선은 고령인구 상위 25%와 접근성 하위 25%의 기준입니다. "
        "붉은 영역은 개선 우선 검토영역이며, 빨간 테두리는 시급성 상위 20개 격자입니다."
    )

with chart_right:
    section_heading("장례서비스 개선 시급지역")
    top_n = st.select_slider("표시 개수", options=[10, 20, 30, 50], value=20)
    ranking = grid.nlargest(top_n, "priority_score").copy()
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking["location_label"] = (
        ranking["rank"].astype(str)
        + ". "
        + ranking["sigungu"]
        + " "
        + ranking["admin_dong"]
    )
    ranking = ranking.sort_values("priority_score")
    fig = px.bar(
        ranking,
        x="priority_score",
        y="location_label",
        orientation="h",
        color="pop80",
        color_continuous_scale="OrRd",
        labels={
            "priority_score": "장례서비스 개선 시급성",
            "location_label": "시군구 · 행정동",
            "pop80": "80세 이상 인구",
        },
    )
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")

section_heading("장례서비스 개선 시급지역 상세표 (상위 20개)")
table_columns = [
    "sigungu",
    "admin_dong",
    "pop80",
    "nearest_km",
    "access_per_1000",
    "priority_score",
]
table = grid.nlargest(20, "priority_score")[table_columns].copy()
table.insert(0, "rank", np.arange(1, len(table) + 1))
table["urgency_grade"] = urgency_grade(table["priority_score"])
table = table.rename(
    columns={
        "rank": "순위",
        "sigungu": "시군구",
        "admin_dong": "행정동",
        "pop80": "80세 이상 인구",
        "nearest_km": "최근접 시설 거리(km)",
        "access_per_1000": "수요 대비 장례시설 접근성",
        "priority_score": "장례서비스 개선 시급성",
        "urgency_grade": "시급성 등급",
    }
)
st.dataframe(
    table,
    width="stretch",
    hide_index=True,
    column_config={
        "80세 이상 인구": st.column_config.NumberColumn(format="%,.0f명"),
        "최근접 시설 거리(km)": st.column_config.NumberColumn(format="%.2f km"),
        "수요 대비 장례시설 접근성": st.column_config.NumberColumn(format="%.4f"),
        "장례서비스 개선 시급성": st.column_config.ProgressColumn(
            min_value=0, max_value=1, format="%.3f"
        ),
    },
)
st.download_button(
    "현재 개선 시급지역 표 CSV 다운로드",
    data=table.to_csv(index=False, encoding="utf-8-sig"),
    file_name="gyeonggi_funeral_urgency_grids.csv",
    mime="text/csv",
)

with st.expander("분석 방법과 해석상 주의사항"):
    st.markdown(
        """
        - **80세 이상 인구:** SGIS 2024년 1 km 격자의 80세 이상 거주인구입니다.
        - **수요 대비 장례시설 접근성:** 한 시설의 공급을 주변 80세 이상 잠재수요가 나누어 이용한다고 가정합니다.
        - **장례서비스 개선 시급성:** 80세 이상 인구가 많고 수요 대비 접근성이 낮을수록 높아지는 상대평가 점수입니다.
        - **장례시설:** 모든 지표 지도에 빨간 점으로 항상 표시되며 격자 레이어 위에 위치합니다.
        - **행정동:** SGIS 읍면동 경계와 격자 대표점을 공간 결합한 결과입니다.
        - **인구 범례:** 0명, 25명 미만, 50명 미만, 100명 미만, 100명 이상의 고정 구간입니다.
        - **접근성 범례:** 경기도 전체 격자의 5분위 상대등급인 매우 불량, 불량, 보통, 양호, 매우 양호입니다.
        - **시급성 범례:** 0.2 간격으로 매우 낮음, 낮음, 보통, 높음, 매우 높음으로 구분합니다.
        - **공간적 범위:** 서울·인천 등 경기도 인접 지역의 장례시설과 잠재수요 정보가 포함되지 않아 분석 정확성에 한계가 있습니다.

        직선거리는 실제 도로 이동시간과 다르며, 80세 이상 인구는 실제 사망자 수가 아닌
        장례서비스 잠재수요의 대리변수입니다. 경기도 밖 인접 시설이 제외된 경우 도 경계지역의
        접근성이 과소평가될 수 있습니다.
        """
    )

st.caption(
    "자료: SGIS 2024년 1 km 격자 인구통계 · SGIS 2025년 경기도·행정동 경계 · 경기도 장례시설현황 2024년"
)
