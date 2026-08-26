from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st


st.set_page_config(
    page_title="경기도 장례시설 접근성",
    page_icon="🗺️",
    layout="wide",
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
        "direction": "많을수록 잠재수요가 큼",
    },
    "수요 대비 장례시설 접근성": {
        "column": "access_per_1000",
        "grade_column": "competitive_grade",
        # Colab의 cool: 매우 불량(cyan) → 매우 양호(magenta)
        "palette": ["#14ebff", "#4ab5ff", "#807fff", "#b54aff", "#eb14ff"],
        "labels": ["매우 불량", "불량", "보통", "양호", "매우 양호"],
        "alpha": 191,  # 75%
        "direction": "주변의 80세 이상 잠재수요와 시설 공급을 함께 고려하며, 클수록 접근성이 좋음",
    },
    "장례서비스 개선 시급성": {
        "column": "priority_score",
        "grade_column": "urgency_grade",
        # Colab의 afmhot 역방향(매우 낮음 → 매우 높음)
        "palette": ["#ffffd7", "#ffeb6b", "#ff8001", "#941400", "#280000"],
        "labels": ["매우 낮음", "낮음", "보통", "높음", "매우 높음"],
        "alpha": 184,  # 72%
        "direction": "80세 이상 인구가 많고 수요 대비 접근성이 낮을수록 높음",
    },
}

POPULATION_LABELS = ["0명", "25명 미만", "50명 미만", "100명 미만", "100명 이상"]
ACCESS_LABELS = ["매우 불량", "불량", "보통", "양호", "매우 양호"]
URGENCY_LABELS = ["매우 낮음", "낮음", "보통", "높음", "매우 높음"]
URGENCY_BINS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


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
        "display:inline-block;margin-right:4px'></i>장례시설(항상 표시)</span>"
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


st.title("경기도 장례시설 접근성 대시보드")
st.caption("2024년 1 km 격자 80세 이상 인구와 장례시설 공급을 결합한 중력모델 결과")

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
selected_metric = st.sidebar.selectbox("표시 지표", list(METRICS))
metric_config = METRICS[selected_metric]
st.sidebar.caption("장례시설은 모든 지도에 빨간 점으로 항상 표시됩니다.")
metric_column = metric_config["column"]
grade_column = metric_config["grade_column"]

positive_population = grid.loc[grid["pop80"] > 0, "pop80"]
max_population = int(max(1, positive_population.max())) if not positive_population.empty else 1
population_threshold = st.sidebar.slider(
    "최소 80세 이상 인구",
    min_value=0,
    max_value=max_population,
    value=0,
    step=max(1, max_population // 100),
)
show_only_priority = st.sidebar.checkbox("개선 시급지역만 표시", value=False)
priority_max = max(1, min(200, len(grid)))
priority_min = 10 if priority_max >= 10 else 1
priority_default = min(30, priority_max)
priority_step = 10 if priority_max >= 20 else 1
if priority_min < priority_max:
    priority_count = st.sidebar.slider(
        "시급지역 개수",
        min_value=priority_min,
        max_value=priority_max,
        value=priority_default,
        step=priority_step,
    )
else:
    priority_count = priority_max
    st.sidebar.caption(f"시급지역 개수: {priority_count}개")

filtered = grid.loc[grid["pop80"].fillna(0) >= population_threshold].copy()
if show_only_priority:
    filtered = filtered.nlargest(priority_count, "priority_score")

if filtered.empty:
    st.info("현재 조건에 맞는 격자가 없습니다. 최소 인구 조건을 낮춰주세요.")
    st.stop()

total_pop80 = filtered["pop80"].sum()
weighted_access = np.average(
    filtered["access_per_1000"].fillna(0),
    weights=filtered["pop80"].fillna(0) + 1e-9,
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("표시 격자", f"{len(filtered):,}개")
kpi2.metric("80세 이상 인구", f"{total_pop80:,.0f}명")
kpi3.metric("장례시설", f"{len(facilities):,}개")
kpi4.metric("인구가중 평균 접근성", format_metric(weighted_access, 3))

display_grid = filtered.copy()
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
    display_grid["admin_dong"] + " · 격자 " + display_grid["grid_id"]
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

st.subheader(selected_metric)
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
    st.subheader("잠재수요와 접근성")
    scatter_data = filtered.loc[
        filtered["pop80"] > 0,
        [
            "grid_id",
            "admin_dong",
            "pop80",
            "access_per_1000",
            "nearest_km",
            "priority_score",
        ],
    ].copy()
    fig = px.scatter(
        scatter_data,
        x="pop80",
        y="access_per_1000",
        color="priority_score",
        size="nearest_km",
        hover_name="admin_dong",
        hover_data={"grid_id": True},
        color_continuous_scale="YlOrRd",
        labels={
            "pop80": "80세 이상 인구(명)",
            "access_per_1000": "수요 대비 장례시설 접근성",
            "priority_score": "장례서비스 개선 시급성",
            "nearest_km": "최근접 거리(km)",
        },
    )
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")

with chart_right:
    st.subheader("장례서비스 개선 시급지역")
    top_n = st.select_slider("표시 개수", options=[10, 20, 30, 50], value=20)
    ranking = filtered.nlargest(top_n, "priority_score").sort_values("priority_score")
    ranking["location_label"] = ranking["admin_dong"] + " · " + ranking["grid_id"]
    fig = px.bar(
        ranking,
        x="priority_score",
        y="location_label",
        orientation="h",
        color="pop80",
        color_continuous_scale="OrRd",
        labels={
            "priority_score": "장례서비스 개선 시급성",
            "location_label": "행정동 · 격자코드",
            "pop80": "80세 이상 인구",
        },
    )
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")

st.subheader("장례서비스 개선 시급지역 상세표")
table_columns = [
    "admin_dong",
    "grid_id",
    "pop80",
    "nearest_km",
    "access_per_1000",
    "priority_score",
]
table = filtered.nlargest(priority_count, "priority_score")[table_columns].copy()
table["urgency_grade"] = urgency_grade(table["priority_score"])
table = table.rename(
    columns={
        "admin_dong": "행정동",
        "grid_id": "격자코드",
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

        직선거리는 실제 도로 이동시간과 다르며, 80세 이상 인구는 실제 사망자 수가 아닌
        장례서비스 잠재수요의 대리변수입니다. 경기도 밖 인접 시설이 제외된 경우 도 경계지역의
        접근성이 과소평가될 수 있습니다.
        """
    )

st.caption(
    "자료: SGIS 2024년 1 km 격자 인구통계 · SGIS 2025년 경기도·행정동 경계 · 경기도 장례시설현황 2024년"
)
