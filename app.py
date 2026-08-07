import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

st.set_page_config(
    page_title="경기도 장례서비스 공간 접근성",
    page_icon="🕯️",
    layout="wide",
)

API_URL = "https://openapi.gg.go.kr/FuneralFacilityStatus"
DATA_DIR = Path(__file__).parent / "data"
DEATHS_PATH = DATA_DIR / "deaths_by_sigun.csv"
GEOJSON_PATH = DATA_DIR / "gyeonggi_sigun.geojson"

SIGUN_CENTERS = {
    "수원시": (127.0286, 37.2636), "성남시": (127.1262, 37.4200), "고양시": (126.8320, 37.6584),
    "용인시": (127.1775, 37.2411), "부천시": (126.7660, 37.5034), "안산시": (126.8309, 37.3219),
    "안양시": (126.9568, 37.3943), "남양주시": (127.2165, 37.6360), "화성시": (126.8312, 37.1995),
    "평택시": (127.1127, 36.9921), "의정부시": (127.0338, 37.7381), "시흥시": (126.8029, 37.3800),
    "파주시": (126.7800, 37.7599), "광명시": (126.8644, 37.4786), "김포시": (126.7157, 37.6153),
    "군포시": (126.9352, 37.3617), "광주시": (127.2551, 37.4294), "이천시": (127.4350, 37.2720),
    "양주시": (127.0458, 37.7853), "오산시": (127.0775, 37.1498), "구리시": (127.1296, 37.5943),
    "안성시": (127.2797, 37.0080), "포천시": (127.2003, 37.8949), "의왕시": (126.9683, 37.3449),
    "하남시": (127.2147, 37.5393), "여주시": (127.6372, 37.2982), "양평군": (127.4876, 37.4917),
    "동두천시": (127.0608, 37.9035), "과천시": (126.9876, 37.4292), "가평군": (127.5096, 37.8315),
    "연천군": (127.0750, 38.0964),
}


def get_secret(name: str, default=""):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return os.getenv(name, default)


def first_existing(columns, candidates):
    lookup = {str(c).upper(): c for c in columns}
    for candidate in candidates:
        if candidate.upper() in lookup:
            return lookup[candidate.upper()]
    return None


def flatten_gg_rows(payload):
    if isinstance(payload, list):
        for block in payload:
            if isinstance(block, dict) and isinstance(block.get("row"), list):
                return block["row"]
            if isinstance(block, dict):
                for value in block.values():
                    rows = flatten_gg_rows(value)
                    if rows:
                        return rows
    elif isinstance(payload, dict):
        if isinstance(payload.get("row"), list):
            return payload["row"]
        for value in payload.values():
            rows = flatten_gg_rows(value)
            if rows:
                return rows
    return []


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_facilities(api_key: str, page_size: int = 1000) -> pd.DataFrame:
    if not api_key:
        raise ValueError("GG_API_KEY가 없습니다. Streamlit Secrets에 인증키를 넣어주세요.")

    all_rows = []
    page = 1
    while True:
        params = {"KEY": api_key, "Type": "json", "pIndex": page, "pSize": page_size}
        response = requests.get(API_URL, params=params, timeout=25)
        response.raise_for_status()
        rows = flatten_gg_rows(response.json())
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        page += 1
        if page > 50:
            break

    if not all_rows:
        raise ValueError("API에서 row 데이터를 찾지 못했습니다. 인증키와 API 응답을 확인하세요.")
    return pd.DataFrame(all_rows)


def normalize_facilities(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    cols = df.columns

    name_col = first_existing(cols, ["BIZPLC_NM", "FACLT_NM", "FCLTY_NM", "NAME", "CMPNM_NM", "INST_NM"])
    sigun_col = first_existing(cols, ["SIGUN_NM", "SIGUN", "SIGUNGU_NM", "SIGNGU_NM"])
    addr_col = first_existing(cols, ["REFINE_ROADNM_ADDR", "ROADNM_ADDR", "REFINE_LOTNO_ADDR", "LOTNO_ADDR", "ADDR"])
    lat_col = first_existing(cols, ["REFINE_WGS84_LAT", "WGS84_LAT", "LAT", "LATITUDE"])
    lon_col = first_existing(cols, ["REFINE_WGS84_LOGT", "REFINE_WGS84_LON", "WGS84_LOGT", "WGS84_LON", "LON", "LONGITUDE"])
    room_col = first_existing(cols, ["FUNERAL_PARLOR_CNT", "MORTUARY_CNT", "ROOM_CNT", "BINSO_CNT"])
    capacity_col = first_existing(cols, ["CORPSE_STORGE_PSBL_CNT", "STORGE_PSBL_CNT", "CAPACITY", "ACCOMMODATE_CNT"])
    type_col = first_existing(cols, ["FACLT_DIV_NM", "FCLTY_DIV_NM", "FACILITY_TYPE", "BIZCOND_NM"])

    out = pd.DataFrame(index=df.index)
    out["facility"] = df[name_col].astype(str) if name_col else "장례시설"
    out["sigun"] = df[sigun_col].astype(str).str.strip() if sigun_col else "미상"
    out["address"] = df[addr_col].astype(str) if addr_col else ""
    out["lat"] = pd.to_numeric(df[lat_col], errors="coerce") if lat_col else np.nan
    out["lon"] = pd.to_numeric(df[lon_col], errors="coerce") if lon_col else np.nan
    out["rooms"] = pd.to_numeric(df[room_col], errors="coerce").fillna(1) if room_col else 1
    out["capacity"] = pd.to_numeric(df[capacity_col], errors="coerce") if capacity_col else np.nan
    out["facility_type"] = df[type_col].astype(str) if type_col else "전체"

    out = out[out["lat"].between(36.7, 38.4) & out["lon"].between(126.3, 127.9)].copy()
    out["rooms"] = out["rooms"].clip(lower=1)
    out["capacity"] = out["capacity"].fillna(out["rooms"]).clip(lower=1)
    out["tooltip_text"] = (
        "<b>" + out["facility"] + "</b><br/>"
        + out["sigun"]
        + "<br/>안치 수용능력 " + out["capacity"].round(0).astype(int).astype(str)
        + "<br/>" + out["address"]
    )
    return out.drop_duplicates(subset=["facility", "lat", "lon"]).reset_index(drop=True)


def haversine_km(lon1, lat1, lon2, lat2):
    radius = 6371.0088
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))


def load_deaths() -> pd.DataFrame:
    if not DEATHS_PATH.exists():
        return pd.DataFrame(columns=["sigun", "deaths", "year"])
    deaths = pd.read_csv(DEATHS_PATH)
    if not {"sigun", "deaths"}.issubset(deaths.columns):
        raise ValueError("data/deaths_by_sigun.csv에는 sigun, deaths 컬럼이 필요합니다.")
    deaths["sigun"] = deaths["sigun"].astype(str).str.strip()
    deaths["deaths"] = pd.to_numeric(deaths["deaths"], errors="coerce")
    return deaths.dropna(subset=["deaths"]).copy()


def build_region_frame(deaths: pd.DataFrame) -> pd.DataFrame:
    regions = pd.DataFrame(
        [{"sigun": name, "source_lon": xy[0], "source_lat": xy[1]} for name, xy in SIGUN_CENTERS.items()]
    )
    if not deaths.empty:
        regions = regions.merge(deaths[["sigun", "deaths"]], on="sigun", how="left")
    else:
        regions["deaths"] = np.nan
    regions["tooltip_text"] = regions.apply(
        lambda r: f"<b>{r['sigun']}</b><br/>사망자 {r['deaths']:,.0f}명" if pd.notna(r["deaths"]) else f"<b>{r['sigun']}</b>",
        axis=1,
    )
    return regions


def build_arcs(regions: pd.DataFrame, facilities: pd.DataFrame, top_n=2, alpha=1.5) -> pd.DataFrame:
    arc_rows = []
    for _, region in regions.iterrows():
        candidates = facilities.copy()
        candidates["distance_km"] = haversine_km(
            region.source_lon,
            region.source_lat,
            candidates["lon"].values,
            candidates["lat"].values,
        )
        candidates["access_score"] = candidates["capacity"] / np.power(
            np.maximum(candidates["distance_km"], 1.0), alpha
        )
        candidates = candidates.nlargest(top_n, "access_score").copy()
        total_score = candidates["access_score"].sum()
        candidates["share"] = candidates["access_score"] / total_score if total_score > 0 else 1 / len(candidates)

        for rank, (_, facility) in enumerate(candidates.iterrows(), start=1):
            deaths = region.get("deaths", np.nan)
            demand_flow = float(deaths * facility["share"]) if pd.notna(deaths) else float(facility["share"])
            arc_rows.append(
                {
                    "sigun": region.sigun,
                    "facility": facility.facility,
                    "source_lon": float(region.source_lon),
                    "source_lat": float(region.source_lat),
                    "target_lon": float(facility.lon),
                    "target_lat": float(facility.lat),
                    "distance_km": float(facility.distance_km),
                    "capacity": float(facility.capacity),
                    "rank": rank,
                    "access_score": float(facility.access_score),
                    "share": float(facility["share"]),
                    "deaths": None if pd.isna(deaths) else float(deaths),
                    "demand_flow": demand_flow,
                }
            )

    arcs = pd.DataFrame(arc_rows)
    if arcs.empty:
        return arcs

    max_capacity = max(float(facilities["capacity"].max()), 1.0)
    max_distance = max(float(arcs["distance_km"].max()), 1.0)
    opacity = arcs["rank"].map({1: 230, 2: 185, 3: 145, 4: 110, 5: 90}).fillna(85).astype(int)
    tilt = arcs["rank"].map({1: 0, 2: 12, 3: -12, 4: 21, 5: -21}).fillna(0)

    # Arc의 굵기는 목적지 장례시설의 안치 수용능력을 직접 반영합니다.
    arcs["arc_width"] = 1.2 + 6.2 * np.sqrt(arcs["capacity"] / max_capacity)
    arcs["arc_height"] = 0.65 + 1.65 * np.sqrt(arcs["distance_km"] / max_distance)
    arcs["arc_tilt"] = tilt
    arcs["source_color"] = opacity.map(lambda a: [37, 99, 235, int(a)])
    arcs["target_color"] = opacity.map(lambda a: [249, 115, 22, int(a)])
    arcs["tooltip_text"] = arcs.apply(
        lambda r: (
            f"<b>{r['sigun']} → {r['facility']}</b><br/>"
            f"거리 {r['distance_km']:.1f} km<br/>"
            f"안치 수용능력 {r['capacity']:.0f}<br/>"
            f"후보 순위 {int(r['rank'])}위<br/>"
            f"잠재 배분 {r['share'] * 100:.1f}%"
        ),
        axis=1,
    )
    return arcs


def aggregate_sigun(regions: pd.DataFrame, facilities: pd.DataFrame, arcs: pd.DataFrame) -> pd.DataFrame:
    facility_summary = facilities.groupby("sigun", as_index=False).agg(
        facilities=("facility", "count"),
        rooms=("rooms", "sum"),
        capacity=("capacity", "sum"),
    )
    nearest = arcs.groupby("sigun", as_index=False)["distance_km"].min().rename(
        columns={"distance_km": "best_access_km"}
    )
    summary = regions.merge(facility_summary, on="sigun", how="left").merge(nearest, on="sigun", how="left")
    summary[["facilities", "rooms", "capacity"]] = summary[["facilities", "rooms", "capacity"]].fillna(0)
    summary["pressure"] = np.where(
        summary["deaths"].notna(),
        summary["deaths"] / summary["capacity"].replace(0, np.nan),
        np.nan,
    )
    return summary


def load_colored_geojson(summary: pd.DataFrame):
    if not GEOJSON_PATH.exists():
        return None

    with open(GEOJSON_PATH, "r", encoding="utf-8") as file:
        geojson = json.load(file)

    value_map = summary.set_index("sigun").to_dict("index")
    values = summary["pressure"].dropna()
    if values.empty:
        values = summary["deaths"].dropna()
    vmin = float(values.min()) if not values.empty else 0.0
    vmax = float(values.max()) if not values.empty else 1.0

    name_keys = ["SIG_KOR_NM", "SIGUN_NM", "sggnm", "name", "NAME", "SIGUN"]
    for feature in geojson.get("features", []):
        props = feature.setdefault("properties", {})
        sigun = next((str(props[key]).strip() for key in name_keys if props.get(key)), None)
        row = value_map.get(sigun, {}) if sigun else {}
        metric = row.get("pressure")
        if metric is None or pd.isna(metric):
            metric = row.get("deaths", 0)
        metric = float(metric or 0)
        t = 0 if vmax <= vmin else np.clip((metric - vmin) / (vmax - vmin), 0, 1)
        props["fill_color"] = [int(45 + 210 * t), int(130 - 75 * t), int(220 - 155 * t), 75]
        props["tooltip_text"] = f"<b>{sigun or '미상'}</b><br/>공급압력 {metric:.1f}"
    return geojson


st.markdown(
    """
    <style>
    .block-container {padding-top: 1.45rem; padding-bottom: 2rem; max-width: 1500px;}
    .hero-kicker {color:#2563eb; font-size:.78rem; font-weight:800; letter-spacing:.16em;}
    .hero-year {display:inline-flex; padding:.28rem .68rem; border-radius:999px; background:#eff6ff; color:#1d4ed8; font-weight:700; font-size:.82rem;}
    [data-testid="stMetric"] {background:linear-gradient(145deg,#ffffff,#f8fafc); border:1px solid #e2e8f0; border-radius:16px; padding:1rem 1.1rem; box-shadow:0 8px 24px rgba(15,23,42,.05);}
    .map-legend {display:flex; flex-wrap:wrap; gap:1rem; color:#475569; font-size:.88rem; margin:.2rem 0 .7rem;}
    .legend-dot {display:inline-block; width:.7rem; height:.7rem; border-radius:50%; margin-right:.35rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="hero-kicker">GYEONGGI FUNERAL INFRASTRUCTURE</div>', unsafe_allow_html=True)
st.title("경기도 장례 인프라 공간 접근성 대시보드")
st.markdown('<span class="hero-year">● KOSIS 2024</span>', unsafe_allow_html=True)
st.caption("31개 시군의 사망 수요와 경기도 장례시설 공급을 ArcLayer로 연결해 잠재 접근성을 탐색합니다.")

api_key = get_secret("GG_API_KEY")

with st.sidebar:
    st.header("ArcLayer 설정")
    view_mode = st.radio("Arc 표시 범위", ["경기도 전체", "선택 지역"], index=0)
    top_n = st.slider("시군별 연결 시설 수", 1, 5, 2)
    alpha = st.slider("거리 감쇠 α", 0.5, 3.0, 1.5, 0.1)
    arc_scale = st.slider("Arc 굵기 배율", 0.6, 2.2, 1.1, 0.05)
    arc_height_scale = st.slider("Arc 높이", 0.5, 2.5, 1.25, 0.05)
    point_scale = st.slider("시설 점 크기 배율", 0.6, 2.2, 1.0, 0.05)

    sigun_options = list(SIGUN_CENTERS.keys())
    if view_mode == "경기도 전체":
        focus_sigun = st.selectbox("전체 지도에서 강조할 지역", ["강조 없음"] + sigun_options, index=0)
        selected_sigun = None
    else:
        selected_sigun = st.selectbox("지역 선택", sigun_options, index=sigun_options.index("양평군"))
        focus_sigun = selected_sigun

    show_region_labels = st.toggle("시군명 표시", value=view_mode == "선택 지역")
    st.divider()
    st.markdown("**지도 읽는 법**")
    st.caption("주황색 점 크기 = 안치 수용능력, Arc 굵기 = 목적지 시설 수용능력, Arc 높이 = 연결거리")
    st.latex(r"Score_{ij}=Capacity_j / Distance_{ij}^{\alpha}")

try:
    raw = fetch_facilities(api_key)
    facilities = normalize_facilities(raw)
except Exception as exc:
    st.error(f"장례시설 API를 불러오지 못했습니다: {exc}")
    st.info('Streamlit App settings → Secrets에 `GG_API_KEY = "발급받은키"`를 설정하세요.')
    st.stop()

if facilities.empty:
    st.warning("유효한 WGS84 좌표가 있는 장례시설을 찾지 못했습니다.")
    st.stop()

deaths = load_deaths()
regions = build_region_frame(deaths)
arcs = build_arcs(regions, facilities, top_n=top_n, alpha=alpha)
summary = aggregate_sigun(regions, facilities, arcs)
colored_geojson = load_colored_geojson(summary)

year = int(deaths["year"].dropna().iloc[0]) if "year" in deaths.columns and deaths["year"].notna().any() else 2024
average_distance = arcs["distance_km"].mean() if not arcs.empty else np.nan

m1, m2, m3, m4 = st.columns(4)
m1.metric(f"{year} 사망자수", f"{deaths['deaths'].sum():,.0f}명" if not deaths.empty else "-")
m2.metric("장례시설 수", f"{len(facilities):,}개")
m3.metric("총 안치 수용능력", f"{facilities['capacity'].sum():,.0f}")
m4.metric("평균 연결거리", f"{average_distance:,.1f} km" if pd.notna(average_distance) else "-")

if view_mode == "경기도 전체":
    map_arcs = arcs.copy()
    map_regions = regions.copy()
    map_facilities = facilities.copy()
    view = pdk.ViewState(latitude=37.48, longitude=127.08, zoom=7.65, pitch=50, bearing=-7)
else:
    map_arcs = arcs[arcs["sigun"] == selected_sigun].copy()
    map_regions = regions[regions["sigun"] == selected_sigun].copy()
    target_coords = map_arcs[["target_lon", "target_lat"]].drop_duplicates()
    map_facilities = facilities.merge(
        target_coords,
        left_on=["lon", "lat"],
        right_on=["target_lon", "target_lat"],
        how="inner",
    ).drop(columns=["target_lon", "target_lat"])
    center_lon = pd.concat([map_regions["source_lon"], map_arcs["target_lon"]]).mean()
    center_lat = pd.concat([map_regions["source_lat"], map_arcs["target_lat"]]).mean()
    view = pdk.ViewState(latitude=float(center_lat), longitude=float(center_lon), zoom=9.25, pitch=52, bearing=-8)

max_capacity = max(float(facilities["capacity"].max()), 1.0)
map_facilities["point_radius_display"] = (
    4.5 + 15.5 * np.sqrt(map_facilities["capacity"] / max_capacity)
) * point_scale
map_arcs["height_display"] = map_arcs["arc_height"] * arc_height_scale
map_arcs["width_display"] = map_arcs["arc_width"] * arc_scale

layers = []

if colored_geojson is not None:
    layers.append(
        pdk.Layer(
            "GeoJsonLayer",
            id="sigun-polygons",
            data=colored_geojson,
            filled=True,
            stroked=True,
            get_fill_color="properties.fill_color",
            get_line_color=[100, 116, 139, 95],
            line_width_min_pixels=0.6,
            pickable=True,
        )
    )

if not map_arcs.empty:
    layers.append(
        pdk.Layer(
            "ArcLayer",
            id="all-access-arcs",
            data=map_arcs,
            get_source_position="[source_lon, source_lat]",
            get_target_position="[target_lon, target_lat]",
            get_source_color="source_color",
            get_target_color="target_color",
            get_width="width_display",
            get_height="height_display",
            get_tilt="arc_tilt",
            width_units="pixels",
            width_min_pixels=1.0,
            width_max_pixels=10.0 if view_mode == "경기도 전체" else 13.0,
            pickable=True,
            auto_highlight=True,
        )
    )

if view_mode == "경기도 전체" and focus_sigun != "강조 없음":
    focus_arcs = arcs[arcs["sigun"] == focus_sigun].copy()
    focus_arcs["width_focus"] = focus_arcs["arc_width"] * arc_scale * 1.65
    focus_arcs["height_focus"] = focus_arcs["arc_height"] * arc_height_scale * 1.12
    focus_arcs["source_focus"] = [[29, 78, 216, 255]] * len(focus_arcs)
    focus_arcs["target_focus"] = [[234, 88, 12, 255]] * len(focus_arcs)
    layers.append(
        pdk.Layer(
            "ArcLayer",
            id="focused-access-arcs",
            data=focus_arcs,
            get_source_position="[source_lon, source_lat]",
            get_target_position="[target_lon, target_lat]",
            get_source_color="source_focus",
            get_target_color="target_focus",
            get_width="width_focus",
            get_height="height_focus",
            get_tilt="arc_tilt",
            width_units="pixels",
            width_min_pixels=2.2,
            width_max_pixels=15,
            pickable=True,
        )
    )

# 장례시설은 3D 기둥 대신 수용능력에 비례하는 원형 점으로 표현합니다.
layers.append(
    pdk.Layer(
        "ScatterplotLayer",
        id="facility-points",
        data=map_facilities,
        get_position="[lon, lat]",
        get_radius="point_radius_display",
        radius_units="pixels",
        radius_min_pixels=4,
        radius_max_pixels=24,
        get_fill_color=[249, 115, 22, 205],
        get_line_color=[154, 52, 18, 245],
        line_width_min_pixels=1.5,
        stroked=True,
        pickable=True,
        auto_highlight=True,
    )
)

layers.append(
    pdk.Layer(
        "ScatterplotLayer",
        id="region-points",
        data=map_regions,
        get_position="[source_lon, source_lat]",
        get_radius=390 if view_mode == "경기도 전체" else 760,
        get_fill_color=[37, 99, 235, 235],
        get_line_color=[239, 246, 255, 255],
        line_width_min_pixels=2,
        stroked=True,
        pickable=True,
    )
)

if show_region_labels:
    layers.append(
        pdk.Layer(
            "TextLayer",
            id="region-labels",
            data=map_regions,
            get_position="[source_lon, source_lat]",
            get_text="sigun",
            get_size=11 if view_mode == "경기도 전체" else 16,
            size_units="pixels",
            get_color=[15, 23, 42, 240],
            get_pixel_offset=[0, -18],
            get_text_anchor="'middle'",
            get_alignment_baseline="'bottom'",
        )
    )

tooltip = {
    "html": "{tooltip_text}",
    "style": {"backgroundColor": "#0f172a", "color": "white", "borderRadius": "10px"},
}
deck = pdk.Deck(layers=layers, initial_view_state=view, tooltip=tooltip, map_style=None)

st.subheader("경기도 전체 지역 → 장례시설 잠재 연결망" if view_mode == "경기도 전체" else f"{selected_sigun} → 장례시설 잠재 연결망")
st.markdown(
    '<div class="map-legend">'
    '<span><i class="legend-dot" style="background:#2563eb"></i>시군 수요 지점</span>'
    '<span>Arc: 수요 → 시설</span>'
    '<span><i class="legend-dot" style="background:#f97316"></i>장례시설 · 점 크기=안치 수용능력</span>'
    '<span>Arc 굵기=목적지 수용능력 · 높이=거리 · tilt=후보순위</span>'
    '</div>',
    unsafe_allow_html=True,
)

if view_mode == "경기도 전체":
    st.info(f"현재 31개 시군에서 시군별 상위 {top_n}개 시설을 연결해 총 {len(map_arcs):,}개의 Arc를 표시하고 있습니다.")

st.pydeck_chart(deck, height=650, use_container_width=True)
st.caption("같은 장례시설로 들어가는 Arc는 동일한 수용능력을 반영해 같은 굵기로 표시됩니다. Arc는 실제 이용 OD가 아니라 잠재적 서비스 연결입니다.")

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.subheader("잠재 공급압력 TOP 10")
    pressure_top = summary.replace([np.inf, -np.inf], np.nan).dropna(subset=["pressure"]).nlargest(10, "pressure")
    if pressure_top.empty:
        st.info("사망자수와 시설 수용력 데이터가 있어야 공급압력을 계산할 수 있습니다.")
    else:
        pressure_chart = pressure_top[["sigun", "pressure"]].sort_values("pressure").rename(
            columns={"sigun": "지역", "pressure": "공급압력"}
        )
        st.bar_chart(pressure_chart, x="지역", y="공급압력", horizontal=True, height=360)

with right:
    st.subheader("지역 상세")
    if view_mode == "선택 지역":
        detail_sigun = selected_sigun
    elif focus_sigun != "강조 없음":
        detail_sigun = focus_sigun
    elif not pressure_top.empty:
        detail_sigun = pressure_top.iloc[0]["sigun"]
    else:
        detail_sigun = regions.iloc[0]["sigun"]

    detail = summary[summary["sigun"] == detail_sigun].iloc[0]
    st.metric("선택 지역", detail_sigun)
    d1, d2 = st.columns(2)
    d1.metric("사망자", f"{detail['deaths']:,.0f}명" if pd.notna(detail["deaths"]) else "-")
    d2.metric("장례시설", f"{int(detail['facilities']):,}개")
    d3, d4 = st.columns(2)
    d3.metric("지역 내 총 안치 수용능력", f"{detail['capacity']:,.0f}")
    d4.metric("최근접 후보", f"{detail['best_access_km']:.1f} km" if pd.notna(detail["best_access_km"]) else "-")
    st.metric("잠재 공급압력", f"{detail['pressure']:,.1f}" if pd.notna(detail["pressure"]) else "-")

st.divider()
st.caption("데이터: 경기도 장례시설 현황 Open API + KOSIS 2024 시군구 사망자수. 분석상 Arc는 실제 이동량이 아닌 잠재 연결입니다.")