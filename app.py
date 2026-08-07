import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

st.set_page_config(page_title="경기도 장례서비스 공간 접근성", page_icon="🕯️", layout="wide")

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
    for c in candidates:
        if c.upper() in lookup:
            return lookup[c.upper()]
    return None


def flatten_gg_rows(payload):
    if isinstance(payload, list):
        for block in payload:
            if isinstance(block, dict) and "row" in block and isinstance(block["row"], list):
                return block["row"]
            if isinstance(block, dict):
                for value in block.values():
                    rows = flatten_gg_rows(value)
                    if rows:
                        return rows
    elif isinstance(payload, dict):
        if "row" in payload and isinstance(payload["row"], list):
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
        r = requests.get(API_URL, params=params, timeout=25)
        r.raise_for_status()
        payload = r.json()
        rows = flatten_gg_rows(payload)
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        page += 1
        if page > 50:
            break

    if not all_rows:
        raise ValueError("API에서 row 데이터를 찾지 못했습니다. 인증키/서비스 응답을 확인하세요.")
    return pd.DataFrame(all_rows)


def normalize_facilities(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    cols = df.columns

    name_col = first_existing(cols, ["BIZPLC_NM", "FACLT_NM", "FCLTY_NM", "NAME", "CMPNM_NM", "INST_NM"])
    sigun_col = first_existing(cols, ["SIGUN_NM", "SIGUN", "SIGUNGU_NM", "SIGNGU_NM"])
    addr_col = first_existing(cols, ["REFINE_ROADNM_ADDR", "ROADNM_ADDR", "REFINE_LOTNO_ADDR", "LOTNO_ADDR", "ADDR"])
    lat_col = first_existing(cols, ["REFINE_WGS84_LAT", "WGS84_LAT", "LAT", "LATITUDE"])
    lon_col = first_existing(cols, ["REFINE_WGS84_LOGT", "REFINE_WGS84_LON", "WGS84_LOGT", "WGS84_LON", "LON", "LONGITUDE"])
    room_col = first_existing(cols, ["FUNERAL_PARLOR_CNT", "MORTUARY_CNT", "ROOM_CNT", "BINso_CNT", "BINSO_CNT"])
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
    out["capacity"] = out["capacity"].fillna(out["rooms"])
    out["capacity"] = out["capacity"].clip(lower=1)
    return out.drop_duplicates(subset=["facility", "lat", "lon"]).reset_index(drop=True)


def haversine_km(lon1, lat1, lon2, lat2):
    r = 6371.0088
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def load_deaths() -> pd.DataFrame:
    if not DEATHS_PATH.exists():
        return pd.DataFrame(columns=["sigun", "deaths"])
    d = pd.read_csv(DEATHS_PATH)
    if not {"sigun", "deaths"}.issubset(d.columns):
        raise ValueError("data/deaths_by_sigun.csv에는 sigun, deaths 컬럼이 필요합니다.")
    d = d[["sigun", "deaths"]].copy()
    d["sigun"] = d["sigun"].astype(str).str.strip()
    d["deaths"] = pd.to_numeric(d["deaths"], errors="coerce")
    return d.dropna(subset=["deaths"])


def build_region_frame(deaths: pd.DataFrame) -> pd.DataFrame:
    rows = [{"sigun": k, "source_lon": v[0], "source_lat": v[1]} for k, v in SIGUN_CENTERS.items()]
    reg = pd.DataFrame(rows)
    if not deaths.empty:
        reg = reg.merge(deaths, on="sigun", how="left")
    else:
        reg["deaths"] = np.nan
    return reg


def build_arcs(regions: pd.DataFrame, facilities: pd.DataFrame, top_n=3, alpha=1.5) -> pd.DataFrame:
    arcs = []
    for _, region in regions.iterrows():
        f = facilities.copy()
        f["distance_km"] = haversine_km(region.source_lon, region.source_lat, f["lon"].values, f["lat"].values)
        f["access_score"] = f["capacity"] / np.power(np.maximum(f["distance_km"], 1.0), alpha)
        f = f.sort_values("access_score", ascending=False).head(top_n).copy()
        total = f["access_score"].sum()
        f["share"] = f["access_score"] / total if total > 0 else 1 / len(f)

        for rank, (_, x) in enumerate(f.iterrows(), start=1):
            deaths = region.get("deaths", np.nan)
            demand_flow = float(deaths * x["share"]) if pd.notna(deaths) else float(x["share"])
            arcs.append({
                "sigun": region.sigun,
                "facility": x.facility,
                "source_lon": region.source_lon,
                "source_lat": region.source_lat,
                "target_lon": x.lon,
                "target_lat": x.lat,
                "distance_km": float(x.distance_km),
                "capacity": float(x.capacity),
                "rooms": float(x.rooms),
                "rank": rank,
                "access_score": float(x.access_score),
                "share": float(x["share"]),
                "deaths": None if pd.isna(deaths) else float(deaths),
                "demand_flow": demand_flow,
            })
    arc_df = pd.DataFrame(arcs)
    if arc_df.empty:
        return arc_df
    if arc_df["deaths"].notna().any():
        mx = arc_df["demand_flow"].max() or 1
        arc_df["arc_width"] = 0.25 + 0.75 * np.sqrt(arc_df["demand_flow"] / mx)
    else:
        arc_df["arc_width"] = 0.25 + 0.75 * arc_df["share"]
    return arc_df


def load_colored_geojson(summary: pd.DataFrame):
    if not GEOJSON_PATH.exists():
        return None
    import json
    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        gj = json.load(f)

    value_map = summary.set_index("sigun").to_dict("index")
    vals = summary["pressure"].dropna() if summary["pressure"].notna().any() else summary["deaths"].dropna()
    if vals.empty:
        vals = summary["facilities"].astype(float)
    vmin, vmax = float(vals.min()), float(vals.max())

    name_keys = ["SIG_KOR_NM", "SIGUN_NM", "sggnm", "name", "NAME", "SIGUN"]
    for feature in gj.get("features", []):
        props = feature.setdefault("properties", {})
        sigun = next((str(props[k]).strip() for k in name_keys if k in props and props[k]), None)
        row = value_map.get(sigun, {}) if sigun else {}
        metric = row.get("pressure")
        if metric is None or pd.isna(metric):
            metric = row.get("deaths")
        if metric is None or pd.isna(metric):
            metric = row.get("facilities", 0)
        metric = float(metric or 0)
        t = 0 if vmax <= vmin else max(0, min(1, (metric - vmin) / (vmax - vmin)))
        props["sigun_dashboard"] = sigun or "미상"
        props["metric_dashboard"] = round(metric, 2)
        props["fill_color"] = [int(45 + 210*t), int(120 - 60*t), int(210 - 150*t), 135]
    return gj


def aggregate_sigun(regions, facilities, arcs):
    fagg = facilities.groupby("sigun", as_index=False).agg(
        facilities=("facility", "count"), rooms=("rooms", "sum"), capacity=("capacity", "sum")
    )
    nearest = arcs.groupby("sigun", as_index=False)["distance_km"].min().rename(columns={"distance_km": "best_access_km"})
    out = regions.merge(fagg, on="sigun", how="left").merge(nearest, on="sigun", how="left")
    out[["facilities", "rooms", "capacity"]] = out[["facilities", "rooms", "capacity"]].fillna(0)
    out["pressure"] = np.where(out["deaths"].notna(), out["deaths"] / out["capacity"].replace(0, np.nan), np.nan)
    return out


st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1420px;}
    .hero-kicker {color:#2563eb; font-size:.82rem; font-weight:800; letter-spacing:.16em; margin-bottom:.25rem;}
    .hero-year {display:inline-flex; align-items:center; gap:.45rem; padding:.28rem .7rem;
        border-radius:999px; background:#eff6ff; color:#1d4ed8; font-weight:700; font-size:.82rem;}
    [data-testid="stMetric"] {background:linear-gradient(145deg,#ffffff,#f8fafc); border:1px solid #e2e8f0;
        border-radius:16px; padding:1rem 1.1rem; box-shadow:0 8px 24px rgba(15,23,42,.05);}
    [data-testid="stMetricLabel"] {color:#64748b;}
    [data-testid="stMetricValue"] {color:#0f172a;}
    .map-legend {display:flex; gap:1.2rem; align-items:center; color:#475569; font-size:.88rem; margin:.25rem 0 .8rem;}
    .legend-dot {display:inline-block; width:.7rem; height:.7rem; border-radius:50%; margin-right:.35rem;}
    .detail-card {border:1px solid #e2e8f0; border-radius:18px; padding:1.15rem 1.25rem;
        background:linear-gradient(145deg,#ffffff,#f8fafc); min-height:300px;}
    .detail-name {font-size:1.65rem; font-weight:800; color:#0f172a; margin-bottom:.8rem;}
    .detail-row {display:flex; justify-content:space-between; padding:.72rem 0; border-bottom:1px solid #e2e8f0;}
    .detail-label {color:#64748b;} .detail-value {font-weight:750; color:#0f172a;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="hero-kicker">GYEONGGI FUNERAL INFRASTRUCTURE</div>', unsafe_allow_html=True)
st.title("경기도 장례 인프라 공간 접근성 대시보드")
st.markdown('<span class="hero-year">● 2024</span>', unsafe_allow_html=True)
st.caption("지역별 사망 수요와 장례시설 공급을 연결해 공간 접근성과 공급압력을 살펴봅니다.")

api_key = get_secret("GG_API_KEY")

with st.sidebar:
    st.header("분석 설정")
    top_n = st.slider("지역별 연결 시설 수", 1, 5, 2)
    alpha = st.slider("거리 감쇠 α", 0.5, 3.0, 1.5, 0.1)
    sigun_options = ["전체"] + list(SIGUN_CENTERS.keys())
    default_sigun = sigun_options.index("양평군")
    selected_sigun = st.selectbox("지역 선택", sigun_options, index=default_sigun)
    arc_strength = st.slider("Arc 강조", 0.6, 1.8, 1.0, 0.1)
    show_columns = st.toggle("시설 수용력 기둥", value=True)
    st.divider()
    st.markdown("**지도 읽는 법**")
    st.caption("파랑은 지역 수요 지점, 주황은 장례시설입니다. Arc의 굵기는 해당 시설로 배분된 잠재 수요 비중을 나타냅니다.")
    st.latex(r"Score_{ij}=Capacity_j / Distance_{ij}^{\alpha}")

try:
    raw = fetch_facilities(api_key)
    facilities = normalize_facilities(raw)
except Exception as e:
    st.error(f"장례시설 API를 불러오지 못했습니다: {e}")
    st.info("Streamlit Community Cloud의 App settings → Secrets에 `GG_API_KEY = \"발급받은키\"`를 설정하세요.")
    st.stop()

if facilities.empty:
    st.warning("유효한 WGS84 좌표가 있는 장례시설을 찾지 못했습니다.")
    st.stop()

deaths = load_deaths()
regions = build_region_frame(deaths)
arcs = build_arcs(regions, facilities, top_n=top_n, alpha=alpha)
summary = aggregate_sigun(regions, facilities, arcs)
colored_geojson = load_colored_geojson(summary)

average_distance = arcs["distance_km"].mean() if not arcs.empty else np.nan
m1, m2, m3, m4 = st.columns(4)
m1.metric("사망자수", f"{deaths['deaths'].sum():,.0f}명" if not deaths.empty else "-")
m2.metric("시설 수", f"{len(facilities):,}개")
m3.metric("총 수용력", f"{facilities['capacity'].sum():,.0f}")
m4.metric("평균 연결거리", f"{average_distance:,.1f} km" if pd.notna(average_distance) else "-")

if selected_sigun == "전체":
    map_arcs = arcs.copy()
    map_regions = regions.copy()
    map_facilities = facilities.copy()
    map_arcs["display_width"] = (0.45 + 0.8 * map_arcs["share"]) * arc_strength
    source_color = [37, 99, 235, 45]
    target_color = [249, 115, 22, 65]
    arc_height = 0.18
    view = pdk.ViewState(latitude=37.45, longitude=127.05, zoom=8.15, pitch=48, bearing=-8)
else:
    map_arcs = arcs[arcs["sigun"] == selected_sigun].copy()
    map_regions = regions[regions["sigun"] == selected_sigun].copy()
    target_names = set(map_arcs["facility"])
    map_facilities = facilities[facilities["facility"].isin(target_names)].copy()
    map_arcs["display_width"] = (2.5 + 4.5 * map_arcs["share"]) * arc_strength
    source_color = [37, 99, 235, 210]
    target_color = [249, 115, 22, 235]
    arc_height = 0.58
    center_lon = pd.concat([map_regions["source_lon"], map_arcs["target_lon"]]).mean()
    center_lat = pd.concat([map_regions["source_lat"], map_arcs["target_lat"]]).mean()
    view = pdk.ViewState(latitude=float(center_lat), longitude=float(center_lon), zoom=9.35, pitch=55, bearing=-12)

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
            line_width_min_pixels=0.7,
            pickable=True,
            auto_highlight=True,
        )
    )

if show_columns:
    layers.append(
        pdk.Layer(
            "ColumnLayer",
            id="facility-columns",
            data=map_facilities,
            get_position="[lon, lat]",
            get_elevation="capacity",
            elevation_scale=2500,
            radius=360 if selected_sigun != "전체" else 220,
            get_fill_color=[249, 115, 22, 205],
            get_line_color=[154, 52, 18, 230],
            stroked=True,
            pickable=True,
            auto_highlight=True,
        )
    )

layers.append(
    pdk.Layer(
        "ScatterplotLayer",
        id="facility-points",
        data=map_facilities,
        get_position="[lon, lat]",
        get_radius=520 if selected_sigun != "전체" else 240,
        get_fill_color=[255, 237, 213, 245],
        get_line_color=[234, 88, 12, 245],
        line_width_min_pixels=2,
        stroked=True,
        pickable=True,
    )
)

if not map_arcs.empty:
    layers.append(
        pdk.Layer(
            "ArcLayer",
            id="access-arcs",
            data=map_arcs,
            get_source_position="[source_lon, source_lat]",
            get_target_position="[target_lon, target_lat]",
            get_source_color=source_color,
            get_target_color=target_color,
            get_width="display_width",
            width_units="pixels",
            get_height=arc_height,
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
        get_radius=900 if selected_sigun != "전체" else 430,
        get_fill_color=[37, 99, 235, 235],
        get_line_color=[239, 246, 255, 255],
        line_width_min_pixels=2,
        stroked=True,
        pickable=True,
    )
)

if selected_sigun != "전체":
    layers.append(
        pdk.Layer(
            "TextLayer",
            id="region-label",
            data=map_regions,
            get_position="[source_lon, source_lat]",
            get_text="sigun",
            get_size=16,
            size_units="pixels",
            get_color=[15, 23, 42, 255],
            get_pixel_offset=[0, -24],
            get_text_anchor="'middle'",
            get_alignment_baseline="'bottom'",
        )
    )
    layers.append(
        pdk.Layer(
            "TextLayer",
            id="facility-labels",
            data=map_facilities,
            get_position="[lon, lat]",
            get_text="facility",
            get_size=12,
            size_units="pixels",
            get_color=[124, 45, 18, 255],
            get_pixel_offset=[0, -20],
            get_text_anchor="'middle'",
            get_alignment_baseline="'bottom'",
        )
    )

tooltip = {
    "html": "<b>{facility}{sigun}</b><br/>거리 {distance_km} km<br/>수용력 {capacity}<br/>잠재 배분 {share}",
    "style": {"backgroundColor": "#0f172a", "color": "white", "borderRadius": "10px"},
}
deck = pdk.Deck(layers=layers, initial_view_state=view, tooltip=tooltip, map_style=None)

st.subheader("지역 → 장례시설 잠재 연결망")
st.markdown(
    '<div class="map-legend">'
    '<span><i class="legend-dot" style="background:#2563eb"></i>지역 수요</span>'
    '<span>Arc 흐름 →</span>'
    '<span><i class="legend-dot" style="background:#f97316"></i>장례시설 공급</span>'
    '</div>',
    unsafe_allow_html=True,
)
st.pydeck_chart(deck, height=585, use_container_width=True)
st.caption("Arc 색상은 파랑(지역)에서 주황(장례시설)으로 이어지며, 굵기는 잠재 배분 비중을 나타냅니다.")

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.subheader("공급압력 TOP 10")
    pressure_top = summary.replace([np.inf, -np.inf], np.nan).dropna(subset=["pressure"]).nlargest(10, "pressure")
    if pressure_top.empty:
        st.info("사망자수와 시설 수용력 데이터가 있어야 공급압력을 계산할 수 있습니다.")
    else:
        pressure_chart = pressure_top[["sigun", "pressure"]].sort_values("pressure")
        st.bar_chart(
            pressure_chart,
            x="sigun",
            y="pressure",
            horizontal=True,
            color="#2563eb",
            height=360,
        )

with right:
    st.subheader("지역 상세정보")
    if selected_sigun == "전체":
        detail_sigun = pressure_top.iloc[0]["sigun"] if not pressure_top.empty else regions.iloc[0]["sigun"]
        st.caption(f"전체 보기에서는 공급압력이 가장 높은 {detail_sigun}을 표시합니다.")
    else:
        detail_sigun = selected_sigun

    detail = summary[summary["sigun"] == detail_sigun].iloc[0]
    detail_deaths = f"{detail['deaths']:,.0f}명" if pd.notna(detail["deaths"]) else "-"
    detail_distance = f"{detail['best_access_km']:.1f} km" if pd.notna(detail["best_access_km"]) else "-"
    st.markdown(
        f"""
        <div class="detail-card">
          <div class="detail-name">{detail_sigun}</div>
          <div class="detail-row"><span class="detail-label">사망자</span><span class="detail-value">{detail_deaths}</span></div>
          <div class="detail-row"><span class="detail-label">장례시설</span><span class="detail-value">{int(detail['facilities']):,}개</span></div>
          <div class="detail-row"><span class="detail-label">총 수용력</span><span class="detail-value">{detail['capacity']:,.0f}</span></div>
          <div class="detail-row"><span class="detail-label">최근접 연결</span><span class="detail-value">{detail_distance}</span></div>
          <div class="detail-row"><span class="detail-label">공급압력</span><span class="detail-value">{detail['pressure']:,.1f}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()
st.caption("주의: Arc는 실제 장례식장 이용 OD가 아니라 시설규모와 거리 감쇠로 산출한 잠재 연결입니다.")
