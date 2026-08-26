# 경기도 장례시설 접근성 대시보드

경기도 1 km 격자별 80세 이상 인구를 잠재수요로 정의하고, 장례시설의 공급량과 거리감쇠를 결합한 중력모델 결과를 탐색하는 Streamlit 대시보드입니다.

## 주요 기능

- 80세 이상 인구 분포, 수요 대비 장례시설 접근성, 장례서비스 개선 시급성 지도
- 모든 지표 지도에 장례시설 빨간 점을 항상 최상단에 표시
- 격자 툴팁과 시급지역 상세표에 행정동 정보 표시
- 최소 고령인구 및 개선 시급지역 개수 필터
- 장례시설 위치 표시
- 잠재수요와 접근성 관계 산점도
- 장례서비스 개선 시급지역 순위와 CSV 다운로드

## 데이터 준비

`data/` 폴더에 아래 파일을 넣습니다.

```text
data/
├── gyeonggi_funeral_access_grid.geojson
└── facilities.csv  # 또는 장례시설현황.csv
```

## 로컬 실행

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Streamlit Community Cloud 배포

1. 이 폴더를 GitHub 공개 저장소에 올립니다.
2. `https://share.streamlit.io`에서 GitHub 계정으로 로그인합니다.
3. 저장소와 `main` 브랜치를 선택합니다.
4. Main file path를 `streamlit_app.py`로 지정합니다.
5. Deploy를 누릅니다.

## 자료

- SGIS 2024년 1 km 격자 인구통계
- SGIS 2025년 경기도 시도경계
- 경기도 장례시설현황 2024년

## 해석상 주의

직선거리는 실제 도로 이동시간과 다르며, 80세 이상 인구는 실제 사망자 수가 아닌 잠재수요 대리변수입니다. SGIS 통계비밀보호 및 경기도 외부 인접시설 제외의 영향을 받을 수 있습니다.
