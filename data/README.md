# 대시보드 데이터

이 폴더에 다음 파일을 넣습니다.

1. `gyeonggi_funeral_access_grid.geojson`
   - Colab 노트북이 생성한 경기도 1 km 격자별 접근성 결과
2. `facilities.csv` 또는 `장례시설현황.csv`
   - 공개용 정리 파일 또는 원본 경기도 장례시설현황 CSV

격자 결과에는 다음 컬럼이 필요합니다.

- `grid_id`
- `admin_dong` (없으면 대시보드에서 `행정동 경계 미연결`로 표시)
- `pop80`
- `nearest_km`
- `access_per_1000`
- `priority_score`
- `geometry`

원본 SGIS 전국 인구 CSV는 대시보드 저장소에 포함할 필요가 없습니다.
