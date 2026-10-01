# 데이터 확보·변환 절차와 데이터 사전

담당: 데이터·도메인 ([@yunanana](https://github.com/yunanana))

`data/sample_gasoline_prices.csv`는 **120일 합성 예제**입니다. 실측 데이터나 성능 증빙으로 쓰지 않습니다.
실측 원본은 아래 절차로 받아 `data/raw/`에 두며, 원본 파일은 커밋하지 않습니다(`data/raw/.gitignore`).

## 1. 휘발유 가격 (`gasoline_price`)

| 항목 | 내용 |
|---|---|
| 출처 | 한국석유공사 오피넷 <https://www.opinet.co.kr> |
| 메뉴 | 국내유가통계 → 주유소 → 평균판매가격 → **제품별** 탭 |
| 조회 조건 | 구분 **일간**, 기간 지정, 제품 **보통휘발유** |
| 저장 | **CSV저장** 버튼 → `data/raw/opinet_gasoline.csv`로 저장 |
| 의미 | 전국 주유소 보통휘발유 평균 판매가격 (부가세 포함) |
| 단위 / 주기 | 원/L, 일간 (주말·공휴일 포함 매일) |
| 원본 형식 | CP949, 헤더 `구분,고급휘발유,보통휘발유,자동차용경유,실내등유`, 날짜 `2026년09월26일` |
| 이용 조건 | 공개 조회 데이터. 재배포 조건은 오피넷 이용약관 확인 전까지 원본을 커밋하지 않음 |

변환 확인:

```bash
.venv/bin/python -m data.opinet data/raw/opinet_gasoline.csv
# 원본 N행 -> 변환 N행 / 기간 ... / 가격 범위 ...
```

### 변환 규칙 (`data/opinet.py`)

- 인코딩: CP949 원본과 UTF-8로 다시 저장한 파일을 모두 읽습니다.
- 날짜: `YYYY년MM월DD일`, `YY년MM월DD일`, ISO 등 → `YYYY-MM-DD`. 날짜순으로 정렬합니다.
- 중복: 같은 날짜·같은 값이면 하나만 남기고, 값이 다르면 오류입니다(어느 값이 맞는지 판단 불가).
- 누락: 평균판매가격은 매일 집계되므로 날짜가 빠지면 **오류**입니다. 앞뒤 값으로 보간하지 않습니다.
  (미래 값을 끌어오는 보간은 예측 시점에 알 수 없는 정보이므로 금지합니다.)
- 비정상 값: 0 이하, 빈 값, NaN/Infinity는 오류입니다.
- 원본 보존: 원본은 `data/raw/`에 그대로 두고, 변환은 항상 원본에서 다시 수행합니다(같은 원본 → 같은 결과).

## 2. 국제유가 (`crude_oil_price`)

| 항목 | 내용 |
|---|---|
| 출처 / 메뉴 | 오피넷 → 유가관련정보 → 국제유가 → 원유 |
| 조회 조건 | 구분 **일간**, 제품 **Dubai**, 단위 버튼 **`$`** (기본값) |
| 저장 | **CSV저장** → `data/raw/opinet_crude.csv` |
| 유종 선택 이유 | 국내 원유 도입의 대부분이 중동산이라 국내 가격 기준 유종으로 두바이유를 사용 |
| 단위 / 주기 | USD/bbl, 거래일만 (주말·휴일 없음) |
| 원본 형식 | CP949, 헤더 `기간,Dubai,Brent,WTI`, 날짜 `26년09월24일` (연도 2자리) |

## 3. 환율 (`usd_krw`)

| 항목 | 내용 |
|---|---|
| 출처 | 한국은행 ECOS Open API, 통계표 `731Y001` 항목 `0000001` (원/미국달러 매매기준율) |
| 받기 | 인증키 무료 발급(<https://ecos.bok.or.kr/api/>) 후 아래 명령 → `data/raw/ecos_usdkrw.csv` |
| 단위 / 주기 | KRW/USD, 영업일만 |

```bash
ECOS_API_KEY=발급받은키 .venv/bin/python -m data.ecos --start 20201201 --end 20231231
```

## 4. 유류세 인하율 (`tax_or_supply_feature`)

- 의미: 해당일에 시행 중인 **휘발유 유류세 인하율(%)**. 0이면 인하 없음.
- 원본: `data/reference/gasoline_fuel_tax_cut.csv` (시행 구간표, 출처 기록). 구간은 빈틈·겹침이 없어야 합니다.
- 구간표 밖의 날짜는 추정하지 않고 오류 처리합니다. 2024년 이후 데이터를 쓰려면 기획재정부 고시를 확인해 구간을 추가하세요.
- 2023년 행(25%)은 연중 연장 고시를 추가로 확인해야 합니다.
- **공급 차질**은 일별 관측 지표가 없어서 피처로 넣지 않고, AIOps 드리프트 시나리오로만 다룹니다.

## 5. 시점 정합 (데이터 누수 방지)

D일 행에는 **D일 예측 시점에 이미 공개된 값**만 들어갑니다 (`data/external.py`).

| 피처 | D일 행에 들어가는 값 | 이유 |
|---|---|---|
| `crude_oil_price` | D-1일 이전 가장 최근 거래일 값 | 두바이 현물은 싱가포르 장 마감 기준이라 D일 값은 늦게 공개될 수 있음 |
| `usd_krw` | D일 이전(당일 포함) 가장 최근 영업일 값 | 매매기준율은 D일 오전에 고시됨 |
| `tax_or_supply_feature` | D일 시행 중인 인하율 | 시행일이 사전 고시되는 정책값 |

- 주말·휴일은 **이전** 관측값을 그대로 씁니다. 뒤 날짜 값으로 채우는 보간은 하지 않습니다.
- 마지막 관측값이 7일보다 오래됐으면 원본 누락으로 보고 오류 처리합니다.
- 맨 앞 날짜 중 공개된 외부 값이 없는 행은 추정하지 않고 제외합니다. 그래서 국제유가·환율은 휘발유 시작일보다 2주 정도 앞에서부터 받아 두세요.

## 6. 정규화 CSV 만들기

```bash
.venv/bin/python -m data.build_dataset
# 기본 입력: data/raw/opinet_gasoline.csv, opinet_crude.csv, ecos_usdkrw.csv
# 출력: data/processed/gasoline_features.csv (커밋하지 않음)
```

출력은 업로드 라우터와 같은 `validate_rows()`로 한 번 더 검증합니다. 이 파일을 대시보드에서 업로드하면 됩니다.

## 데이터 사전 (정규화 CSV)

컬럼 순서는 `data/features.py`의 `FEATURE_COLUMNS`와 같습니다.

| 컬럼 | 의미 | 단위 | 출처 |
|---|---|---|---|
| `date` | 기준일 | ISO 날짜 | 오피넷 |
| `gasoline_price` | 전국 평균 보통휘발유 판매가격 (예측 대상) | 원/L | 오피넷 |
| `crude_oil_price` | 두바이유 현물 가격, D-1일까지 공개된 값 | USD/bbl | 오피넷 |
| `usd_krw` | 원/미국달러 매매기준율, D일까지 고시된 값 | KRW/USD | 한국은행 ECOS |
| `tax_or_supply_feature` | 휘발유 유류세 인하율 | % (0~37) | 기획재정부 고시 기반 구간표 |

## 경유 전환 제안 (#24~#27)

팀의 기존 `main` 계약과 API는 아직 휘발유입니다. 다음 경유 경로는 데이터 담당 검토용이며,
`diesel_features.csv`를 기존 `/data/upload`에 올리면 `gasoline_price` 컬럼 검사에 실패합니다.
PM과 모델·서빙·AIOps 담당이 공통 계약과 품질 게이트를 확정한 뒤 서비스 전환을 진행합니다.

| 항목 | 경유 데이터 계약 |
|---|---|
| 오피넷 원본 | 주유소 → 평균판매가격 → 제품별 → 일간 → **자동차용경유** 선택, CSV저장 |
| 원본 형식 | CP949 또는 UTF-8, `구분,자동차용경유` (선택한 제품만 내려받으면 2열) |
| 목표/단위 | `diesel_price`, 전국 평균 자동차용경유 가격, 원/L |
| 최종 컬럼 순서 | `date,diesel_price,crude_oil_price,usd_krw,tax_or_supply_feature` |
| 경유 유류세 피처 | 경유 기본 탄력세율 375원/L 대비 인하율(%), 소수 첫째 자리 |
| 원본 보존 | `data/raw/`에 보관, 이용 조건 확인 전 Git 커밋 금지 |

경유 원본만 확인하려면:

```bash
.venv/bin/python -c 'from data.opinet import load_opinet_diesel; import sys; rows=load_opinet_diesel(sys.argv[1]); print(len(rows), rows[0], rows[-1])' data/raw/opinet_diesel.csv
```

기존 두바이유(`$` 단위)와 ECOS 환율 원본이 준비되면 아래처럼 합칩니다. 두바이유는 D-1일까지,
환율은 D일까지 공개된 최근 거래일 값을 사용하며, 경유 정책값은 D일 시행값을 사용합니다.

```bash
.venv/bin/python -m data.build_diesel_dataset
# 입력: data/raw/opinet_diesel.csv, opinet_crude.csv, ecos_usdkrw.csv
# 출력: data/processed/diesel_features.csv (실측 원본/가공본은 커밋하지 않음)
```

경유 유류세 원본은 [교통·에너지·환경세법 시행령 제정·개정이유](https://law.go.kr/LSW/lsRvsRsnListP.do?chrClsCd=010202&lsId=002619&lsRvsGubun=all)입니다.
`data/reference/diesel_fuel_tax_cut.csv`에 시행 구간별 **경유** 탄력세율(원/L)과 법령 번호를 기록하고
`(1 - 탄력세율 / 375) × 100`을 소수 첫째 자리로 반올림한 값을 피처로 사용합니다. 휘발유 표와 혼용하지 않습니다.
구간표 밖의 날짜는 추정하지 않고 오류로 처리합니다.
