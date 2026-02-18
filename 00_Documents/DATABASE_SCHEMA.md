# Database Schema Documentation

## Overview

LocalDB 프로젝트는 태양/우주환경 데이터를 관리하기 위해 2개의 PostgreSQL 데이터베이스를 사용합니다.

| Database | Purpose | Tables |
|----------|---------|--------|
| `solar_images` | 태양 이미지 데이터 (FITS 파일 메타데이터) | `sdo`, `lasco`, `secchi` |
| `space_weather` | 우주환경 시계열 데이터 | `omni_low_resolution`, `omni_high_resolution`, `omni_high_resolution_5min` |

---

## 1. solar_images Database

태양 이미지 데이터 (SDO, LASCO, SECCHI)의 메타데이터를 저장합니다.

### 1.1 sdo Table

**SDO (Solar Dynamics Observatory)** - AIA/HMI 망원경 데이터

```sql
CREATE TABLE sdo (
    telescope     VARCHAR(10) NOT NULL,    -- 'aia' or 'hmi'
    channel       VARCHAR(20) NOT NULL,    -- '193', '211', 'm_45s', etc.
    datetime      TIMESTAMP NOT NULL,      -- Observation time (UTC)
    file_path     VARCHAR(512) NOT NULL,   -- Absolute path to FITS file
    quality       INTEGER,                 -- Data quality flag (0 = good)
    wavelength    INTEGER,                 -- Wavelength in Angstrom (AIA only)
    exposure_time REAL,                    -- Exposure time in seconds

    PRIMARY KEY (telescope, channel, datetime),
    UNIQUE (file_path)
);

-- Indexes
CREATE INDEX idx_sdo_datetime ON sdo(datetime);
CREATE INDEX idx_sdo_telescope ON sdo(telescope);
CREATE INDEX idx_sdo_quality ON sdo(quality);
```

#### Telescope & Channel Values

| Telescope | Channel | Wavelength (Å) | Description |
|-----------|---------|----------------|-------------|
| `aia` | `94` | 94 | Fe XVIII, flare plasma |
| `aia` | `131` | 131 | Fe VIII/XXI, flare plasma |
| `aia` | `171` | 171 | Fe IX, quiet corona |
| `aia` | `193` | 193 | Fe XII/XXIV, corona/flare |
| `aia` | `211` | 211 | Fe XIV, active region |
| `aia` | `304` | 304 | He II, chromosphere |
| `aia` | `335` | 335 | Fe XVI, active region |
| `hmi` | `m_45s` | NULL | Magnetogram 45s cadence |
| `hmi` | `m_720s` | NULL | Magnetogram 720s cadence |
| `hmi` | `ic_45s` | NULL | Continuum intensity 45s |

#### Quality Values

- `0`: Good quality data (정상)
- Non-zero: Various quality issues (문제 있음)

#### Example Queries

```sql
-- Get all AIA 193 images for a date
SELECT * FROM sdo
WHERE telescope = 'aia' AND channel = '193'
  AND datetime >= '2025-01-01' AND datetime < '2025-01-02'
ORDER BY datetime;

-- Get only good quality data
SELECT * FROM sdo
WHERE telescope = 'aia' AND channel = '193' AND quality = 0
ORDER BY datetime;

-- Find closest image to a target time
SELECT * FROM sdo
WHERE telescope = 'aia' AND channel = '193' AND quality = 0
ORDER BY ABS(EXTRACT(EPOCH FROM (datetime - '2025-01-01 12:00:00'::timestamp)))
LIMIT 1;
```

---

### 1.2 lasco Table

**LASCO (Large Angle and Spectrometric Coronagraph)** - SOHO 코로나그래프 데이터

```sql
CREATE TABLE lasco (
    camera        VARCHAR(4) NOT NULL,     -- 'c1', 'c2', 'c3', 'c4'
    datetime      TIMESTAMP NOT NULL,      -- Observation time (UTC)
    file_path     VARCHAR(512) NOT NULL,   -- Absolute path to FITS file
    exposure_time REAL,                    -- Exposure time in seconds
    filter        VARCHAR(20),             -- Filter name

    PRIMARY KEY (camera, datetime),
    UNIQUE (file_path)
);

CREATE INDEX idx_lasco_datetime ON lasco(datetime);
```

#### Camera Values

| Camera | Field of View | Description |
|--------|---------------|-------------|
| `c1` | 1.1-3 R☉ | Inner corona (discontinued) |
| `c2` | 2-6 R☉ | Outer corona (orange) |
| `c3` | 3.7-30 R☉ | Extended corona (blue) |
| `c4` | | Spectrometer |

---

### 1.3 secchi Table

**SECCHI (Sun Earth Connection Coronal and Heliospheric Investigation)** - STEREO 데이터

```sql
CREATE TABLE secchi (
    datatype      VARCHAR(10) NOT NULL,    -- 'science' or 'beacon'
    spacecraft    VARCHAR(10) NOT NULL,    -- 'ahead' or 'behind'
    instrument    VARCHAR(10) NOT NULL,    -- 'cor1', 'cor2', 'euvi', 'hi_1', 'hi_2'
    channel       VARCHAR(20),             -- EUVI wavelength channel
    datetime      TIMESTAMP NOT NULL,      -- Observation time (UTC)
    file_path     VARCHAR(512) NOT NULL,   -- Absolute path to FITS file
    exposure_time REAL,                    -- Exposure time in seconds
    filter        VARCHAR(20),             -- Filter name
    wavelength    INTEGER,                 -- Wavelength in Angstrom (EUVI only)

    PRIMARY KEY (datatype, spacecraft, instrument, channel, datetime),
    UNIQUE (file_path)
);

CREATE INDEX idx_secchi_datetime ON secchi(datetime);
CREATE INDEX idx_secchi_spacecraft ON secchi(spacecraft);
CREATE INDEX idx_secchi_instrument ON secchi(instrument);
```

#### Field Values

| Field | Values | Description |
|-------|--------|-------------|
| `datatype` | `science`, `beacon` | 과학 데이터 vs 실시간 비콘 |
| `spacecraft` | `ahead`, `behind` | STEREO-A vs STEREO-B |
| `instrument` | `cor1`, `cor2`, `euvi`, `hi_1`, `hi_2` | 관측 장비 |
| `channel` | `171`, `195`, `284`, `304` | EUVI 파장 (Å) |

---

## 2. space_weather Database

우주환경 시계열 데이터를 저장합니다.

### 2.1 omni_low_resolution Table

**OMNI Hourly Data** - 1시간 해상도 태양풍/자기장 데이터

```sql
CREATE TABLE omni_low_resolution (
    datetime TIMESTAMP PRIMARY KEY,

    -- Time info
    year INTEGER NOT NULL,
    decimal_day INTEGER NOT NULL,
    hour INTEGER NOT NULL,

    -- Magnetic field (IMF)
    b_field_magnitude_avg_nt REAL,
    bx_gse_gsm_nt REAL,
    by_gse_nt REAL,
    bz_gse_nt REAL,
    by_gsm_nt REAL,
    bz_gsm_nt REAL,

    -- Plasma parameters
    proton_temperature_k REAL,
    proton_density_n_cm3 REAL,
    plasma_flow_speed_km_s REAL,
    flow_pressure_npa REAL,
    electric_field_mv_m REAL,
    plasma_beta REAL,
    alfven_mach_number REAL,

    -- Geomagnetic indices
    kp_index SMALLINT,
    dst_index_nt INTEGER,
    ae_index_nt INTEGER,
    ap_index_nt INTEGER,

    -- Solar indices
    sunspot_number_r INTEGER,
    f10_7_index_sfu REAL,

    -- Proton flux
    proton_flux_gt10mev REAL,
    proton_flux_gt30mev REAL,
    proton_flux_gt60mev REAL,

    -- ... (55 columns total)
);
```

### 2.2 omni_high_resolution Table

**OMNI 1-minute Data** - 1분 해상도 데이터

```sql
CREATE TABLE omni_high_resolution (
    datetime TIMESTAMP PRIMARY KEY,

    -- Time info
    year INTEGER NOT NULL,
    day INTEGER NOT NULL,
    hour INTEGER NOT NULL,
    minute INTEGER NOT NULL,

    -- Magnetic field
    b_magnitude_nt REAL,
    bx_gse_nt REAL,
    by_gse_nt REAL,
    bz_gse_nt REAL,

    -- Plasma
    flow_speed_km_s REAL,
    proton_density_n_cc REAL,
    temperature_k REAL,

    -- Geomagnetic indices
    ae_index_nt INTEGER,
    sym_h_nt INTEGER,

    -- ... (46 columns total)
);
```

### 2.3 omni_high_resolution_5min Table

**OMNI 5-minute Data** - 5분 해상도 데이터 (proton flux 포함)

1분 데이터와 동일한 구조 + proton flux 3개 컬럼 (49 columns total)

---

## 3. Data Storage Policy

| Data Source | Policy |
|-------------|--------|
| SDO | 모든 유효한 데이터 저장 (quality 무관, best match는 조회 시 선택) |
| LASCO | 모든 데이터 저장 |
| SECCHI | 모든 데이터 저장 |
| OMNI | 모든 데이터 저장 |

---

## 4. Configuration Files

### 4.1 solar_images_config.yaml

```yaml
db_config:
  host: localhost
  port: 5432
  database: solar_images
  user: ${DB_USER}
  password: ${DB_PASSWORD}

download_config:
  sdo:
    download_root: "/opt/archive/solar_images/sdo"
    dirs:
      aia: "aia"
      hmi: "hmi"
      downloaded: "downloaded"
      invalid_file: "invalid_file"
      invalid_header: "invalid_header"
      invalid_data: "invalid_data"
```

---

## 5. File Storage Structure

```
/opt/archive/
├── solar_images/
│   ├── sdo/
│   │   ├── aia/
│   │   │   └── YYYY/YYYYMMDD/
│   │   │       └── aia.lev1_euv_12s.YYYY-MM-DDTHH_MM_SS.*.fits
│   │   ├── hmi/
│   │   │   └── YYYY/YYYYMMDD/
│   │   │       └── hmi.m_45s.YYYY.MM.DD_HH_MM_SS.*.fits
│   │   ├── downloaded/      # Temporary download folder
│   │   ├── invalid_file/    # Corrupted files
│   │   ├── invalid_header/  # Missing headers
│   │   └── invalid_data/    # Data errors
│   ├── lasco/
│   │   └── cX/YYYY/YYYYMM/
│   │       └── YYYYMMDD_HHMMSS_cX.fts
│   └── secchi/
│       └── datatype/spacecraft/instrument/YYYY/YYYYMMDD/
└── space_weather/
    └── omni/
        ├── low_resolution/
        └── high_resolution/
```

---

## 6. Common Queries

### SDO: Get best quality data for a time range

```sql
SELECT DISTINCT ON (DATE_TRUNC('hour', datetime))
    telescope, channel, datetime, file_path, quality
FROM sdo
WHERE telescope = 'aia'
  AND channel = '193'
  AND quality = 0
  AND datetime BETWEEN '2025-01-01' AND '2025-01-02'
ORDER BY DATE_TRUNC('hour', datetime),
         ABS(EXTRACT(EPOCH FROM (datetime - DATE_TRUNC('hour', datetime))));
```

### SDO: Count records by telescope/channel

```sql
SELECT telescope, channel, COUNT(*) as count
FROM sdo
GROUP BY telescope, channel
ORDER BY telescope, channel;
```

### OMNI: Get solar wind speed for a period

```sql
SELECT datetime, plasma_flow_speed_km_s, bz_gsm_nt, dst_index_nt
FROM omni_low_resolution
WHERE datetime BETWEEN '2025-01-01' AND '2025-01-07'
ORDER BY datetime;
```
