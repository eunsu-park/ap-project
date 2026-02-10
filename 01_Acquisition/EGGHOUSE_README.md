# PostgresManager - Simple PostgreSQL Utilities

간단하고 실용적인 PostgreSQL 데이터베이스 관리 유틸리티입니다.

## 설치

### 필수 의존성

```bash
pip install psycopg2-binary
```

### 선택적 의존성

pandas DataFrame 변환 기능을 사용하려면:

```bash
pip install pandas
```

또는 egghouse 패키지 설치 시:

```bash
pip install git+https://github.com/eunsu-park/egghouse.git
```

## 빠른 시작

### 기본 사용법

```python
from egghouse.database import PostgresManager

# 데이터베이스 연결
db = PostgresManager(
    host='localhost',
    database='mydb',
    user='user',
    password='password',
    log_queries=True  # 쿼리 로깅 활성화
)

# 테이블 생성
db.create_table('users', {
    'id': 'SERIAL PRIMARY KEY',
    'name': 'VARCHAR(100)',
    'email': 'VARCHAR(255)',
    'created_at': 'TIMESTAMP DEFAULT NOW()'
})

# 데이터 삽입
db.insert('users', {'name': 'Eunsu', 'email': 'eunsu@example.com'})

# 데이터 조회
users = db.select('users', where={'name': 'Eunsu'})
print(users)

# 연결 종료
db.close()
```

### 날짜 범위 조회 & DataFrame 변환

```python
from egghouse.database import PostgresManager, to_dataframe
from datetime import datetime

with PostgresManager(**db_config) as db:
    # 날짜 범위로 조회
    results = db.select_date_range(
        table_name='observations',
        date_column='timestamp',
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        where={'region': 'AR12345'}
    )
    
    # pandas DataFrame으로 변환
    df = to_dataframe(results, parse_dates=['timestamp'])
    print(df.head())
```

## 주요 기능

### 1. 데이터베이스 관리

```python
# 데이터베이스 생성
db.create_database('new_db')

# 데이터베이스 삭제
db.drop_database('old_db', force=True)

# 데이터베이스 목록
databases = db.list_databases()
```

### 2. 스키마 관리

```python
# 스키마 생성
db.create_schema('research')

# 스키마 삭제
db.drop_schema('research', cascade=True)

# 스키마 목록
schemas = db.list_schemas()
```

### 3. 테이블 관리

```python
# 테이블 생성
db.create_table('experiments', {
    'id': 'SERIAL PRIMARY KEY',
    'name': 'VARCHAR(100) NOT NULL',
    'date': 'DATE',
    'result': 'JSONB'
}, schema='research')

# 테이블 삭제
db.drop_table('experiments', schema='research')

# 테이블 목록 (full info with size)
tables = db.list_tables(schema='research')
# Returns: [{'name': 'experiments', 'size': '8192 bytes'}, ...]

# 테이블 이름만 가져오기 (names only)
table_names = db.list_tables(schema='research', names_only=True)
# Returns: ['experiments', 'observations', ...]

# 테이블 구조 확인
columns = db.describe_table('experiments', schema='research')

# 모든 테이블 순회
for table_name in db.list_tables(names_only=True):
    print(f"Table: {table_name}")
    columns = db.describe_table(table_name)
    print(f"  Columns: {len(columns)}")

# 테이블 존재 확인
exists = db.table_exists('experiments', schema='research')
```

### 4. 데이터 작업 (CRUD)

#### INSERT

```python
# 단일 레코드 삽입
db.insert('users', {'name': 'Alice', 'email': 'alice@example.com'})

# 여러 레코드 삽입
db.insert('users', [
    {'name': 'Bob', 'email': 'bob@example.com'},
    {'name': 'Charlie', 'email': 'charlie@example.com'}
])

# ID 반환
user_id = db.insert('users', {'name': 'David'}, return_id=True)
```

#### SELECT

```python
# 전체 조회
all_users = db.select('users')

# WHERE 조건
users = db.select('users', where={'name': 'Alice'})

# 특정 컬럼만 선택
users = db.select('users', columns=['id', 'name'])

# 정렬과 제한
users = db.select('users', 
                  order_by='created_at DESC', 
                  limit=10)

# 복합 조건
users = db.select('users',
                  columns=['name', 'email'],
                  where={'name': 'Alice'},
                  order_by='created_at DESC',
                  limit=5)
```

#### SELECT - 날짜 범위 조회

날짜/timestamp 기반 조회를 위한 편리한 메서드:

```python
from datetime import datetime

# 기본 날짜 범위 조회
start = datetime(2024, 1, 1)
end = datetime(2024, 12, 31)

results = db.select_date_range(
    table_name='observations',
    date_column='date',
    start_date=start,
    end_date=end
)

# 추가 조건과 함께
results = db.select_date_range(
    table_name='observations',
    date_column='timestamp',
    start_date=start,
    end_date=end,
    where={'region': 'AR12345'},           # 추가 WHERE 조건
    columns=['timestamp', 'flux', 'region'], # 특정 컬럼만
    order_by='timestamp DESC',              # 정렬
    limit=100,                              # 제한
    inclusive_end=True                      # end 포함 (기본: False)
)

# pandas DataFrame으로 변환
from egghouse.database import to_dataframe

df = to_dataframe(results, parse_dates=['timestamp'])
print(df.head())
```


#### UPDATE

```python
# 데이터 업데이트
affected = db.update('users',
                     data={'email': 'newemail@example.com'},
                     where={'name': 'Alice'})
print(f"Updated {affected} rows")
```

#### DELETE

```python
# 데이터 삭제
deleted = db.delete('users', where={'name': 'Bob'})
print(f"Deleted {deleted} rows")
```

### 5. pandas DataFrame 변환

쿼리 결과를 pandas DataFrame으로 쉽게 변환:

```python
from egghouse.database import PostgresManager, to_dataframe
from datetime import datetime

with PostgresManager(**db_config) as db:
    # 쿼리 실행
    results = db.select('observations')
    
    # DataFrame으로 변환
    df = to_dataframe(results)
    
    # 날짜 컬럼 자동 파싱
    df = to_dataframe(results, parse_dates=['date', 'timestamp'])
    print(df.dtypes)  # date와 timestamp가 datetime64[ns]로 변환됨

# 날짜 범위 조회 + DataFrame 변환 (한 번에)
with PostgresManager(**db_config) as db:
    df = to_dataframe(
        db.select_date_range(
            'observations', 'timestamp',
            datetime(2024, 1, 1),
            datetime(2024, 12, 31)
        ),
        parse_dates=['timestamp']
    )
    
    # 시계열 인덱스 설정
    df.set_index('timestamp', inplace=True)
    
    # 리샘플링
    daily = df.resample('D').mean()
    print(daily.head())
```

### 6. 유틸리티 기능

```python
# 레코드 수 세기
total = db.count('users')
active_users = db.count('users', where={'status': 'active'})

# 테이블 비우기
db.truncate('users')

# 데이터베이스 최적화
db.vacuum('users', full=True, analyze=True)
```

### 7. 원시 SQL 실행

```python
# SELECT 쿼리
results = db.execute("""
    SELECT event_type, COUNT(*) as count
    FROM solar_events
    WHERE intensity > 7.0
    GROUP BY event_type
""", fetch=True)

# INSERT/UPDATE/DELETE
db.execute("UPDATE users SET status = 'active' WHERE last_login > NOW() - INTERVAL '30 days'")

# 파라미터화된 쿼리
users = db.execute(
    "SELECT * FROM users WHERE name = %s AND age > %s",
    params=('Alice', 25),
    fetch=True
)
```

## Context Manager 사용

자동으로 연결을 닫고 싶다면 context manager를 사용하세요:

```python
with PostgresManager(host='localhost', database='mydb', 
                     user='user', password='pass') as db:
    db.create_table('temp_data', {'id': 'SERIAL PRIMARY KEY', 'value': 'TEXT'})
    db.insert('temp_data', {'value': 'test'})
    results = db.select('temp_data')
    print(results)
# 자동으로 db.close() 호출됨
```

## 로깅 설정

### 기본 로깅

```python
# 쿼리 로깅 활성화
db = PostgresManager(
    host='localhost',
    database='mydb',
    user='user',
    password='pass',
    log_queries=True  # 모든 쿼리를 로그에 기록
)
```

### 커스텀 로거 사용

```python
import logging

# 커스텀 로거 생성
logger = logging.getLogger('my_research')
logger.setLevel(logging.DEBUG)

handler = logging.FileHandler('database_queries.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# PostgresManager에 전달
db = PostgresManager(
    host='localhost',
    database='mydb',
    user='user',
    password='pass',
    log_queries=True,
    logger=logger  # 커스텀 로거 사용
)
```

## 실전 예제: 태양 물리 데이터 관리

```python
from egghouse.database import PostgresManager, to_dataframe
from datetime import datetime
import pandas as pd

# 연결
db = PostgresManager(
    host='localhost',
    database='solar_physics',
    user='researcher',
    password='secure_password',
    log_queries=True
)

# 스키마 및 테이블 생성
db.create_schema('observations')

db.create_table('sdo_images', {
    'id': 'SERIAL PRIMARY KEY',
    'observation_time': 'TIMESTAMP NOT NULL',
    'wavelength': 'INTEGER NOT NULL',
    'filepath': 'TEXT NOT NULL',
    'quality_flag': 'INTEGER DEFAULT 0',
    'created_at': 'TIMESTAMP DEFAULT NOW()'
}, schema='observations')

db.create_table('solar_wind', {
    'id': 'SERIAL PRIMARY KEY',
    'timestamp': 'TIMESTAMP NOT NULL',
    'speed': 'FLOAT',
    'density': 'FLOAT',
    'temperature': 'FLOAT',
    'bz': 'FLOAT',
    'created_at': 'TIMESTAMP DEFAULT NOW()'
}, schema='observations')

# 데이터 삽입
sdo_data = [
    {'observation_time': '2025-01-20 12:00:00', 'wavelength': 193, 
     'filepath': '/data/sdo/20250120_1200_193.fits'},
    {'observation_time': '2025-01-20 12:15:00', 'wavelength': 193, 
     'filepath': '/data/sdo/20250120_1215_193.fits'}
]
db.insert('sdo_images', sdo_data, schema='observations')

# 날짜 범위로 데이터 조회
start = datetime(2025, 1, 1)
end = datetime(2025, 1, 31)

results = db.select_date_range(
    table_name='sdo_images',
    date_column='observation_time',
    start_date=start,
    end_date=end,
    where={'wavelength': 193},
    order_by='observation_time DESC',
    schema='observations'
)

# DataFrame으로 변환 및 분석
df = to_dataframe(results, parse_dates=['observation_time'])
print(f"Retrieved {len(df)} SDO images")
print(df.head())

# 시계열 분석
df.set_index('observation_time', inplace=True)
hourly_count = df.resample('H').size()
print(f"Images per hour:\n{hourly_count}")

# 통계 쿼리
stats = db.execute("""
    SELECT 
        wavelength,
        COUNT(*) as image_count,
        MIN(observation_time) as first_obs,
        MAX(observation_time) as last_obs
    FROM observations.sdo_images
    GROUP BY wavelength
""", fetch=True)

# 정리
db.close()
```

## 시계열 데이터 분석 예제

```python
from egghouse.database import PostgresManager, to_dataframe
from datetime import datetime

with PostgresManager(**db_config) as db:
    # OMNI 태양풍 데이터 조회
    start = datetime(2024, 3, 1)
    end = datetime(2024, 3, 31)
    
    results = db.select_date_range(
        table_name='omni_data',
        date_column='timestamp',
        start_date=start,
        end_date=end,
        columns=['timestamp', 'bz_gsm', 'v_sw', 'n_p'],
        order_by='timestamp'
    )
    
    # DataFrame 변환
    df = to_dataframe(results, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # 통계 분석
    print("Solar wind statistics:")
    print(df.describe())
    
    # High-speed solar wind 이벤트
    high_speed = df[df['v_sw'] > 500]
    print(f"\nHigh-speed events (>500 km/s): {len(high_speed)}")
    
    # 일별 평균 계산
    daily_avg = df.resample('D').mean()
    
    # CSV 저장
    df.to_csv('solar_wind_march_2024.csv')
    print("Data saved!")
```

## 에러 처리

```python
try:
    db = PostgresManager(host='localhost', database='mydb', 
                         user='user', password='pass')
    
    db.create_table('test', {'id': 'SERIAL PRIMARY KEY'})
    db.insert('test', {'id': 1})
    
except Exception as e:
    print(f"Error: {e}")
finally:
    if db:
        db.close()
```

## 팁

1. **대량 데이터 삽입**: 리스트로 한번에 삽입하면 훨씬 빠릅니다
   ```python
   # 느림: 루프에서 하나씩 삽입
   for record in data:
       db.insert('table', record)
   
   # 빠름: 한번에 삽입
   db.insert('table', data)
   ```

2. **WHERE 절은 필수**: UPDATE와 DELETE는 반드시 WHERE 조건이 필요합니다
   ```python
   # 에러 발생
   db.update('users', {'status': 'inactive'})  # WHERE 없음
   
   # 올바름
   db.update('users', {'status': 'inactive'}, where={'last_login': None})
   ```

3. **스키마 사용**: 데이터를 논리적으로 분리하세요
   ```python
   db.create_schema('raw_data')
   db.create_schema('processed_data')
   db.create_table('observations', {...}, schema='raw_data')
   ```

4. **테이블 존재 확인**: 생성 전에 확인하세요
   ```python
   if not db.table_exists('my_table'):
       db.create_table('my_table', columns)
   ```

5. **날짜 범위 조회**: `select_date_range()` 사용으로 코드 간소화
   ```python
   # 이전: Raw SQL 필요
   results = db.execute(
       "SELECT * FROM obs WHERE date >= %s AND date < %s",
       (start, end)
   )
   
   # 개선: 간단한 메서드
   results = db.select_date_range('obs', 'date', start, end)
   ```

6. **DataFrame 활용**: 데이터 분석에는 pandas DataFrame 사용
   ```python
   from egghouse.database import to_dataframe
   
   # 쿼리 + 변환을 한 줄로
   df = to_dataframe(
       db.select_date_range('observations', 'date', start, end),
       parse_dates=['date']
   )
   
   # 시계열 분석
   df.set_index('date', inplace=True)
   daily = df.resample('D').mean()
   ```

7. **대용량 데이터 처리**: 청크로 나눠서 처리
   ```python
   from datetime import timedelta
   
   current = start_date
   chunk_size = timedelta(days=7)
   
   while current < end_date:
       chunk_end = min(current + chunk_size, end_date)
       results = db.select_date_range('table', 'date', current, chunk_end)
       # 처리...
       current = chunk_end
   ```

## API 레퍼런스

### PostgresManager 클래스

#### 생성자
```python
PostgresManager(
    host: str,
    database: str,
    user: str,
    password: str,
    port: int = 5432,
    log_queries: bool = False,
    logger: logging.Logger = None
)
```

#### 주요 메서드

**데이터베이스 관리**
- `create_database(name: str)` - 데이터베이스 생성
- `drop_database(name: str, force: bool = False)` - 데이터베이스 삭제
- `list_databases() -> List[Dict]` - 데이터베이스 목록

**스키마 관리**
- `create_schema(name: str)` - 스키마 생성
- `drop_schema(name: str, cascade: bool = False)` - 스키마 삭제
- `list_schemas() -> List[Dict]` - 스키마 목록

**테이블 관리**
- `create_table(table_name: str, columns: Dict, schema: str = None)` - 테이블 생성
- `drop_table(table_name: str, schema: str = None, cascade: bool = False)` - 테이블 삭제
- `list_tables(schema: str = 'public', names_only: bool = False) -> Union[List[Dict], List[str]]` - 테이블 목록
- `describe_table(table_name: str, schema: str = None) -> List[Dict]` - 테이블 구조
- `table_exists(table_name: str, schema: str = None) -> bool` - 테이블 존재 확인

**데이터 작업**
- `insert(table_name: str, data: Union[Dict, List[Dict]], schema: str = None, return_id: bool = False)` - 데이터 삽입
- `select(table_name: str, columns: List[str] = None, where: Dict = None, schema: str = None, order_by: str = None, limit: int = None) -> List[Dict]` - 데이터 조회
- `select_date_range(table_name: str, date_column: str, start_date, end_date, columns: List[str] = None, where: Dict = None, schema: str = None, order_by: str = None, limit: int = None, inclusive_end: bool = False) -> List[Dict]` - 날짜 범위 조회
- `update(table_name: str, data: Dict, where: Dict, schema: str = None) -> int` - 데이터 업데이트
- `delete(table_name: str, where: Dict, schema: str = None) -> int` - 데이터 삭제

**유틸리티**
- `count(table_name: str, where: Dict = None, schema: str = None) -> int` - 레코드 수
- `truncate(table_name: str, schema: str = None)` - 테이블 비우기
- `vacuum(table_name: str = None, full: bool = False, analyze: bool = False)` - 데이터베이스 최적화
- `execute(query: str, params: tuple = None, fetch: bool = False) -> Optional[List[Dict]]` - 원시 SQL 실행
- `close()` - 연결 종료

### 유틸리티 함수

#### to_dataframe()
```python
to_dataframe(
    results: List[Dict],
    parse_dates: List[str] = None
) -> pd.DataFrame
```

쿼리 결과를 pandas DataFrame으로 변환합니다.

**파라미터:**
- `results`: 쿼리 결과 (딕셔너리 리스트)
- `parse_dates`: datetime으로 파싱할 컬럼 리스트 (선택)

**반환값:**
- pandas DataFrame

**예시:**
```python
from egghouse.database import to_dataframe

results = db.select('observations')
df = to_dataframe(results, parse_dates=['date', 'timestamp'])
```

## 제한사항

- 프로덕션 레벨이 아님 (연구용)
- 트랜잭션 관리 없음 (autocommit 사용)
- 커넥션 풀링 없음
- 복잡한 쿼리 빌더 없음
- ORM 기능 없음

고급 기능이 필요하다면 SQLAlchemy나 Django ORM을 고려하세요.

## 라이선스

연구용으로 자유롭게 사용하세요.
