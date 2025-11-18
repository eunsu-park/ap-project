"""
cleanup_duplicates.py

date_rounded로 그룹화하여 중복 데이터를 정리하는 스크립트
- date_rounded와 가장 가까운 date를 가진 레코드만 남김
- 나머지는 DB와 스토리지에서 삭제
"""

import os
from datetime import datetime
import psycopg2
from psycopg2.extras import execute_batch
from tqdm import tqdm

from utils_database import (
    DB_CONFIG,
    get_session,
    TABLE_MODELS
)


DATA_ROOT = "/Users/eunsupark/Data/sdo/fits"


def find_duplicates(table_name, date_rounded_start=None, date_rounded_end=None):
    """
    중복된 date_rounded를 찾기
    
    Args:
        table_name: 테이블명
        date_rounded_start: 검색 시작 시간 (선택)
        date_rounded_end: 검색 종료 시간 (선택)
    
    Returns:
        list: 중복이 있는 date_rounded 리스트
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # 중복 찾기 쿼리
        if date_rounded_start and date_rounded_end:
            query = f"""
                SELECT date_rounded, COUNT(*) as count
                FROM {table_name}
                WHERE date_rounded >= %s AND date_rounded <= %s
                GROUP BY date_rounded
                HAVING COUNT(*) > 1
                ORDER BY date_rounded
            """
            cursor.execute(query, (date_rounded_start, date_rounded_end))
        else:
            query = f"""
                SELECT date_rounded, COUNT(*) as count
                FROM {table_name}
                GROUP BY date_rounded
                HAVING COUNT(*) > 1
                ORDER BY date_rounded
            """
            cursor.execute(query)
        
        duplicates = [(row[0], row[1]) for row in cursor.fetchall()]
        
        cursor.close()
        conn.close()
        
        return duplicates
    
    except psycopg2.Error as e:
        print(f"✗ Database error: {e}")
        if conn:
            conn.close()
        return []


def cleanup_duplicates_for_time(table_name, date_rounded, dry_run=True):
    """
    특정 date_rounded의 중복 데이터 정리
    
    Args:
        table_name: 테이블명
        date_rounded: 정리할 date_rounded
        dry_run: True면 실제로 삭제하지 않고 확인만
    
    Returns:
        dict: 정리 결과 통계
    """
    stats = {
        'total_records': 0,
        'kept_record': None,
        'deleted_records': [],
        'db_deleted': 0,
        'file_deleted': 0,
        'file_delete_failed': 0
    }
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # 해당 date_rounded의 모든 레코드 조회
        query = f"""
            SELECT id, date_rounded, date, file_name
            FROM {table_name}
            WHERE date_rounded = %s
            ORDER BY date
        """
        cursor.execute(query, (date_rounded,))
        records = cursor.fetchall()
        
        stats['total_records'] = len(records)
        
        if len(records) <= 1:
            cursor.close()
            conn.close()
            return stats
        
        # date_rounded와 가장 가까운 레코드 찾기
        best_record = min(records, key=lambda r: abs((r[1] - r[2]).total_seconds()))
        best_id = best_record[0]
        
        stats['kept_record'] = {
            'id': best_record[0],
            'date_rounded': best_record[1],
            'date': best_record[2],
            'file_name': best_record[3],
            'diff_seconds': abs((best_record[1] - best_record[2]).total_seconds())
        }
        
        # 삭제할 레코드들
        to_delete = [r for r in records if r[0] != best_id]
        
        for record in to_delete:
            stats['deleted_records'].append({
                'id': record[0],
                'date_rounded': record[1],
                'date': record[2],
                'file_name': record[3],
                'diff_seconds': abs((record[1] - record[2]).total_seconds())
            })
        
        if not dry_run:
            # DB에서 삭제
            delete_ids = [r[0] for r in to_delete]
            if delete_ids:
                delete_query = f"DELETE FROM {table_name} WHERE id = ANY(%s)"
                cursor.execute(delete_query, (delete_ids,))
                stats['db_deleted'] = cursor.rowcount
                conn.commit()
            
            # 스토리지에서 파일 삭제
            instrument = 'aia' if table_name.startswith('aia') else 'hmi'
            
            for record in to_delete:
                file_name = record[3]
                file_path = f"{DATA_ROOT}/{instrument}/{file_name}"
                
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        stats['file_deleted'] += 1
                    else:
                        print(f"  ⚠ File not found: {file_name}")
                except Exception as e:
                    print(f"  ✗ Failed to delete file {file_name}: {e}")
                    stats['file_delete_failed'] += 1
        
        cursor.close()
        conn.close()
        
    except psycopg2.Error as e:
        print(f"✗ Database error: {e}")
        if conn:
            conn.rollback()
            conn.close()
    
    return stats


def cleanup_all_duplicates(table_name, date_rounded_start=None, date_rounded_end=None, dry_run=True):
    """
    테이블의 모든 중복 데이터 정리
    
    Args:
        table_name: 테이블명
        date_rounded_start: 검색 시작 시간 (선택)
        date_rounded_end: 검색 종료 시간 (선택)
        dry_run: True면 실제로 삭제하지 않고 확인만
    
    Returns:
        dict: 전체 통계
    """
    print(f"\n{'='*60}")
    print(f"테이블: {table_name}")
    print(f"모드: {'DRY RUN (확인만)' if dry_run else 'ACTUAL DELETE (실제 삭제)'}")
    print(f"{'='*60}")
    
    # 중복 찾기
    print("\n중복된 date_rounded 찾는 중...")
    duplicates = find_duplicates(table_name, date_rounded_start, date_rounded_end)
    
    if not duplicates:
        print("중복된 데이터가 없습니다!")
        return None
    
    print(f"총 {len(duplicates)}개의 중복된 date_rounded 발견")
    
    # 전체 통계
    total_stats = {
        'duplicate_times': len(duplicates),
        'total_records': 0,
        'total_deleted_db': 0,
        'total_deleted_files': 0,
        'total_failed_files': 0,
        'details': []
    }
    
    # 각 date_rounded 처리
    for date_rounded, count in tqdm(duplicates, desc="Processing"):
        stats = cleanup_duplicates_for_time(table_name, date_rounded, dry_run)
        
        if stats['total_records'] > 1:
            total_stats['total_records'] += stats['total_records']
            
            # dry_run일 때는 삭제될 개수를 예상치로 계산
            if dry_run:
                expected_deletes = len(stats['deleted_records'])
                total_stats['total_deleted_db'] += expected_deletes
                total_stats['total_deleted_files'] += expected_deletes
            else:
                total_stats['total_deleted_db'] += stats['db_deleted']
                total_stats['total_deleted_files'] += stats['file_deleted']
                total_stats['total_failed_files'] += stats['file_delete_failed']
            
            total_stats['details'].append({
                'date_rounded': date_rounded,
                'count': count,
                'stats': stats
            })
    
    # 결과 출력
    print(f"\n{'='*60}")
    print("정리 결과")
    print(f"{'='*60}")
    print(f"중복 시간대:           {total_stats['duplicate_times']}")
    print(f"총 레코드:             {total_stats['total_records']}")
    print(f"유지:                  {total_stats['duplicate_times']}")
    
    if dry_run:
        print(f"삭제 예정 (DB):        {total_stats['total_deleted_db']}")
        print(f"삭제 예정 (파일):      {total_stats['total_deleted_files']}")
    else:
        print(f"DB 삭제:               {total_stats['total_deleted_db']}")
        print(f"파일 삭제:             {total_stats['total_deleted_files']}")
        print(f"파일 삭제 실패:        {total_stats['total_failed_files']}")
    
    print(f"{'='*60}")
    
    # 상세 정보 출력 (처음 5개만)
    if total_stats['details'] and dry_run:
        print(f"\n상세 정보 (처음 5개):")
        for detail in total_stats['details'][:5]:
            date_rounded = detail['date_rounded']
            stats = detail['stats']
            
            print(f"\n  {date_rounded}:")
            print(f"    총 {stats['total_records']}개 레코드")
            
            if stats['kept_record']:
                kept = stats['kept_record']
                print(f"    ✓ 유지: {kept['file_name']}")
                print(f"       차이: {kept['diff_seconds']:.1f}초")
            
            for deleted in stats['deleted_records']:
                print(f"    ✗ 삭제: {deleted['file_name']}")
                print(f"       차이: {deleted['diff_seconds']:.1f}초")
    
    return total_stats


# ==========================================================
# 사용 예시
# ==========================================================

if __name__ == "__main__":
    import sys
    
    # 테이블 선택
    table_name = 'aia_193'  # 또는 'aia_211', 'hmi_magnetogram'
    
    # 시간 범위 (선택사항)
    date_start = None  # datetime(2014, 4, 1)
    date_end = None    # datetime(2014, 4, 30)
    
    print("="*60)
    print("중복 데이터 정리 도구")
    print("="*60)
    print(f"테이블: {table_name}")
    if date_start and date_end:
        print(f"범위: {date_start} ~ {date_end}")
    else:
        print(f"범위: 전체")
    
    # 1단계: DRY RUN (확인만)
    print("\n1단계: 확인 모드 (실제 삭제하지 않음)")
    stats = cleanup_all_duplicates(
        table_name,
        date_start,
        date_end,
        dry_run=True
    )
    
    if stats is None:
        print("\n정리할 데이터가 없습니다.")
        sys.exit(0)
    
    # 2단계: 사용자 확인
    print(f"\n{'='*60}")
    print("⚠️  경고: 실제로 삭제하시겠습니까?")
    print(f"{'='*60}")
    print(f"삭제될 DB 레코드: {stats['total_deleted_db']}개")
    print(f"삭제될 파일:      {stats['total_deleted_files']}개")
    print(f"\n이 작업은 되돌릴 수 없습니다!")
    
    confirm = input("\n실제로 삭제하시겠습니까? (yes/no): ")
    
    if confirm.lower() == 'yes':
        # 3단계: 실제 삭제
        print("\n2단계: 실제 삭제 모드")
        final_stats = cleanup_all_duplicates(
            table_name,
            date_start,
            date_end,
            dry_run=False
        )
        
        print("\n✓ 정리 완료!")
    else:
        print("\n취소되었습니다.")

# """
# cleanup_duplicates.py

# date_rounded로 그룹화하여 중복 데이터를 정리하는 스크립트
# - date_rounded와 가장 가까운 date를 가진 레코드만 남김
# - 나머지는 DB와 스토리지에서 삭제
# """

# import os
# from datetime import datetime
# import psycopg2
# from psycopg2.extras import execute_batch
# from tqdm import tqdm

# from utils_database import (
#     DB_CONFIG,
#     get_session,
#     TABLE_MODELS
# )


# DATA_ROOT = "/Users/eunsupark/Data/sdo/fits"


# def find_duplicates(table_name, date_rounded_start=None, date_rounded_end=None):
#     """
#     중복된 date_rounded를 찾기
    
#     Args:
#         table_name: 테이블명
#         date_rounded_start: 검색 시작 시간 (선택)
#         date_rounded_end: 검색 종료 시간 (선택)
    
#     Returns:
#         list: 중복이 있는 date_rounded 리스트
#     """
#     try:
#         conn = psycopg2.connect(**DB_CONFIG)
#         cursor = conn.cursor()
        
#         # 중복 찾기 쿼리
#         if date_rounded_start and date_rounded_end:
#             query = f"""
#                 SELECT date_rounded, COUNT(*) as count
#                 FROM {table_name}
#                 WHERE date_rounded >= %s AND date_rounded <= %s
#                 GROUP BY date_rounded
#                 HAVING COUNT(*) > 1
#                 ORDER BY date_rounded
#             """
#             cursor.execute(query, (date_rounded_start, date_rounded_end))
#         else:
#             query = f"""
#                 SELECT date_rounded, COUNT(*) as count
#                 FROM {table_name}
#                 GROUP BY date_rounded
#                 HAVING COUNT(*) > 1
#                 ORDER BY date_rounded
#             """
#             cursor.execute(query)
        
#         duplicates = [(row[0], row[1]) for row in cursor.fetchall()]
        
#         cursor.close()
#         conn.close()
        
#         return duplicates
    
#     except psycopg2.Error as e:
#         print(f"✗ Database error: {e}")
#         if conn:
#             conn.close()
#         return []


# def cleanup_duplicates_for_time(table_name, date_rounded, dry_run=True):
#     """
#     특정 date_rounded의 중복 데이터 정리
    
#     Args:
#         table_name: 테이블명
#         date_rounded: 정리할 date_rounded
#         dry_run: True면 실제로 삭제하지 않고 확인만
    
#     Returns:
#         dict: 정리 결과 통계
#     """
#     stats = {
#         'total_records': 0,
#         'kept_record': None,
#         'deleted_records': [],
#         'db_deleted': 0,
#         'file_deleted': 0,
#         'file_delete_failed': 0
#     }
    
#     try:
#         conn = psycopg2.connect(**DB_CONFIG)
#         cursor = conn.cursor()
        
#         # 해당 date_rounded의 모든 레코드 조회
#         query = f"""
#             SELECT id, date_rounded, date, file_name
#             FROM {table_name}
#             WHERE date_rounded = %s
#             ORDER BY date
#         """
#         cursor.execute(query, (date_rounded,))
#         records = cursor.fetchall()
        
#         stats['total_records'] = len(records)
        
#         if len(records) <= 1:
#             cursor.close()
#             conn.close()
#             return stats
        
#         # date_rounded와 가장 가까운 레코드 찾기
#         best_record = min(records, key=lambda r: abs((r[1] - r[2]).total_seconds()))
#         best_id = best_record[0]
        
#         stats['kept_record'] = {
#             'id': best_record[0],
#             'date_rounded': best_record[1],
#             'date': best_record[2],
#             'file_name': best_record[3],
#             'diff_seconds': abs((best_record[1] - best_record[2]).total_seconds())
#         }
        
#         # 삭제할 레코드들
#         to_delete = [r for r in records if r[0] != best_id]
        
#         for record in to_delete:
#             stats['deleted_records'].append({
#                 'id': record[0],
#                 'date_rounded': record[1],
#                 'date': record[2],
#                 'file_name': record[3],
#                 'diff_seconds': abs((record[1] - record[2]).total_seconds())
#             })
        
#         if not dry_run:
#             # DB에서 삭제
#             delete_ids = [r[0] for r in to_delete]
#             if delete_ids:
#                 delete_query = f"DELETE FROM {table_name} WHERE id = ANY(%s)"
#                 cursor.execute(delete_query, (delete_ids,))
#                 stats['db_deleted'] = cursor.rowcount
#                 conn.commit()
            
#             # 스토리지에서 파일 삭제
#             instrument = 'aia' if table_name.startswith('aia') else 'hmi'
            
#             for record in to_delete:
#                 file_name = record[3]
#                 file_path = f"{DATA_ROOT}/{instrument}/{file_name}"
                
#                 try:
#                     if os.path.exists(file_path):
#                         os.remove(file_path)
#                         stats['file_deleted'] += 1
#                     else:
#                         print(f"  ⚠ File not found: {file_name}")
#                 except Exception as e:
#                     print(f"  ✗ Failed to delete file {file_name}: {e}")
#                     stats['file_delete_failed'] += 1
        
#         cursor.close()
#         conn.close()
        
#     except psycopg2.Error as e:
#         print(f"✗ Database error: {e}")
#         if conn:
#             conn.rollback()
#             conn.close()
    
#     return stats


# def cleanup_all_duplicates(table_name, date_rounded_start=None, date_rounded_end=None, dry_run=True):
#     """
#     테이블의 모든 중복 데이터 정리
    
#     Args:
#         table_name: 테이블명
#         date_rounded_start: 검색 시작 시간 (선택)
#         date_rounded_end: 검색 종료 시간 (선택)
#         dry_run: True면 실제로 삭제하지 않고 확인만
    
#     Returns:
#         dict: 전체 통계
#     """
#     print(f"\n{'='*60}")
#     print(f"테이블: {table_name}")
#     print(f"모드: {'DRY RUN (확인만)' if dry_run else 'ACTUAL DELETE (실제 삭제)'}")
#     print(f"{'='*60}")
    
#     # 중복 찾기
#     print("\n중복된 date_rounded 찾는 중...")
#     duplicates = find_duplicates(table_name, date_rounded_start, date_rounded_end)
    
#     if not duplicates:
#         print("중복된 데이터가 없습니다!")
#         return None
    
#     print(f"총 {len(duplicates)}개의 중복된 date_rounded 발견")
    
#     # 전체 통계
#     total_stats = {
#         'duplicate_times': len(duplicates),
#         'total_records': 0,
#         'total_deleted_db': 0,
#         'total_deleted_files': 0,
#         'total_failed_files': 0,
#         'details': []
#     }
    
#     # 각 date_rounded 처리
#     for date_rounded, count in tqdm(duplicates, desc="Processing"):
#         stats = cleanup_duplicates_for_time(table_name, date_rounded, dry_run)
        
#         if stats['total_records'] > 1:
#             total_stats['total_records'] += stats['total_records']
#             total_stats['total_deleted_db'] += stats['db_deleted']
#             total_stats['total_deleted_files'] += stats['file_deleted']
#             total_stats['total_failed_files'] += stats['file_delete_failed']
#             total_stats['details'].append({
#                 'date_rounded': date_rounded,
#                 'count': count,
#                 'stats': stats
#             })
    
#     # 결과 출력
#     print(f"\n{'='*60}")
#     print("정리 결과")
#     print(f"{'='*60}")
#     print(f"중복 시간대:           {total_stats['duplicate_times']}")
#     print(f"총 레코드:             {total_stats['total_records']}")
#     print(f"유지:                  {total_stats['duplicate_times']}")
#     print(f"DB 삭제:               {total_stats['total_deleted_db']}")
#     print(f"파일 삭제:             {total_stats['total_deleted_files']}")
#     print(f"파일 삭제 실패:        {total_stats['total_failed_files']}")
#     print(f"{'='*60}")
    
#     # 상세 정보 출력 (처음 5개만)
#     if total_stats['details'] and dry_run:
#         print(f"\n상세 정보 (처음 5개):")
#         for detail in total_stats['details'][:5]:
#             date_rounded = detail['date_rounded']
#             stats = detail['stats']
            
#             print(f"\n  {date_rounded}:")
#             print(f"    총 {stats['total_records']}개 레코드")
            
#             if stats['kept_record']:
#                 kept = stats['kept_record']
#                 print(f"    ✓ 유지: {kept['file_name']}")
#                 print(f"       차이: {kept['diff_seconds']:.1f}초")
            
#             for deleted in stats['deleted_records']:
#                 print(f"    ✗ 삭제: {deleted['file_name']}")
#                 print(f"       차이: {deleted['diff_seconds']:.1f}초")
    
#     return total_stats


# # ==========================================================
# # 사용 예시
# # ==========================================================

# if __name__ == "__main__":
#     import sys
    
#     # 테이블 선택
#     table_name = 'aia_211'  # 또는 'aia_211', 'hmi_magnetogram'
    
#     # 시간 범위 (선택사항)
#     date_start = None  # datetime(2014, 4, 1)
#     date_end = None    # datetime(2014, 4, 30)
    
#     print("="*60)
#     print("중복 데이터 정리 도구")
#     print("="*60)
#     print(f"테이블: {table_name}")
#     if date_start and date_end:
#         print(f"범위: {date_start} ~ {date_end}")
#     else:
#         print(f"범위: 전체")
    
#     # 1단계: DRY RUN (확인만)
#     print("\n1단계: 확인 모드 (실제 삭제하지 않음)")
#     stats = cleanup_all_duplicates(
#         table_name,
#         date_start,
#         date_end,
#         dry_run=True
#     )
    
#     if stats is None:
#         print("\n정리할 데이터가 없습니다.")
#         sys.exit(0)
    
#     # 2단계: 사용자 확인
#     print(f"\n{'='*60}")
#     print("⚠️  경고: 실제로 삭제하시겠습니까?")
#     print(f"{'='*60}")
#     print(f"삭제될 DB 레코드: {stats['total_deleted_db']}개")
#     print(f"삭제될 파일:      {stats['total_deleted_files']}개")
#     print(f"\n이 작업은 되돌릴 수 없습니다!")
    
#     confirm = input("\n실제로 삭제하시겠습니까? (yes/no): ")
    
#     if confirm.lower() == 'yes':
#         # 3단계: 실제 삭제
#         print("\n2단계: 실제 삭제 모드")
#         final_stats = cleanup_all_duplicates(
#             table_name,
#             date_start,
#             date_end,
#             dry_run=False
#         )
        
#         print("\n✓ 정리 완료!")
#     else:
#         print("\n취소되었습니다.")