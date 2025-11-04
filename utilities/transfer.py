#!/usr/bin/env python3
"""
HDD에서 로컬로 폴더 이동 (단일 HDD 최적화)
원본 파일은 이동 후 삭제됩니다.
"""

import shutil
import os
from pathlib import Path
from typing import List, Tuple
import time
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('move.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FolderMove:
    def __init__(self, source: str, destination: str):
        """
        Args:
            source: 원본 폴더 경로
            destination: 목적지 폴더 경로
        """
        self.source = Path(source)
        self.destination = Path(destination)
        self.total_files = 0
        self.moved_files = 0
        self.failed_files: List[Tuple[str, str]] = []
        
    def cleanup_parent_dirs(self, file_path: Path):
        """파일이 이동된 후 상위 디렉토리가 비었으면 즉시 삭제"""
        parent = file_path.parent
        
        # 원본 폴더 자체는 삭제하지 않음
        while parent != self.source and parent > self.source:
            try:
                # 빈 디렉토리만 삭제 (파일이나 하위 폴더가 있으면 예외 발생)
                parent.rmdir()
                logger.debug(f"빈 디렉토리 삭제: {parent}")
                parent = parent.parent
            except OSError:
                # 디렉토리가 비어있지 않으면 상위로 올라가지 않고 중단
                break
    
    def process_files_lazy(self) -> bool:
        """파일을 찾으면서 즉시 처리 (lazy evaluation)"""
        logger.info("파일 이동 시작 (파일 탐색과 동시 처리)...")
        start_time = time.time()
        
        self.total_files = 0
        self.moved_files = 0
        
        # 파일을 찾으면서 바로 처리
        for src_file in self.source.rglob('*'):
            if not src_file.is_file():
                continue
            
            self.total_files += 1
            
            # 상대 경로 계산
            rel_path = src_file.relative_to(self.source)
            dst_file = self.destination / rel_path
            
            # 파일 이동 (복사 → 검증 → 삭제)
            success, src_path, error = self.copy_file(src_file, dst_file)
            self.moved_files += 1
            
            if success:
                # 파일 이동 성공 시 즉시 상위 디렉토리 정리
                self.cleanup_parent_dirs(src_file)
                
                # 진행률 출력 (100개마다)
                if self.moved_files % 100 == 0:
                    elapsed = time.time() - start_time
                    speed = self.moved_files / elapsed if elapsed > 0 else 0
                    logger.info(f"진행: {self.moved_files:,}개 완료 ({speed:.1f}개/초)")
            else:
                self.failed_files.append((src_path, error))
                logger.error(f"실패: {src_path} - {error}")
        
        # 완료 통계
        elapsed_time = time.time() - start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"이동 완료!")
        logger.info(f"총 시간: {elapsed_time:.2f}초")
        logger.info(f"총 파일: {self.total_files:,}개")
        logger.info(f"성공: {self.moved_files - len(self.failed_files):,}개")
        logger.info(f"실패: {len(self.failed_files):,}개")
        if elapsed_time > 0:
            logger.info(f"평균 속도: {self.total_files / elapsed_time:.1f}개/초")
        logger.info(f"{'='*60}")
        
        return len(self.failed_files) == 0
    
    def copy_file(self, src_file: Path, dst_file: Path) -> Tuple[bool, str, str]:
        """
        단일 파일 이동 (복사 → 검증 → 삭제)
        
        Returns:
            (성공여부, 원본경로, 에러메시지)
        """
        try:
            # 목적지 디렉토리 생성
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 1단계: 파일 복사 (메타데이터 포함)
            shutil.copy2(src_file, dst_file)
            
            # 2단계: 검증 (파일 크기 비교)
            src_size = src_file.stat().st_size
            dst_size = dst_file.stat().st_size
            
            if src_size != dst_size:
                # 검증 실패 시 복사된 파일 삭제
                dst_file.unlink()
                return False, str(src_file), f"파일 크기 불일치 (원본: {src_size}, 복사본: {dst_size})"
            
            # 3단계: 검증 성공 시 원본 삭제
            src_file.unlink()
            
            return True, str(src_file), ""
                
        except Exception as e:
            # 오류 발생 시 복사된 파일이 있으면 삭제
            try:
                if dst_file.exists():
                    dst_file.unlink()
            except:
                pass
            return False, str(src_file), str(e)
    
    def cleanup_empty_dirs(self):
        """최종 정리: 남아있는 빈 디렉토리 삭제"""
        logger.info("최종 빈 디렉토리 정리 중...")
        removed_count = 0
        
        # 하위 디렉토리부터 처리 (역순 정렬)
        for dirpath in sorted(self.source.rglob('*'), reverse=True):
            if dirpath.is_dir():
                try:
                    # 빈 디렉토리만 삭제
                    dirpath.rmdir()
                    removed_count += 1
                    logger.debug(f"최종 정리: {dirpath}")
                except OSError:
                    # 디렉토리가 비어있지 않으면 무시
                    pass
        
        # 원본 폴더 자체도 비었으면 삭제
        try:
            self.source.rmdir()
            removed_count += 1
            logger.info(f"원본 폴더 삭제 완료: {self.source}")
        except OSError:
            logger.info(f"원본 폴더에 파일이 남아있어 삭제하지 않음: {self.source}")
        
        if removed_count > 0:
            logger.info(f"최종 정리: 빈 디렉토리 {removed_count}개 삭제 완료")
        else:
            logger.info(f"최종 정리: 삭제할 빈 디렉토리 없음 (실시간으로 모두 정리됨)")
    
    def verify_transfer(self) -> bool:
        """이동 검증 (원본 폴더 비어있는지, 목적지 파일 존재 확인)"""
        logger.info("이동 검증 중...")
        
        # 원본 폴더에 남은 파일 확인
        src_remaining = [f for f in self.source.rglob('*') if f.is_file()]
        if src_remaining:
            logger.warning(f"원본 폴더에 {len(src_remaining):,}개 파일이 남아있습니다.")
            for f in src_remaining[:10]:  # 최대 10개만 표시
                logger.warning(f"  남은 파일: {f}")
        
        # 목적지 파일 확인
        dst_files = [f for f in self.destination.rglob('*') if f.is_file()]
        
        if len(src_remaining) > 0:
            logger.error("⚠ 원본 폴더가 완전히 비워지지 않았습니다.")
            return False
        
        logger.info(f"✓ 검증 완료: {len(dst_files):,}개 파일이 목적지로 이동되었습니다.")
        logger.info("✓ 원본 폴더가 비워졌습니다.")
        return True


def main():
    # ========== 설정 ==========
    SOURCE_PATH = r"/Volumes/usbshare1/data/lasco"  # 네트워크 드라이브 경로
    DESTINATION_PATH = r"/Users/eunsupark/Data/lasco"  # 로컬 경로
    # ===========================
    
    try:
        # 경로 검증
        if not os.path.exists(SOURCE_PATH):
            logger.error(f"원본 경로가 존재하지 않습니다: {SOURCE_PATH}")
            return
        
        # 이동 객체 생성
        transfer = FolderMove(SOURCE_PATH, DESTINATION_PATH)
        
        # 파일 이동 (lazy evaluation - 찾으면서 바로 처리)
        success = transfer.process_files_lazy()
        
        # 검증 및 최종 정리
        if success:
            if transfer.verify_transfer():
                # 최종 빈 디렉토리 정리 (실시간으로 못 지운 것들)
                transfer.cleanup_empty_dirs()
        else:
            # 실패가 있어도 빈 디렉토리는 정리
            logger.info("일부 파일 이동 실패, 빈 디렉토리 정리 시작...")
            transfer.cleanup_empty_dirs()
        
        # 실패한 파일 목록 저장
        if transfer.failed_files:
            with open('failed_files.txt', 'w', encoding='utf-8') as f:
                for file_path, error in transfer.failed_files:
                    f.write(f"{file_path}\t{error}\n")
            logger.info(f"실패한 파일 목록: failed_files.txt")
        
    except KeyboardInterrupt:
        logger.warning("\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)


if __name__ == "__main__":
    main()
