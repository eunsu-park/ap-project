import argparse
import datetime

from utils import get_file_list, download_parallel, parse_date_range


def main():
    parser = argparse.ArgumentParser(description="SDO HMI JP2 파일 다운로드")
    parser.add_argument("--start-date", type=str, help="시작 날짜 (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="종료 날짜 (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, help="최근 며칠 다운로드")

    # 다운로드 옵션
    parser.add_argument("--extensions", type=str, nargs="+", default=["jp2"],
                        help="파일 확장자")
    parser.add_argument("--wavelengths", type=int, nargs="+", default=[193, 211],
                        help="파장 (예: 193, 211, 304, 335, 94, 131, 171)")
    parser.add_argument("--destination-root", type=str, default="./data",
                        help="저장 디렉토리")
    parser.add_argument("--parallel", type=int, default=1, help="병렬 다운로드 수")
    parser.add_argument("--overwrite", action="store_true", help="기존 파일 덮어쓰기")
    parser.add_argument("--max-retries", type=int, default=3, help="최대 재시도 횟수")

    args = parser.parse_args()
    args.mission_start_date = "2010-09-01"

    start_date, end_date = parse_date_range(args)
    print(f"Downloading from {start_date} to {end_date}")

    current_date = start_date
    while current_date <= end_date:
        for wave in args.wavelengths:
            base_url = f"https://gs671-suske.ndc.nasa.gov/jp2/AIA/{current_date:%Y/%m/%d}/{wave}"
            save_dir = f"{args.destination_root}/sdo_jp2/aia/{wave}/{current_date:%Y}/{current_date:%Y%m%d}"

            print(f"Fetching file list from {base_url}")
            file_list = get_file_list(base_url, args.extensions)
            if not file_list:
                print(f"No files found at {base_url}")
                continue

            download_tasks = []
            for filename in file_list:
                source = f"{base_url}/{filename}"
                ## Filterting
                # f"%Y_%m_%d__%H_%M_%S_%f__SDO_{instrument_upper}_{instrument_upper}_{wavelength}.jp2"
                filedate = datetime.datetime.strptime(filename, f"%Y_%m_%d__%H_%M_%S_%f__SDO_AIA_AIA_{wave}.jp2")
                if filedate.hour in (2,3, 5,6, 8,9, 11,12, 14,15, 17,18, 20,21, 23,0) :
                    destination = f"{save_dir}/{filename}"
                    download_tasks.append((source, destination))

            result = download_parallel(download_tasks, overwrite=args.overwrite,
                                       max_retries=args.max_retries, parallel=args.parallel)
            print(f"Downloaded: {result['downloaded']}, Failed: {result['failed']}")

        current_date += datetime.timedelta(days=1)


if __name__ == "__main__" :
    main()
