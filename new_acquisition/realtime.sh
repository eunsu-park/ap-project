nohup python download_sdo_hmi_jp2.py    --parallel 4 --destination-root /Volumes/usbshare1/data --days 14 1>/dev/null 2>&1 &
nohup python download_sdo_aia_jp2.py.   --parallel 4 --destination-root /Volumes/usbshare1/data --days 14 1>/dev/null 2>&1 &
nohup python download_lasco_realtime.py --parallel 4 --destination-root /Volumes/usbshare1/data --days 14 1>/dev/null 2>&1 &
