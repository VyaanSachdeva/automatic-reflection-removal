"""
Downloads KITTI raw drives and keeps only the left colour camera
(image_02/data), as clean frames for VRGS: each drive becomes its own scene
folder, organized/KITTI_Clean/drive_XXXX/rgb/. The zips also hold the other
cameras and the laser scans (about 4 MB per frame), so they are deleted after
extraction. Resumable: re-run it after a dropped connection.

    python download_kitti.py 0002 0020 0079 0113

KITTI raw is CC BY-NC-SA 3.0 (https://www.cvlibs.net/datasets/kitti/raw_data.php).
"""
from __future__ import annotations

import subprocess
import sys
import zipfile

from raindrop_paths import RAINDROP_ROOT

URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/{n}/{n}_sync.zip"
DOWNLOADS = RAINDROP_ROOT / "downloads"
CLEAN = RAINDROP_ROOT / "organized" / "KITTI_Clean"


def fetch(drive: str) -> None:
    name = f"2011_09_26_drive_{drive}"
    out_dir = CLEAN / f"drive_{drive}" / "rgb"
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"{name}: already extracted ({len(list(out_dir.iterdir()))} frames)", flush=True)
        return
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    zip_path = DOWNLOADS / f"{name}_sync.zip"
    subprocess.run(["curl.exe", "-L", "--retry", "50", "--retry-all-errors", "--retry-delay", "5", "-C", "-",
                    "-sS", "-o", str(zip_path), URL.format(n=name)], check=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if "/image_02/data/" in m and m.endswith(".png")]
        for m in members:
            (out_dir / m.rsplit("/", 1)[1]).write_bytes(z.read(m))
    zip_path.unlink()
    print(f"{name}: {len(members)} frames -> {out_dir}", flush=True)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        fetch(d)
