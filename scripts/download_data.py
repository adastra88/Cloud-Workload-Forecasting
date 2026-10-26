"""Download and verify the fixed ``data-v1`` release.

Usage
-----
    python scripts/download_data.py --dest challenge_data

The script:
1. downloads ``data-v1.tar.gz`` from the pinned release URL (or uses
   ``--archive`` for an already-downloaded copy),
2. verifies the archive SHA-256 against the digest recorded at
   ``--expected-archive-sha256`` (default: the digest published for data-v1),
3. extracts it into ``--dest``,
4. verifies every entry of ``SHA256SUMS``,
5. writes ``reports/artifacts/data_checksums.csv``.

It never writes anything outside ``--dest`` and ``reports/artifacts``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cwf.data import write_sha256_verification  # noqa: E402

RELEASE_URL = "https://github.com/atomityhq/forecasting-challenge/releases/download/data-v1/data-v1.tar.gz"
#: Digest published in the data-v1 release metadata.
EXPECTED_ARCHIVE_SHA256 = "d04d402ac1719c1f5b869e3699fc3d57f6f8273751f1ef19bedb29397b39caeb"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}\n         -> {dest}")
    with urllib.request.urlopen(url) as response, open(dest, "wb") as out:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default="challenge_data", help="extraction directory")
    parser.add_argument("--archive", default=None, help="use an existing data-v1.tar.gz instead of downloading")
    parser.add_argument("--expected-archive-sha256", default=EXPECTED_ARCHIVE_SHA256)
    parser.add_argument("--skip-extract", action="store_true", help="only verify checksums")
    args = parser.parse_args(argv)

    dest = Path(args.dest)
    if not args.skip_extract:
        if args.archive:
            archive = Path(args.archive)
        else:
            archive = Path("data-v1.tar.gz")
            if not archive.exists():
                download(RELEASE_URL, archive)
        actual = sha256_of(archive)
        if actual != args.expected_archive_sha256:
            print(f"ERROR: archive sha256 {actual} != expected {args.expected_archive_sha256}", file=sys.stderr)
            return 2
        print(f"archive sha256 OK ({actual})")
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(dest, filter="data")
        print(f"extracted into {dest}")

    table = write_sha256_verification(dest, ROOT / "reports" / "artifacts" / "data_checksums.csv")
    print(table.to_string(index=False))
    ok = bool(len(table)) and bool(table["ok"].all())
    print("SHA256SUMS: " + ("ALL OK" if ok else "MISMATCH — do not proceed"))
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
