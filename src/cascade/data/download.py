"""Download raw XMC benchmark archives (PECOS xmc-base format).

Each archive (eurlex-4k.tar.gz, wiki10-31k.tar.gz) extracts to a folder containing:
  X.trn.txt, X.tst.txt        - one raw document per line
  Y.trn.npz, Y.tst.npz        - sparse (doc x label) label matrices
  output-items.txt            - one label surface-text per line

Primary source: PECOS dataset mirror on archive.org. If that fails, the user is
pointed to the Extreme Classification Repository as a manual fallback.
"""

import tarfile
from pathlib import Path

import requests

from cascade.utils.io import ensure_dir, load_yaml
from cascade.utils.logging import get_logger

logger = get_logger(__name__)

REQUIRED_FILES = ["X.trn.txt", "X.tst.txt", "Y.trn.npz", "Y.tst.npz", "output-items.txt"]

XC_REPO_FALLBACK_URL = "https://manikvarma.org/downloads/XC/XMLRepository.html"


def _find_dataset_root(extract_dir: Path) -> Path:
    """PECOS archives nest contents under xmc-base/<dataset>/; locate the real root."""
    if all((extract_dir / f).exists() for f in REQUIRED_FILES):
        return extract_dir
    for candidate in extract_dir.rglob(REQUIRED_FILES[0]):
        root = candidate.parent
        if all((root / f).exists() for f in REQUIRED_FILES):
            return root
    raise FileNotFoundError(
        f"Could not locate required files {REQUIRED_FILES} under {extract_dir}. "
        f"If the PECOS mirror has moved, download manually from {XC_REPO_FALLBACK_URL} "
        f"and place the files directly under {extract_dir}."
    )


def download_dataset(dataset_cfg: dict, force: bool = False) -> Path:
    raw_dir = ensure_dir(dataset_cfg["raw_dir"])
    url = dataset_cfg["pecos_url"]
    archive_path = raw_dir / Path(url).name

    if force or not archive_path.exists():
        logger.info(f"Downloading {url} -> {archive_path}")
        try:
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            with open(archive_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        except requests.RequestException as e:
            raise RuntimeError(
                f"Failed to download {url}: {e}. "
                f"Fallback: download manually from {XC_REPO_FALLBACK_URL} and place "
                f"{REQUIRED_FILES} under {raw_dir}."
            ) from e
    else:
        logger.info(f"Archive already present at {archive_path}, skipping download.")

    marker = raw_dir / ".extracted"
    if force or not marker.exists():
        logger.info(f"Extracting {archive_path} -> {raw_dir}")
        with tarfile.open(archive_path) as tar:
            tar.extractall(raw_dir, filter="data")
        marker.touch()

    dataset_root = _find_dataset_root(raw_dir)
    logger.info(f"Dataset root resolved to {dataset_root}")
    return dataset_root


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="e.g. eurlex-4k or wiki10-31k")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(f"configs/datasets/{args.dataset}.yaml")
    download_dataset(cfg, force=args.force)


if __name__ == "__main__":
    main()
