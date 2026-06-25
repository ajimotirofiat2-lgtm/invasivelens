"""
Utilities for robust data downloading and validation.

Provides:
  - Retry logic with exponential backoff for network requests
  - MD5/SHA256 checksums for file integrity
  - Download resumption from partial failures
"""
import hashlib
import time
from pathlib import Path
from typing import Callable, Optional

import requests
from tqdm import tqdm


class RetryConfig:
    """Configuration for exponential backoff retry strategy."""
    def __init__(
        self,
        max_retries: int = 5,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor

    def get_wait_time(self, attempt: int) -> float:
        """Calculate exponential backoff wait time."""
        delay = self.initial_delay * (self.backoff_factor ** attempt)
        return min(delay, self.max_delay)


def retry_with_backoff(
    func: Callable,
    *args,
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable] = None,
    **kwargs,
):
    """
    Call func with exponential backoff retry on exception.

    Args:
        func: Callable to execute
        args, kwargs: Arguments to pass to func
        config: RetryConfig (uses defaults if None)
        on_retry: Optional callback(attempt, delay, exception) called before retrying

    Returns:
        Result of func(*args, **kwargs)

    Raises:
        Last exception if all retries exhausted
    """
    config = config or RetryConfig()
    last_exception = None

    for attempt in range(config.max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < config.max_retries - 1:
                delay = config.get_wait_time(attempt)
                if on_retry:
                    on_retry(attempt + 1, delay, e)
                time.sleep(delay)
            else:
                raise

    raise last_exception


def download_file(
    url: str,
    dest_path: Path,
    timeout: float = 30.0,
    retry_config: Optional[RetryConfig] = None,
    expected_md5: Optional[str] = None,
) -> Path:
    """
    Download a file with retry logic and optional checksum verification.

    Args:
        url: URL to download from
        dest_path: Where to save file
        timeout: Request timeout in seconds
        retry_config: RetryConfig for retries (default: 3 retries)
        expected_md5: If provided, verify MD5 after download

    Returns:
        Path to downloaded file

    Raises:
        requests.RequestException: If all retries fail
        ValueError: If MD5 checksum doesn't match
    """
    retry_config = retry_config or RetryConfig(max_retries=3)
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip if file already exists and checksum matches
    if dest_path.exists() and expected_md5:
        if compute_md5(dest_path) == expected_md5:
            return dest_path

    def _download():
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        total_size = int(resp.headers.get("content-length", 0))

        with open(dest_path, "wb") as f:
            if total_size > 0:
                with tqdm(total=total_size, unit="B", unit_scale=True, desc=dest_path.name) as pbar:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

    def _retry_callback(attempt, delay, exc):
        print(f"  Retry {attempt}/{retry_config.max_retries} for {url} in {delay:.1f}s... ({exc})")

    retry_with_backoff(_download, config=retry_config, on_retry=_retry_callback)

    # Verify checksum if provided
    if expected_md5:
        actual_md5 = compute_md5(dest_path)
        if actual_md5 != expected_md5:
            raise ValueError(
                f"MD5 mismatch for {dest_path}: expected {expected_md5}, got {actual_md5}"
            )

    return dest_path


def compute_md5(file_path: Path | str) -> str:
    """Compute MD5 checksum of a file."""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def compute_sha256(file_path: Path | str) -> str:
    """Compute SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def validate_image_file(file_path: Path | str) -> tuple[bool, Optional[str]]:
    """
    Validate that a file is a valid image.

    Returns:
        (is_valid, error_message)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return False, "File does not exist"

    # Check file size (corrupted/truncated images often < 1KB)
    if file_path.stat().st_size < 1024:
        return False, f"File too small: {file_path.stat().st_size} bytes"

    # Try to open with PIL
    try:
        from PIL import Image
        img = Image.open(file_path)
        img.verify()
        return True, None
    except Exception as e:
        return False, f"PIL validation failed: {e}"


class DownloadManifest:
    """Track downloaded files and retries for resumable downloads."""
    def __init__(self, manifest_path: Path | str):
        self.manifest_path = Path(manifest_path)
        self.data = {}
        self._load()

    def _load(self):
        """Load manifest from disk if it exists."""
        if self.manifest_path.exists():
            import json
            self.data = json.loads(self.manifest_path.read_text())

    def mark_downloaded(self, url: str, filepath: Path, md5: Optional[str] = None):
        """Record a successful download."""
        self.data[url] = {
            "filepath": str(filepath),
            "md5": md5,
            "timestamp": time.time(),
            "status": "success",
        }
        self._save()

    def mark_failed(self, url: str, error: str):
        """Record a failed download."""
        self.data[url] = {
            "error": error,
            "timestamp": time.time(),
            "status": "failed",
        }
        self._save()

    def get(self, url: str) -> Optional[dict]:
        """Retrieve record for a URL."""
        return self.data.get(url)

    def _save(self):
        """Save manifest to disk."""
        import json
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(self.data, indent=2))

    def get_failed_urls(self) -> list[str]:
        """Get list of URLs that failed on last attempt."""
        return [url for url, record in self.data.items() if record.get("status") == "failed"]
