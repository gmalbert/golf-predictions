# Compatibility shim for tests that import `shared_utils` directly.
# Prefer the real implementation under `scrapers/shared_utils.py`.
from scrapers.shared_utils import *

# Explicit exports (re-export common helpers used across the repo)
__all__ = [
    "polite_get",
    "get_random_headers",
    "get_cache_path",
    "CACHE_DIR",
    "DATA_DIR",
    "clear_cache",
]
