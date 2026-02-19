"""
Shared utilities for web scraping.
Includes caching, user agent rotation, and polite request handling.
"""

import time
import random
import requests
import hashlib
from pathlib import Path
from typing import Optional

# Cache directory for downloaded HTML
CACHE_DIR = Path("data_files/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("data_files")
DATA_DIR.mkdir(exist_ok=True)

# Rotate through different user agents to avoid detection
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]


def get_random_headers() -> dict:
    """Get headers with a random user agent."""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }


def get_cache_path(url: str) -> Path:
    """Generate a cache file path from a URL."""
    # Create a hash of the URL to use as filename
    url_hash = hashlib.md5(url.encode()).hexdigest()
    return CACHE_DIR / f"{url_hash}.html"


def polite_get(
    url: str, 
    use_cache: bool = True,
    force_refresh: bool = False,
    delay_range: tuple = (1.5, 3.5),
    **kwargs
) -> requests.Response:
    """
    GET with random delay, user agent rotation, and local caching.
    
    Args:
        url: URL to fetch
        use_cache: Whether to use/save local cache
        force_refresh: If True, ignore cache and re-download
        delay_range: Min/max delay in seconds between requests
        **kwargs: Additional arguments for requests.get()
    
    Returns:
        requests.Response object
    """
    cache_path = get_cache_path(url)
    
    # Check cache first
    if use_cache and not force_refresh and cache_path.exists():
        print(f"[cache] Using cached: {url}")
        # Create a mock response from cache
        response = requests.Response()
        response.status_code = 200
        response._content = cache_path.read_bytes()
        response.url = url
        return response
    
    # Add delay to be polite
    time.sleep(random.uniform(*delay_range))
    
    # Merge headers with any provided
    headers = get_random_headers()
    if "headers" in kwargs:
        headers.update(kwargs.pop("headers"))
    
    print(f"[fetch] Downloading: {url}")
    response = requests.get(url, headers=headers, timeout=30, **kwargs)
    response.raise_for_status()
    
    # Save to cache
    if use_cache:
        cache_path.write_bytes(response.content)
        print(f"[cache] Saved to: {cache_path.name}")
    
    return response


def clear_cache():
    """Clear all cached HTML files."""
    if CACHE_DIR.exists():
        count = 0
        for cache_file in CACHE_DIR.glob("*.html"):
            cache_file.unlink()
            count += 1
        print(f"[cache] Cleared {count} cached files")
    else:
        print("No cache to clear")


if __name__ == "__main__":
    # Test the caching system
    print("Testing cache system...")
    
    test_url = "https://www.pgatour.com/stats"
    
    print("\n1. First request (should download):")
    resp1 = polite_get(test_url)
    print(f"Status: {resp1.status_code}, Length: {len(resp1.content)} bytes")
    
    print("\n2. Second request (should use cache):")
    resp2 = polite_get(test_url)
    print(f"Status: {resp2.status_code}, Length: {len(resp2.content)} bytes")
    
    print("\n3. Force refresh:")
    resp3 = polite_get(test_url, force_refresh=True)
    print(f"Status: {resp3.status_code}, Length: {len(resp3.content)} bytes")
    
    print("\nDone!")
