"""Debug script to inspect OWGR JSON structure."""

import json
from pathlib import Path
from bs4 import BeautifulSoup

# Find most recent cache file
cache_dir = Path("../data_files/cache")
cache_files = sorted(cache_dir.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True)

if not cache_files:
    print("No cache files found!")
    exit(1)

cache_file = cache_files[0]
print(f"Using: {cache_file.name}\n")

html = cache_file.read_text(encoding="utf-8")
soup = BeautifulSoup(html, "html.parser")

# Find __NEXT_DATA__
next_data = soup.find("script", {"id": "__NEXT_DATA__"})
if next_data:
    data = json.loads(next_data.string)
    
    def print_structure(obj, indent=0, max_depth=5, path=""):
        """Print JSON structure recursively."""
        prefix = "  " * indent
        
        if indent > max_depth:
            print(f"{prefix}... (max depth)")
            return
        
        if isinstance(obj, dict):
            print(f"{prefix}{{")
            for key, value in list(obj.items())[:10]:  # Limit to first 10 keys
                print(f"{prefix}  '{key}': ", end="")
                if isinstance(value, (dict, list)):
                    print()
                    print_structure(value, indent + 2, max_depth, f"{path}.{key}")
                else:
                    print(f"{type(value).__name__} = {str(value)[:50]}")
            if len(obj) > 10:
                print(f"{prefix}  ... ({len(obj) - 10} more keys)")
            print(f"{prefix}}}")
        
        elif isinstance(obj, list):
            print(f"{prefix}[ {len(obj)} items")
            if obj:
                print(f"{prefix}  [0]: ", end="")
                if isinstance(obj[0], (dict, list)):
                    print()
                    print_structure(obj[0], indent + 2, max_depth, f"{path}[0]")
                else:
                    print(f"{type(obj[0]).__name__} = {str(obj[0])[:50]}")
            print(f"{prefix}]")
        else:
            print(f"{prefix}{type(obj).__name__} = {str(obj)[:100]}")
    
    print("="*60)
    print("NEXT_DATA JSON Structure")
    print("="*60)
    print_structure(data)
    
    # Look for pageProps specifically
    if "props" in data and "pageProps" in data["props"]:
        print("\n" + "="*60)
        print("props.pageProps Structure")
        print("="*60)
        print_structure(data["props"]["pageProps"], max_depth=7)
