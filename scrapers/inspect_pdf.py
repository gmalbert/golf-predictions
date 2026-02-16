"""
Quick script to inspect PDF structure and text content
"""
import pdfplumber
from pathlib import Path
import sys

pdf_path = sys.argv[1] if len(sys.argv) > 1 else 'data_files/owgr_pdfs/2024_owgr01f2024.pdf'
pdf_path = Path(pdf_path)

print(f"📄 Inspecting: {pdf_path.name}")
print("=" * 70)

with pdfplumber.open(pdf_path) as pdf:
    print(f"Pages: {len(pdf.pages)}\n")
    
    for page_num, page in enumerate(pdf.pages[:2], 1):  # First 2 pages only
        print(f"\n📄 PAGE {page_num}")
        print("-" * 70)
        
        # Try to extract tables
        tables = page.extract_tables()
        print(f"Tables found: {len(tables)}")
        
        if tables:
            for t_num, table in enumerate(tables, 1):
                print(f"\n  Table {t_num}: {len(table)} rows")
                if table:
                    print(f"  Headers: {table[0]}")
                    if len(table) > 1:
                        print(f"  Sample row 1: {table[1]}")
                    if len(table) > 2:
                        print(f"  Sample row 2: {table[2]}")
        
        # Extract text
        text = page.extract_text()
        if text:
            lines = text.split('\n')[:20]  # First 20 lines
            print(f"\nText extraction (first 20 lines):")
            for i, line in enumerate(lines, 1):
                print(f"  {i:2d}: {line[:100]}")
