"""
OWGR PDF Parser
Extracts ranking data from OWGR PDF files and converts to structured format.
"""

import pdfplumber
import pandas as pd
import re
from pathlib import Path
from datetime import datetime
import argparse


def extract_metadata_from_filename(filename):
    """Extract year, week, and ranking type from filename."""
    # Pattern: YYYY_owgrWWfYYYY.pdf or YYYY_Week XX Ranking - Sunday, Date YYYY.pdf
    match = re.search(r'^(\d{4})_', filename)
    year_prefix = int(match.group(1)) if match else None
    
    # Extract week number
    week_match = re.search(r'(?:owgr|Week\s+)(\d+)', filename)
    week = int(week_match.group(1)) if week_match else None
    
    # Determine ranking type
    if 'Federation' in filename or 'f2020' in filename.lower():
        ranking_type = 'federation'
    else:
        ranking_type = 'world'
    
    # Extract date if possible
    date_match = re.search(r'(\w+ \d+,? \d{4})', filename)
    date_str = date_match.group(1) if date_match else None
    
    return {
        'year': year_prefix,
        'week': week,
        'ranking_type': ranking_type,
        'date_str': date_str
    }


def clean_table_data(table):
    """Clean extracted table data."""
    if not table or len(table) < 2:
        return None
    
    # Make column names unique by adding suffix
    headers = table[0]
    unique_headers = []
    header_counts = {}
    
    for header in headers:
        if header in header_counts:
            header_counts[header] += 1
            unique_headers.append(f"{header}_{header_counts[header]}")
        else:
            header_counts[header] = 0
            unique_headers.append(header)
    
    # Convert to DataFrame with unique column names
    df = pd.DataFrame(table[1:], columns=unique_headers)
    
    # Remove completely empty rows
    df = df.dropna(how='all')
    
    # Remove rows where all values are empty strings
    df = df[~df.apply(lambda row: all(str(val).strip() == '' for val in row), axis=1)]
    
    return df


def parse_owgr_pdf(pdf_path):
    """
    Parse a single OWGR PDF file.
    
    Returns:
        DataFrame with columns: rank, player_name, country, points, events_played, etc.
    """
    pdf_path = Path(pdf_path)
    metadata = extract_metadata_from_filename(pdf_path.name)
    
    print(f"📄 Parsing: {pdf_path.name}")
    print(f"   Year: {metadata['year']}, Week: {metadata['week']}, Type: {metadata['ranking_type']}")
    
    all_data = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"   Pages: {len(pdf.pages)}")
            
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract tables from page
                tables = page.extract_tables()
                
                if not tables:
                    # Try extracting text as fallback
                    text = page.extract_text()
                    if text:
                        print(f"   Page {page_num}: No tables found, text extraction available")
                    continue
                
                for table_num, table in enumerate(tables, 1):
                    if not table or len(table) < 2:
                        continue
                    
                    # Clean and process table
                    df = clean_table_data(table)
                    if df is not None and not df.empty:
                        # Add metadata
                        df['source_year'] = metadata['year']
                        df['source_week'] = metadata['week']
                        df['ranking_type'] = metadata['ranking_type']
                        df['source_file'] = pdf_path.name
                        df['page_number'] = page_num
                        
                        all_data.append(df)
                        print(f"   Page {page_num}, Table {table_num}: {len(df)} rows extracted")
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None
    
    if not all_data:
        print(f"   ⚠️  No data extracted")
        return None
    
    # Combine all tables
    result = pd.concat(all_data, ignore_index=True)
    print(f"   ✓ Total rows: {len(result)}")
    
    return result


def test_sample_pdfs(pdf_dir, sample_size=5):
    """Test parser on sample PDFs from each year."""
    pdf_dir = Path(pdf_dir)
    all_pdfs = sorted(pdf_dir.glob('*.pdf'))
    
    print(f"\n📊 Testing parser on {len(all_pdfs)} PDFs")
    print("=" * 70)
    
    # Sample PDFs from different years
    sample_pdfs = []
    years = {}
    
    for pdf in all_pdfs:
        year = pdf.name[:4]
        if year not in years:
            years[year] = []
        years[year].append(pdf)
    
    # Take one from each year
    for year in sorted(years.keys()):
        if years[year]:
            sample_pdfs.append(years[year][0])  # First PDF of each year
    
    print(f"\nTesting {len(sample_pdfs)} sample PDFs (one per year):\n")
    
    results = []
    for pdf_path in sample_pdfs[:sample_size]:
        df = parse_owgr_pdf(pdf_path)
        if df is not None:
            results.append(df)
            print(f"\n   Sample columns: {list(df.columns[:8])}")
            print(f"   Sample row:\n{df.iloc[0] if len(df) > 0 else 'Empty'}\n")
        print("-" * 70)
    
    return results


def parse_all_pdfs(pdf_dir, output_file='data_files/owgr_rankings.parquet'):
    """Parse all OWGR PDFs and save to parquet."""
    pdf_dir = Path(pdf_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    all_pdfs = sorted(pdf_dir.glob('*.pdf'))
    print(f"\n📊 Parsing {len(all_pdfs)} PDFs from {pdf_dir}")
    print("=" * 70)
    
    all_data = []
    errors = []
    
    for i, pdf_path in enumerate(all_pdfs, 1):
        print(f"\n[{i}/{len(all_pdfs)}] ", end='')
        
        try:
            df = parse_owgr_pdf(pdf_path)
            if df is not None and not df.empty:
                all_data.append(df)
        except Exception as e:
            errors.append((pdf_path.name, str(e)))
            print(f"   ❌ Failed: {e}")
    
    if not all_data:
        print("\n❌ No data extracted from any PDFs")
        return None
    
    # Combine all data
    print(f"\n\n📦 Combining data from {len(all_data)} successful parses...")
    combined = pd.concat(all_data, ignore_index=True)
    
    # Save to parquet
    print(f"💾 Saving to {output_file}...")
    combined.to_parquet(output_file, index=False, compression='snappy')
    
    file_size = output_file.stat().st_size / 1024 / 1024
    print(f"✓ Saved {len(combined):,} rows to {output_file} ({file_size:.1f} MB)")
    
    if errors:
        print(f"\n⚠️  {len(errors)} files had errors:")
        for fname, error in errors[:10]:
            print(f"   - {fname}: {error}")
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("📊 Summary Statistics:")
    print(f"   Total rows: {len(combined):,}")
    print(f"   Columns: {list(combined.columns)}")
    print(f"   Years covered: {sorted(combined['source_year'].unique())}")
    print(f"   Ranking types: {combined['ranking_type'].value_counts().to_dict()}")
    
    return combined


def main():
    parser = argparse.ArgumentParser(description='Parse OWGR PDF files')
    parser.add_argument('--test', action='store_true', 
                       help='Test mode: parse sample PDFs only')
    parser.add_argument('--pdf-dir', default='data_files/owgr_pdfs',
                       help='Directory containing PDF files')
    parser.add_argument('--output', default='data_files/owgr_rankings.parquet',
                       help='Output parquet file path')
    parser.add_argument('--sample-size', type=int, default=5,
                       help='Number of samples to test')
    
    args = parser.parse_args()
    
    if args.test:
        print("🧪 TEST MODE: Parsing sample PDFs\n")
        test_sample_pdfs(args.pdf_dir, args.sample_size)
    else:
        print("🚀 FULL MODE: Parsing all PDFs\n")
        parse_all_pdfs(args.pdf_dir, args.output)


if __name__ == '__main__':
    main()
