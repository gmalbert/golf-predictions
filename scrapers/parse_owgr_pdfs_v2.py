"""
OWGR PDF Parser - Text-based extraction
Extracts ranking data from OWGR PDF files using text parsing.
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
    if 'Federation' in filename or 'Federation' in filename:
        ranking_type = 'federation'
    else:
        ranking_type = 'world'
    
    # Extract date if possible
    date_match = re.search(r'Ending?\s+(\d+\s+\w+\s+\d{4})', filename.replace('_', ' '), re.IGNORECASE)
    if not date_match:
        date_match = re.search(r'(\w+\s+\d+,?\s+\d{4})', filename)
    date_str = date_match.group(1).strip() if date_match else None
    
    return {
        'year': year_prefix,
        'week': week,
        'ranking_type': ranking_type,
        'date_str': date_str
    }


def parse_ranking_text(text):
    """Parse OWGR ranking data from text."""
    lines = text.split('\n')
    
    # Find header line (contains "This Last")
    header_idx = None
    for i, line in enumerate(lines):
        if 'This' in line and 'Last' in line and 'Name' in line:
            header_idx = i
            break
    
    if header_idx is None:
        return None
    
    # Parse header to understand column positions
    # Typical format:
    # This Last End Name Country Average Total Played Lost in Won/Gained in Played
    # Week Week 2023 Points Points (Divisor) 2024 2024 (Actual)
    
    ranks_data = []
    
    # Start after "Min-40 Max-52" line
    data_start = header_idx + 2
    
    for line in lines[data_start:]:
        line = line.strip()
        if not line:
            continue
        
        # Stop at page footer or other non-data lines
        if any(keyword in line.lower() for keyword in ['official', 'page', 'note:', 'denotes']):
            break
        
        # Parse ranking line
        # Format: 1 (1) <1> Dustin Johnson United States 11.1941 514.928 46 -9.477 56.000 46
        
        # Try to parse the first part which contains rank and name
        # Use simpler pattern to match any content in parentheses and angle brackets
        rank_name_match = re.match(
            r'^(\d+)\s+\(([^)]+)\)\s+<([^>]+)>\s+(.+)',
            line
        )
        
        if not rank_name_match:
            continue
        
        this_week = int(rank_name_match.group(1))
        last_week = rank_name_match.group(2).replace('T', '').replace('-', '0')
        end_year = rank_name_match.group(3).replace('-', '0')
        remainder = rank_name_match.group(4).strip()
        
        # The remainder contains: Name Country  Avg Total Div Lost Gained Actual
        # Extract the numeric values from the end (last 6 numbers)
        numbers = re.findall(r'-?[\d.]+', remainder)
        
        if len(numbers) < 6:
            continue
        
        # Last 6 numbers are: avg, total, divisor, lost, gained, actual
        avg_points = float(numbers[-6])
        total_points = float(numbers[-5])
        divisor = int(float(numbers[-4]))
        points_lost = float(numbers[-3])
        points_gained = float(numbers[-2])
        actual = int(float(numbers[-1]))
        
        # The rest before the numbers is the player name and country
        # Remove all the numbers we extracted
        name_country = remainder
        for num in numbers:
            name_country = name_country.replace(str(num), '', 1)
        name_country = name_country.strip()
        
        # Country is typically the last 1-3 words
        # Common patterns: "United States", "England", "Northern Ireland", "Korea; Republic of"
        words = name_country.split()
        if len(words) >= 2:
            # Heuristic: if last word looks like a country component, take it
            if len(words) >= 3 and words[-2] in ['United', 'Northern', 'South', 'New', 'Saudi']:
                country = ' '.join(words[-2:])
                player_name = ' '.join(words[:-2])
            elif len(words) >= 3 and ';' in words[-1]:  # "Korea; Republic of"
                country = ' '.join(words[-3:])
                player_name = ' '.join(words[:-3])
            else:
                country = words[-1]
                player_name = ' '.join(words[:-1])
        else:
            player_name = name_country
            country = ''
        
        ranks_data.append({
                'rank_this_week': this_week,
                'rank_last_week': last_week,
                'rank_end_prev_year': end_year,
                'player_name': player_name.strip(),
                'country': country.strip(),
                'avg_points': avg_points,
                'total_points': total_points,
                'events_played_divisor': divisor,
                'points_lost': points_lost,
                'points_gained': points_gained,
                'events_played_actual': actual
            })
    
    if not ranks_data:
        return None
    
    return pd.DataFrame(ranks_data)


def parse_owgr_pdf(pdf_path):
    """Parse a single OWGR PDF file."""
    pdf_path = Path(pdf_path)
    metadata = extract_metadata_from_filename(pdf_path.name)
    
    print(f"📄 {pdf_path.name}")
    
    all_data = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text from page
                text = page.extract_text()
                
                if not text:
                    continue
                
                # Parse ranking data from text
                df = parse_ranking_text(text)
                
                if df is not None and not df.empty:
                    # Add metadata
                    df['source_year'] = metadata['year']
                    df['source_week'] = metadata['week']
                    df['ranking_type'] = metadata['ranking_type']
                    df['date_str'] = metadata['date_str']
                    df['source_file'] = pdf_path.name
                    df['page_number'] = page_num
                    
                    all_data.append(df)
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None
    
    if not all_data:
        return None
    
    # Combine all pages
    result = pd.concat(all_data, ignore_index=True)
    print(f"   ✓ {len(result)} players")
    
    return result


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
        if i % 50 == 0:
            print(f"\n[{i}/{len(all_pdfs)}]")
        
        try:
            df = parse_owgr_pdf(pdf_path)
            if df is not None and not df.empty:
                all_data.append(df)
        except Exception as e:
            errors.append((pdf_path.name, str(e)))
            print(f"   ❌ {pdf_path.name}: {e}")
    
    if not all_data:
        print("\n❌ No data extracted from any PDFs")
        return None
    
    # Combine all data
    print(f"\n\n📦 Combining data from {len(all_data)} successful parses...")
    combined = pd.concat(all_data, ignore_index=True)
    
    # Convert rank columns to integers
    for col in ['rank_last_week', 'rank_end_prev_year']:
        combined[col] = pd.to_numeric(combined[col], errors='coerce').fillna(0).astype(int)
    
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
    print(f"   Unique players: {combined['player_name'].nunique():,}")
    print(f"   Years covered: {sorted(combined['source_year'].dropna().unique())}")
    print(f"   Weeks covered: {combined['source_week'].min()}-{combined['source_week'].max()}")
    print(f"   Ranking types: {combined['ranking_type'].value_counts().to_dict()}")
    print(f"   Date range: {combined['date_str'].min()} to {combined['date_str'].max()}")
    
    # Sample of data
    print("\n📋 Sample data (top 10 from most recent week):")
    latest = combined[combined['source_year'] == combined['source_year'].max()]
    latest = latest[latest['source_week'] == latest['source_week'].max()]
    latest = latest.sort_values('rank_this_week').head(10)
    print(latest[['rank_this_week', 'player_name', 'country', 'avg_points', 'total_points']].to_string(index=False))
    
    return combined


def main():
    parser = argparse.ArgumentParser(description='Parse OWGR PDF files')
    parser.add_argument('--pdf-dir', default='data_files/owgr_pdfs',
                       help='Directory containing PDF files')
    parser.add_argument('--output', default='data_files/owgr_rankings.parquet',
                       help='Output parquet file path')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: parse first 10 PDFs only')
    
    args = parser.parse_args()
    
    if args.test:
        print("🧪 TEST MODE: Parsing first 10 PDFs\n")
        pdf_dir = Path(args.pdf_dir)
        test_pdfs = sorted(pdf_dir.glob('*.pdf'))[:10]
        print(f"Found {len(test_pdfs)} PDFs in test mode")
        test_data = []
        for pdf_path in test_pdfs:
            print(f"\nProcessing: {pdf_path.name}")
            df = parse_owgr_pdf(pdf_path)
            if df is not None:
                print(f"  Got {len(df)} rows")
                test_data.append(df)
            else:
                print("  No data extracted")
        if test_data:
            combined = pd.concat(test_data, ignore_index=True)
            print(f"\n✓ Successfully parsed {len(combined):,} rows from {len(test_data)} PDFs")
            print(f"\nSample:\n{combined.head(10)}")
        else:
            print("\n❌ No data extracted from any PDFs")
    else:
        print("🚀 FULL MODE: Parsing all PDFs\n")
        parse_all_pdfs(args.pdf_dir, args.output)


if __name__ == '__main__':
    main()
