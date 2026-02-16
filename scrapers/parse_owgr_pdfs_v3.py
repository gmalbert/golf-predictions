"""
OWGR PDF Parser v3 - Improved country detection

Key improvements:
1. Uses comprehensive country list for backward matching
2. Handles amateur indicators like (Am), (May), etc.
3. Better handling of spacing issues in PDF text
"""

import pdfplumber
import pandas as pd
import re
from pathlib import Path
import argparse


# Comprehensive list of golf countries (ordered by length for longest-first matching)
GOLF_COUNTRIES = sorted([
    'United States', 'Northern Ireland', 'Chinese Taipei', 'South Africa',
    'Korea; Republic of', 'United Arab Emirates', 'Puerto Rico', 'Costa Rica',
    'Dominican Republic', 'Czech Republic', 'Saudi Arabia', 'New Zealand',
    'South Korea', 'England', 'Australia', 'Scotland', 'Japan', 'Sweden',
    'Thailand', 'India', 'China', 'France', 'Denmark', 'Spain', 'Germany',
    'Mexico', 'Italy', 'Canada', 'Finland', 'Ireland', 'Argentina',
    'Netherlands', 'Norway', 'Austria', 'Malaysia', 'Colombia', 'Switzerland',
    'Belgium', 'Wales', 'Philippines', 'Chile', 'Indonesia', 'Brazil',
    'Portugal', 'Singapore', 'Vietnam', 'Zimbabwe', 'Kenya', 'Nigeria',
    'Morocco', 'Tunisia', 'Turkey', 'Peru', 'Venezuela', 'Paraguay',
    'Uruguay', 'Ecuador', 'Jamaica', 'Russia', 'Ukraine', 'Poland',
    'Hungary', 'Greece', 'Bulgaria', 'Romania', 'Croatia', 'Slovenia',
    'Slovakia', 'Latvia', 'Lithuania', 'Estonia', 'Iceland', 'Cyprus',
    'Malta', 'Luxembourg', 'Liechtenstein', 'Monaco', 'Andorra',
    'Hong Kong', 'Taiwan', 'Macau', 'Fiji', 'Samoa', 'Tonga',
    'Papua New Guinea', 'Guam', 'Bangladesh', 'Pakistan', 'Sri Lanka',
    'Nepal', 'Myanmar', 'Cambodia', 'Laos', 'Brunei', 'Mongolia',
    'Kazakhstan', 'Uzbekistan', 'Kyrgyzstan', 'Tajikistan', 'Turkmenistan',
    'Israel', 'Egypt', 'Lebanon', 'Jordan', 'Syria', 'Iraq', 'Iran',
    'Afghanistan', 'Algeria', 'Libya', 'Sudan', 'Ethiopia', 'Ghana',
    'Ivory Coast', 'Senegal', 'Tanzania', 'Uganda', 'Zambia', 'Botswana',
    'Namibia', 'Mauritius', 'Reunion', 'Madagascar', 'Mozambique',
], key=len, reverse=True)  # Longest first for greedy matching


def extract_metadata_from_filename(filename):
    """Extract year, week, and ranking type from filename."""
    match = re.search(r'^(\d{4})_', filename)
    year_prefix = int(match.group(1)) if match else None
    
    week_match = re.search(r'(?:owgr|Week\s+)(\d+)', filename)
    week = int(week_match.group(1)) if week_match else None
    
    ranking_type = 'federation' if 'Federation' in filename else 'world'
    
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


def split_name_country(text_segment):
    """
    Split player name and country using country list.
    Works backwards from end to find country match.
    
    Args:
        text_segment: Text containing "Player Name Country"
        
    Returns:
        (player_name, country) tuple
    """
    text_segment = text_segment.strip()
    
    # Try each known country (longest first)
    for country in GOLF_COUNTRIES:
        # Check if text ends with this country (allowing for missing space)
        if text_segment.endswith(country):
            player_name = text_segment[:-len(country)].strip()
            return player_name, country
        
        # Check if country appears at end without space (PDF rendering issue)
        # e.g., "Tom KimKorea; Republic of"
        if text_segment.endswith(country.replace(' ', '')):
            player_name = text_segment[:-len(country.replace(' ', ''))].strip()
            return player_name, country
            
    # No match found - try heuristic approach
    # Assume last 1-3 words are country
    words = text_segment.split()
    if len(words) >= 2:
        # Common multi-word countries
        if len(words) >= 2 and words[-2] in ['United', 'Northern', 'South', 'New', 'Saudi', 'Chinese', 'Costa', 'Puerto', 'Czech', 'Hong']:
            return ' '.join(words[:-2]), ' '.join(words[-2:])
        else:
            return ' '.join(words[:-1]), words[-1]
    
    return text_segment, 'Unknown'


def parse_ranking_text(text):
    """Parse OWGR ranking data from text using improved country detection."""
    lines = text.split('\n')
    
    # Find header line
    header_idx = None
    for i, line in enumerate(lines):
        if 'This' in line and 'Last' in line and 'Week' in line:
            header_idx = i
            break
    
    if header_idx is None:
        return None
    
    ranks_data = []
    
    # Parse data lines (skip header and the Min-Max line)
    for line in lines[header_idx + 2:]:
        line = line.strip()
        if not line:
            continue
        
        # Stop at page footer
        if 'Week' in line and ('Page' in line or 'Footnotes' in line):
            break
        
        # Parse ranking line: "1 (1) <1> Player Name Country 9.8078 441.351 45 -9.878 9.961 45"
        rank_match = re.match(r'^(\d+)\s+\(([^)]+)\)\s+<([^>]+)>\s+(.+)', line)
        
        if not rank_match:
            continue
        
        this_week = int(rank_match.group(1))
        last_week = rank_match.group(2).replace('T', '').replace('-', '0')
        end_year = rank_match.group(3).replace('-', '0')
        remainder = rank_match.group(4).strip()
        
        # Extract last 6 numbers from remainder
        numbers = re.findall(r'-?[\d.]+', remainder)
        
        if len(numbers) < 6:
            continue
        
        # Last 6 numbers are the data columns
        avg_points = float(numbers[-6])
        total_points = float(numbers[-5])
        divisor = int(float(numbers[-4]))
        points_lost = float(numbers[-3])
        points_gained = float(numbers[-2])
        actual = int(float(numbers[-1]))
        
        # Get name + country by finding where first data number starts
        # The first data number is avg_points (typically in range 0.0001 to ~15.0)
        first_num = numbers[-6]
        first_num_pos = remainder.find(first_num)
        name_country = remainder[:first_num_pos].strip()
        
        # Split into player name and country
        player_name, country = split_name_country(name_country)
        
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
            'events_played_actual': actual,
        })
    
    if not ranks_data:
        return None
    
    return pd.DataFrame(ranks_data)


def parse_owgr_pdf(pdf_path):
    """Parse a single OWGR PDF file."""
    pdf_path = Path(pdf_path)
    print(f"📄 {pdf_path.name}")
    
    # Extract metadata from filename
    metadata = extract_metadata_from_filename(pdf_path.name)
    
    all_data = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                
                if not text:
                    continue
                
                # Parse ranking data
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
    success_count = 0
    
    for i, pdf_path in enumerate(all_pdfs, 1):
        df = parse_owgr_pdf(pdf_path)
        if df is not None:
            all_data.append(df)
            success_count += 1
        
        # Progress indicator
        if i % 50 == 0:
            print(f"\n[{i}/{len(all_pdfs)}]")
    
    if not all_data:
        print("\n❌ No data extracted!")
        return
    
    # Combine all data
    print(f"\n📦 Combining {len(all_data)} successful parses...")
    combined = pd.concat(all_data, ignore_index=True)
    
    # Save to parquet
    print(f"💾 Saving {len(combined):,} rows to {output_file}...")
    combined.to_parquet(output_file, index=False)
    
    print(f"\n✅ SUCCESS!")
    print(f"   Parsed: {success_count}/{len(all_pdfs)} PDFs")
    print(f"   Total rows: {len(combined):,}")
    print(f"   Unique players: {combined['player_name'].nunique():,}")
    print(f"   File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Parse OWGR PDF files (v3 - improved country detection)')
    parser.add_argument('--pdf-dir', default='data_files/owgr_pdfs',
                       help='Directory containing PDF files')
    parser.add_argument('--output', default='data_files/owgr_rankings_v3.parquet',
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
                print(f"\n  Sample data:")
                print(df[['rank_this_week', 'player_name', 'country', 'avg_points']].head(5))
                test_data.append(df)
            else:
                print("  No data extracted")
        if test_data:
            combined = pd.concat(test_data, ignore_index=True)
            print(f"\n✓ Successfully parsed {len(combined):,} rows from {len(test_data)} PDFs")
            print(f"\nCountry distribution:")
            print(combined['country'].value_counts().head(20))
        else:
            print("\n❌ No data extracted from any PDFs")
    else:
        print("🚀 FULL MODE: Parsing all PDFs\n")
        parse_all_pdfs(args.pdf_dir, args.output)


if __name__ == '__main__':
    main()
