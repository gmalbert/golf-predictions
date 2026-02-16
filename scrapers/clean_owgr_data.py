"""
Clean and normalize OWGR ranking data.

Fixes:
1. Country name parsing errors
2. Missing date_str values
3. Player name normalization
"""

import pandas as pd
import re
from pathlib import Path


# Known golf countries (most common)
GOLF_COUNTRIES = {
    # Full names
    'United States', 'England', 'Australia', 'South Africa', 'Japan', 'Sweden',
    'Thailand', 'India', 'China', 'France', 'South Korea', 'Denmark', 'Spain',
    'Scotland', 'Germany', 'Mexico', 'Italy', 'Canada', 'Finland', 'Ireland',
    'Argentina', 'Netherlands', 'New Zealand', 'Norway', 'Austria', 'Malaysia',
    'Colombia', 'Switzerland', 'Belgium', 'Wales', 'Philippines', 'Chile',
    'Indonesia', 'Brazil', 'Portugal', 'Singapore', 'Vietnam', 'Czech Republic',
    'South Africa', 'Zimbabwe', 'Kenya', 'Nigeria', 'Morocco', 'Tunisia',
    
    # Multi-word countries
    'Northern Ireland', 'Chinese Taipei', 'Hong Kong', 'New Zealand',
    'Czech Republic', 'South Korea', 'South Africa', 'Saudi Arabia',
    'United Arab Emirates', 'Puerto Rico', 'Costa Rica', 'Dominican Republic',
    
    # Special formats in OWGR
    'Korea; Republic of', 'Korea Republic of', 'Chinese Taipei)',
    'Taipei)', 'Taipei', 'Republic of Korea',
}

# Country normalization mapping
COUNTRY_FIXES = {
    'Korea; Republic': 'South Korea',
    'Korea; Republic of': 'South Korea',
    'Korea Republic of': 'South Korea',
    'Republic of Korea': 'South Korea',
    'Chinese Taipei)': 'Chinese Taipei',
    'Taipei)': 'Chinese Taipei',
    'Taipei': 'Chinese Taipei',
    'of': 'Unknown',  # Parsing artifact
    'States': 'United States',
    'TK': 'Unknown',  # Parsing artifact
    
    # Common 3-letter codes
    'USA': 'United States',
    'GBR': 'England',
    'AUS': 'Australia',
    'RSA': 'South Africa',
    'JPN': 'Japan',
    'SWE': 'Sweden',
    'THA': 'Thailand',
    'IND': 'India',
    'CHN': 'China',
    'FRA': 'France',
    'KOR': 'South Korea',
    'DEN': 'Denmark',
    'ESP': 'Spain',
    'SCO': 'Scotland',
    'GER': 'Germany',
    'MEX': 'Mexico',
    'ITA': 'Italy',
    'CAN': 'Canada',
    'FIN': 'Finland',
    'IRL': 'Ireland',
    'NED': 'Netherlands',
    'NZL': 'New Zealand',
    'NOR': 'Norway',
    'AUT': 'Austria',
    'MAS': 'Malaysia',
    'COL': 'Colombia',
    'SUI': 'Switzerland',
    'BEL': 'Belgium',
    'WAL': 'Wales',
    'PHI': 'Philippines',
    'CHI': 'Chile',
    'INA': 'Indonesia',
    'BRA': 'Brazil',
    'POR': 'Portugal',
    'SIN': 'Singapore',
    'VIE': 'Vietnam',
    'CZE': 'Czech Republic',
    'PUE': 'Puerto Rico',
}


def fix_country_name(country: str) -> str:
    """Normalize country name."""
    if pd.isna(country):
        return 'Unknown'
    
    country = country.strip()
    
    # Direct mapping
    if country in COUNTRY_FIXES:
        return COUNTRY_FIXES[country]
    
    # Remove parsing artifacts (player names stuck to country)
    # Pattern: name(Am)Country or nameCountry
    if '(' in country:
        # Likely has amateur indicator stuck to it
        # Extract just the country part (usually at the end)
        for known_country in GOLF_COUNTRIES:
            if country.endswith(known_country):
                return known_country
            if known_country in country:
                return known_country
        return 'Unknown'
    
    # Already valid
    if country in GOLF_COUNTRIES:
        return country
    
    # Check if it's a substring match (for truncated names)
    for known_country in GOLF_COUNTRIES:
        if known_country.startswith(country):
            return known_country
        if country.startswith(known_country):
            return known_country
    
    # Unknown
    return country  # Keep original for manual review


def extract_date_from_pdf_text(pdf_path: Path) -> str:
    """Extract date from PDF content (for newer format PDFs)."""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            if len(pdf.pages) > 0:
                text = pdf.pages[0].extract_text()
                lines = text.split('\n')
                
                # Look for "Ending DD Month YYYY" pattern
                for line in lines[:5]:
                    if 'Ending' in line:
                        # Extract: "Ending 07 January 2024"
                        match = re.search(r'Ending\s+(\d{1,2}\s+\w+\s+\d{4})', line)
                        if match:
                            return match.group(1)
    except Exception:
        pass
    
    return None


def clean_owgr_data(input_file='data_files/owgr_rankings.parquet',
                   output_file='data_files/owgr_rankings_clean.parquet'):
    """Clean and normalize OWGR data."""
    print("📊 Loading OWGR data...")
    df = pd.read_parquet(input_file)
    print(f"   Loaded {len(df):,} rows")
    
    # Fix country names
    print("\n🌍 Fixing country names...")
    df['country_original'] = df['country']
    df['country'] = df['country'].apply(fix_country_name)
    
    fixed_countries = (df['country'] != df['country_original']).sum()
    print(f"   Fixed {fixed_countries:,} country entries ({fixed_countries/len(df)*100:.2f}%)")
    
    # Check remaining unknowns
    unknown_countries = df[df['country'] == 'Unknown']
    print(f"   Remaining unknown: {len(unknown_countries):,} rows")
    
    # Show top unknown originals for manual review
    if len(unknown_countries) > 0:
        print("\n   Top unknown country values:")
        top_unknown = df[df['country'] == 'Unknown']['country_original'].value_counts().head(20)
        for country, count in top_unknown.items():
            print(f"      {country}: {count}")
    
    # Fix date_str for newer PDFs
    print("\n📅 Fixing missing dates...")
    null_dates = df['date_str'].isnull()
    print(f"   Found {null_dates.sum():,} null dates")
    
    # For newer format, we could extract from PDF content, but that's slow
    # For now, just mark them clearly
    df.loc[null_dates, 'date_str'] = ''
    
    # Show data quality summary
    print("\n📈 Data Quality Summary:")
    print(f"   Total rows: {len(df):,}")
    print(f"   Unique players: {df['player_name'].nunique():,}")
    print(f"   Unique countries: {df['country'].nunique()}")
    print(f"   Date coverage: {(~df['date_str'].isnull()).sum():,}/{len(df):,}")
    print(f"   Years: {sorted(df['source_year'].unique())}")
    
    # Save cleaned data
    print(f"\n💾 Saving cleaned data to {output_file}...")
    df.to_parquet(output_file, index=False)
    print("   ✓ Done!")
    
    return df


if __name__ == '__main__':
    clean_owgr_data()
