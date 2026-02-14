# Feature Engineering

This module handles feature engineering and data preparation for the Fairway Oracle prediction models.

## Player ID System

The player ID system creates stable, unique identifiers for golfers across multiple data sources. This is essential for:
- Tracking player performance over time
- Building historical features (form, momentum, course history)
- Matching players across ESPN, PGA Tour, OWGR, and other sources

### Key Components

#### `player_ids.py`
Core utilities for player identification:

**Functions:**
- `normalize_name(name)` - Standardizes player names (lowercase, remove suffixes like Jr./III)
- `generate_player_id(name, country)` - Creates 8-char hex ID from MD5 hash
- `name_similarity(name1, name2)` - Calculates similarity ratio (0.0-1.0) for fuzzy matching

**Classes:**
- `PlayerRegistry` - Maintains registry of all known players
  - `add_player(name, country, source)` - Add/retrieve player ID
  - `get_id(name, fuzzy=False)` - Look up ID by name (with optional fuzzy match)
  - `get_player_info(player_id)` - Retrieve full player details
  - `save()` - Persist registry to parquet

#### `apply_player_ids.py`
Batch processing script to apply IDs to scraped data:

```bash
# Process all ESPN data files
python features/apply_player_ids.py

# Process a specific file
python features/apply_player_ids.py --file data_files/espn_pga_2023.parquet

# Validate existing IDs
python features/apply_player_ids.py --validate
```

**What it does:**
1. Scans all ESPN parquet files
2. Builds/updates player registry from unique (name, country) pairs
3. Applies player_id to each row
4. Saves updated data files

#### `check_player_ids.py`
Validation and statistics:

```bash
python features/check_player_ids.py
```

Shows:
- Player ID field completeness (null/unknown/valid counts)
- Top players by tournament appearances
- ID stability check (1 ID per player)
- Country distribution
- Sample records

### Player Registry File

**Location:** `data_files/player_registry.parquet`

**Schema:**
- `player_id` (str) - 8-char hex identifier (e.g., "5bbc1c05")
- `name` (str) - Display name (e.g., "Scottie Scheffler")
- `normalized_name` (str) - Search key (e.g., "scottie scheffler")
- `country` (str) - Country/region (e.g., "United States")
- `first_seen` (datetime) - When player was first added
- `source` (str) - Data source that added player (e.g., "espn")

### How Player IDs Work

1. **Name Normalization:**
   - "Davis Love III" → "davis love"
   - "Scottie Scheffler" → "scottie scheffler"
   - Removes Jr., Sr., II, III, IV suffixes
   - Lowercase, collapsed whitespace

2. **ID Generation:**
   - MD5 hash of `normalized_name|country`
   - First 8 hex chars used as ID
   - Same player always gets same ID (deterministic)

3. **Fuzzy Matching:**
   - Uses SequenceMatcher for similarity scoring
   - 85%+ similarity triggers fuzzy match
   - Example: "Scotty Scheffler" → "Scottie Scheffler" (90.91% match)

4. **Cross-Source Matching:**
   - Registry merges players from ESPN, PGA Tour, etc.
   - Normalized names ensure consistent matching
   - Country helps disambiguate common names

### Example Usage

```python
from features.player_ids import PlayerRegistry, normalize_name

# Load registry
registry = PlayerRegistry()

# Add a player
player_id = registry.add_player(
    name="Scottie Scheffler",
    country="United States",
    source="espn"
)
print(player_id)  # "50dd10c6"

# Look up by exact name
pid = registry.get_id("Scottie Scheffler")

# Fuzzy match
pid = registry.get_id("Scotty Scheffler", fuzzy=True)  # Matches to Scottie

# Get player info
info = registry.get_player_info(pid)
print(info['name'])  # "Scottie Scheffler"

# Normalize a name
normalized = normalize_name("Davis Love III")  # "davis love"
```

### Current Status

**ESPN 2022 Data:**
- ✅ 4,080 rows processed
- ✅ 80 unique players identified
- ✅ 100% of rows have valid player IDs (was 100% null before)
- ✅ Stable ID mapping (1:1 player-to-ID)

**Registry:**
- 80 players
- 21 countries
- 47 US players, 6 English, 4 Canadian, etc.

### Next Steps

1. **Multi-Year Expansion:** Apply to 2018-2024 data when scraped
2. **Cross-Source Matching:** Match ESPN players to PGA Tour, OWGR data
3. **Disambiguation:** Handle edge cases (same name, different players)
4. **Metadata Enrichment:** Add birth year, turned pro, handedness, etc.

### Testing

Run the test suite:

```bash
# Test core utilities
python features/player_ids.py

# Apply to data
python features/apply_player_ids.py

# Validate results
python features/apply_player_ids.py --validate

# Check stats
python features/check_player_ids.py
```

All 4 scripts include comprehensive validation and reporting.
