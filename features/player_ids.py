"""
Player ID Mapping and Normalization

Creates stable, unique player IDs from names and maintains a player registry
that can match players across different data sources (ESPN, PGA Tour, etc.).
"""

import re
import hashlib
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple
from difflib import SequenceMatcher

DATA_DIR = Path("data_files")
PLAYER_REGISTRY = DATA_DIR / "player_registry.parquet"


def normalize_name(name: str) -> str:
    """
    Normalize a player name for consistent matching.
    
    Examples:
        "Scottie Scheffler" -> "scottie scheffler"
        "Davis Love III" -> "davis love"
        "Xander Schauffele" -> "xander schauffele"
        "Min Woo Lee" -> "min woo lee"
    
    Args:
        name: Raw player name
    
    Returns:
        Normalized name (lowercase, no suffixes, collapsed whitespace)
    """
    if not name or pd.isna(name):
        return ""
    
    name = str(name).strip()
    
    # Remove suffixes (Jr., Sr., III, IV, II, etc.)
    name = re.sub(r'\s*(Jr\.?|Sr\.?|III|IV|II|V)\s*$', '', name, flags=re.IGNORECASE)
    
    # Collapse multiple spaces
    name = re.sub(r'\s+', ' ', name)
    
    # Lowercase
    name = name.lower().strip()
    
    return name


def generate_player_id(name: str, country: Optional[str] = None) -> str:
    """
    Generate a stable, unique player ID from normalized name.
    
    Uses first 8 chars of MD5 hash of normalized name + country (if available).
    This ensures the same player always gets the same ID.
    
    Args:
        name: Player name (will be normalized)
        country: Optional country for disambiguation
    
    Returns:
        8-character hex player ID (e.g., "a3f4b2c1")
    
    Examples:
        >>> generate_player_id("Scottie Scheffler", "USA")
        'e8a9c4d2'
        >>> generate_player_id("Rory McIlroy", "NIR")
        '7f3b1a8e'
    """
    norm = normalize_name(name)
    
    if not norm:
        return "unknown"
    
    # Include country if available for disambiguation
    # (though unlikely to be needed for pro golfers)
    if country and not pd.isna(country):
        key = f"{norm}|{str(country).lower()}"
    else:
        key = norm
    
    # Generate hash
    hash_obj = hashlib.md5(key.encode('utf-8'))
    return hash_obj.hexdigest()[:8]


def name_similarity(name1: str, name2: str) -> float:
    """
    Calculate similarity between two names (0.0 to 1.0).
    
    Args:
        name1, name2: Names to compare (will be normalized)
    
    Returns:
        Similarity ratio (1.0 = identical, 0.0 = completely different)
    """
    norm1 = normalize_name(name1)
    norm2 = normalize_name(name2)
    
    if not norm1 or not norm2:
        return 0.0
    
    return SequenceMatcher(None, norm1, norm2).ratio()


class PlayerRegistry:
    """
    Maintains a registry of all known players with stable IDs.
    
    Automatically assigns IDs to new players and can fuzzy-match
    similar names to detect duplicates.
    """
    
    def __init__(self, registry_path: Path = PLAYER_REGISTRY):
        self.registry_path = registry_path
        self.players: pd.DataFrame = self._load_or_create()
    
    def _load_or_create(self) -> pd.DataFrame:
        """Load existing registry or create a new one."""
        if self.registry_path.exists():
            print(f"📂 Loading player registry from {self.registry_path.name}")
            df = pd.read_parquet(self.registry_path)
            print(f"   {len(df):,} players loaded")
            return df
        else:
            print("📝 Creating new player registry")
            # Create empty DataFrame with proper dtypes
            return pd.DataFrame({
                'player_id': pd.Series(dtype='str'),
                'name': pd.Series(dtype='str'),
                'normalized_name': pd.Series(dtype='str'),
                'country': pd.Series(dtype='str'),
                'first_seen': pd.Series(dtype='datetime64[ns]'),
                'source': pd.Series(dtype='str')
            })
    
    def save(self):
        """Save registry to disk."""
        self.players.to_parquet(self.registry_path, index=False)
        print(f"💾 Saved {len(self.players):,} players to {self.registry_path.name}")
    
    def add_player(
        self, 
        name: str, 
        country: Optional[str] = None,
        source: str = "unknown",
        check_duplicates: bool = True
    ) -> str:
        """
        Add a player to the registry (or return existing ID if already present).
        
        Args:
            name: Player display name
            country: Player country
            source: Data source (e.g., "espn", "pga_tour")
            check_duplicates: If True, warn about similar names
        
        Returns:
            player_id (8-char hex)
        """
        norm_name = normalize_name(name)
        
        if not norm_name:
            return "unknown"
        
        # Check if player already exists (by normalized name)
        existing = self.players[self.players['normalized_name'] == norm_name]
        
        if not existing.empty:
            # Player exists - return existing ID
            return existing.iloc[0]['player_id']
        
        # Check for similar names (potential duplicates)
        if check_duplicates and len(self.players) > 0:
            similarities = self.players['normalized_name'].apply(
                lambda x: name_similarity(norm_name, x)
            )
            max_sim = similarities.max()
            
            if max_sim > 0.9 and max_sim < 1.0:  # Very similar but not identical
                similar_idx = similarities.idxmax()
                similar_name = self.players.loc[similar_idx, 'name']
                print(f"⚠️  Similar name detected: '{name}' vs '{similar_name}' ({max_sim:.2%})")
        
        # Generate new ID
        player_id = generate_player_id(name, country)
        
        # Add to registry
        new_player = pd.DataFrame([{
            'player_id': player_id,
            'name': name,
            'normalized_name': norm_name,
            'country': country if country else None,
            'first_seen': pd.Timestamp.now(),
            'source': source,
        }])
        
        self.players = pd.concat([self.players, new_player], ignore_index=True)
        
        return player_id
    
    def get_id(self, name: str, fuzzy: bool = False) -> Optional[str]:
        """
        Look up a player ID by name.
        
        Args:
            name: Player name to look up
            fuzzy: If True, use fuzzy matching for close names
        
        Returns:
            player_id or None if not found
        """
        norm_name = normalize_name(name)
        
        # Exact match first
        match = self.players[self.players['normalized_name'] == norm_name]
        if not match.empty:
            return match.iloc[0]['player_id']
        
        # Fuzzy match
        if fuzzy and len(self.players) > 0:
            similarities = self.players['normalized_name'].apply(
                lambda x: name_similarity(norm_name, x)
            )
            max_sim = similarities.max()
            
            if max_sim > 0.85:  # 85% similarity threshold
                idx = similarities.idxmax()
                matched_name = self.players.loc[idx, 'name']
                print(f"🔍 Fuzzy match: '{name}' -> '{matched_name}' ({max_sim:.2%})")
                return self.players.loc[idx, 'player_id']
        
        return None
    
    def get_player_info(self, player_id: str) -> Optional[Dict]:
        """Get all info for a player by ID."""
        match = self.players[self.players['player_id'] == player_id]
        if match.empty:
            return None
        return match.iloc[0].to_dict()
    
    def stats(self):
        """Print registry statistics."""
        print(f"\n{'='*60}")
        print("Player Registry Stats")
        print(f"{'='*60}")
        print(f"Total players: {len(self.players):,}")
        
        if len(self.players) > 0:
            print(f"Countries: {self.players['country'].nunique()}")
            print(f"Sources: {self.players['source'].value_counts().to_dict()}")
            
            print(f"\nTop 5 countries:")
            top_countries = self.players['country'].value_counts().head(5)
            for country, count in top_countries.items():
                print(f"  {country}: {count}")
        
        print(f"{'='*60}\n")


def build_registry_from_scraped_data(
    data_files: list[Path],
    source_name: str = "espn"
) -> PlayerRegistry:
    """
    Build or update player registry from scraped parquet files.
    
    Args:
        data_files: List of parquet file paths
        source_name: Name of the data source
    
    Returns:
        Updated PlayerRegistry
    """
    registry = PlayerRegistry()
    
    print(f"\n{'='*60}")
    print(f"Building player registry from {len(data_files)} file(s)")
    print(f"{'='*60}\n")
    
    total_players = 0
    new_players = 0
    
    for file_path in data_files:
        print(f"Processing: {file_path.name}")
        df = pd.read_parquet(file_path)
        
        # Get unique players from this file
        players = df[['name', 'country']].drop_duplicates()
        
        initial_count = len(registry.players)
        
        for _, row in players.iterrows():
            player_id = registry.add_player(
                name=row['name'],
                country=row.get('country'),
                source=source_name,
                check_duplicates=False  # Skip warnings during bulk import
            )
            total_players += 1
        
        new_in_file = len(registry.players) - initial_count
        new_players += new_in_file
        print(f"  Added {new_in_file} new players ({players.shape[0]} total in file)")
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total player records processed: {total_players:,}")
    print(f"  New unique players added: {new_players:,}")
    print(f"  Registry size: {len(registry.players):,}")
    print(f"{'='*60}\n")
    
    registry.save()
    return registry


if __name__ == "__main__":
    # Demo / test
    print("Testing player ID utilities...\n")
    
    # Test normalization
    test_names = [
        "Scottie Scheffler",
        "Davis Love III",
        "Xander Schauffele",
        "Rory McIlroy",
        "Tiger Woods",
        "Jon Rahm",
    ]
    
    print("Name Normalization:")
    for name in test_names:
        norm = normalize_name(name)
        pid = generate_player_id(name, "USA")
        print(f"  {name:25s} -> {norm:20s} -> {pid}")
    
    print("\n" + "="*60)
    
    # Test registry
    print("\nTesting PlayerRegistry:")
    registry = PlayerRegistry()
    
    # Add players
    for name in test_names:
        pid = registry.add_player(name, country="USA", source="test")
        print(f"  Added: {name:25s} -> {pid}")
    
    # Test lookup
    print("\nLookup test:")
    test_id = registry.get_id("Scottie Scheffler")
    print(f"  ID for 'Scottie Scheffler': {test_id}")
    
    info = registry.get_player_info(test_id)
    print(f"  Player info: {info}")
    
    # Test fuzzy matching
    print("\nFuzzy match test:")
    fuzzy_id = registry.get_id("Scotty Scheffler", fuzzy=True)
    print(f"  'Scotty Scheffler' fuzzy matched to: {fuzzy_id}")
    
    registry.stats()
    
    print("✅ Tests complete!")
