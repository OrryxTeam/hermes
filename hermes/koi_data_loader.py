"""
KOI Data Loader - loading using local CSV + NASA API.

This module provides intelligent data loading by:
1. Reading local KOI CSV with metadata
2. Downloading light curves on-demand from NASA using lightkurve
3. Caching downloaded data locally for reuse
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Iterator
import numpy as np
import pandas as pd
from absl import logging
import gin

# Try importing lightkurve for NASA downloads
try:
    import lightkurve as lk
    LK_AVAILABLE = True
except ImportError:
    logging.warning("lightkurve not installed. Install with: pip install lightkurve")
    LK_AVAILABLE = False


@gin.configurable
class KOIDataLoader:
    """Smart data loader using KOI CSV + NASA API."""
    
    def __init__(self,
                 koi_csv_path: str = "dataset/koi.csv",
                 cache_dir: str = "dataset/kepler_cache",
                 batch_size: int = 32,
                 feature_dim: int = 2048,
                 max_planets: int = 5,
                 disposition_filter: Optional[List[str]] = None,
                 download_on_demand: bool = True,
                 seed: int = 42):
        """
        Initialize KOI data loader.
        
        Args:
            koi_csv_path: Path to KOI CSV file
            cache_dir: Directory to cache downloaded light curves
            batch_size: Batch size for training
            feature_dim: Dimension of feature vector (light curve length)
            max_planets: Maximum number of planets to detect
            disposition_filter: Filter by disposition (e.g., ['CONFIRMED', 'CANDIDATE'])
            download_on_demand: If True, download light curves as needed
            seed: Random seed
        """
        self.koi_csv_path = Path(koi_csv_path)
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.max_planets = max_planets
        self.download_on_demand = download_on_demand
        self.rng = np.random.RandomState(seed)
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load KOI catalog
        self.koi_df = self._load_koi_catalog(disposition_filter)
        
        # Group by star to identify multi-planet systems
        self.star_groups = self.koi_df.groupby('kepoi_name').first()
        
        logging.info(f"Loaded {len(self.koi_df)} KOIs from {len(self.star_groups)} unique stars")
        
    def _load_koi_catalog(self, disposition_filter: Optional[List[str]] = None) -> pd.DataFrame:
        """Load and filter KOI catalog."""
        if not self.koi_csv_path.exists():
            raise FileNotFoundError(f"KOI CSV not found: {self.koi_csv_path}")
            
        df = pd.read_csv(self.koi_csv_path)
        
        # Apply disposition filter if provided
        if disposition_filter:
            df = df[df['koi_disposition'].isin(disposition_filter)]
            logging.info(f"Filtered to {len(df)} KOIs with disposition: {disposition_filter}")
        
        # Drop rows with missing critical data
        required_cols = ['kepoi_name', 'koi_period']
        df = df.dropna(subset=required_cols)
        
        # Extract KIC ID from KOI name (e.g., K00752.01 -> 00752)
        df['kic_id'] = df['kepoi_name'].str.extract(r'K(\d+)\.').astype(int)
        
        return df
    
    def _get_cache_path(self, kic_id: int) -> Path:
        """Get cache file path for a given KIC ID."""
        return self.cache_dir / f"kic_{kic_id:09d}.pkl"
    
    def _download_light_curve(self, kic_id: int) -> Optional[Dict]:
        """
        Download light curve from NASA using lightkurve.
        
        Returns:
            Dict with 'time' and 'flux' arrays, or None if download fails
        """
        if not LK_AVAILABLE:
            logging.warning("lightkurve not available, returning synthetic data")
            return self._generate_synthetic_light_curve(kic_id)
        
        try:
            # Search for Kepler light curve
            search_result = lk.search_lightcurve(
                target=f"KIC {kic_id}",
                mission="Kepler",
                cadence="long"
            )
            
            if len(search_result) == 0:
                logging.warning(f"No light curve found for KIC {kic_id}")
                return None
            
            # Download all quarters
            lc_collection = search_result.download_all(quality_bitmask="default")
            
            if lc_collection is None or len(lc_collection) == 0:
                return None
            
            # Stitch quarters together
            lc = lc_collection.stitch()
            
            # Remove NaN values
            mask = ~np.isnan(lc.flux.value)
            time = lc.time.value[mask]
            flux = lc.flux.value[mask]
            
            # Normalize flux
            flux = flux / np.median(flux)
            
            return {
                'time': time,
                'flux': flux,
                'kic_id': kic_id
            }
            
        except Exception as e:
            logging.error(f"Failed to download KIC {kic_id}: {e}")
            return None
    
    def _generate_synthetic_light_curve(self, kic_id: int) -> Dict:
        """Generate synthetic light curve for testing without NASA access."""
        # Use KIC ID as seed for reproducibility
        rng = np.random.RandomState(kic_id)
        
        # Generate time series
        time = np.linspace(0, 90, 4000)  # 90 days of observations
        flux = np.ones_like(time) + rng.normal(0, 0.001, len(time))
        
        # Add some transits based on KOIs for this star
        star_kois = self.koi_df[self.koi_df['kic_id'] == kic_id]
        
        for _, koi in star_kois.iterrows():
            if pd.notna(koi['koi_period']):
                period = koi['koi_period']
                depth = 0.01 * (koi['koi_prad'] / 10.0 if pd.notna(koi['koi_prad']) else 1.0)
                
                # Add periodic transits
                for transit_time in np.arange(0, 90, period):
                    transit_mask = np.abs(time - transit_time) < 0.2
                    flux[transit_mask] *= (1 - depth)
        
        return {
            'time': time,
            'flux': flux,
            'kic_id': kic_id
        }
    
    def get_light_curve(self, kic_id: int) -> Optional[Dict]:
        """
        Get light curve, using cache if available.
        
        Args:
            kic_id: KIC identifier
            
        Returns:
            Dict with light curve data or None
        """
        cache_path = self._get_cache_path(kic_id)
        
        # Check cache first
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # Download if enabled
        if self.download_on_demand:
            lc_data = self._download_light_curve(kic_id)
            
            if lc_data is not None:
                # Save to cache
                with open(cache_path, 'wb') as f:
                    pickle.dump(lc_data, f)
                
                return lc_data
        
        return None
    
    def create_features(self, time: np.ndarray, flux: np.ndarray) -> np.ndarray:
        """
        Create fixed-size feature vector from variable-length light curve.
        
        Args:
            time: Time array
            flux: Flux array
            
        Returns:
            Feature vector of shape (feature_dim,)
        """
        # Simple approach: interpolate to fixed size
        if len(flux) == 0:
            return np.zeros(self.feature_dim)
        
        # Resample to fixed size
        if len(flux) != self.feature_dim:
            from scipy import interpolate
            f = interpolate.interp1d(
                np.linspace(0, 1, len(flux)),
                flux,
                kind='linear',
                fill_value='extrapolate'
            )
            features = f(np.linspace(0, 1, self.feature_dim))
        else:
            features = flux[:self.feature_dim]
        
        # Normalize
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        
        return features.astype(np.float32)
    
    def prepare_batch(self, kic_ids: List[int]) -> Dict:
        """
        Prepare a batch of data.
        
        Args:
            kic_ids: List of KIC IDs
            
        Returns:
            Batch dict with 'light_curve', 'planet_count', 'planet_exists'
        """
        batch_features = []
        batch_planet_counts = []
        batch_planet_exists = []
        
        for kic_id in kic_ids:
            # Get light curve
            lc_data = self.get_light_curve(kic_id)
            
            if lc_data is None:
                # Use zeros if no data available
                features = np.zeros(self.feature_dim, dtype=np.float32)
            else:
                features = self.create_features(lc_data['time'], lc_data['flux'])
            
            # Get planet information
            star_kois = self.koi_df[self.koi_df['kic_id'] == kic_id]
            confirmed_planets = star_kois[star_kois['koi_disposition'] == 'CONFIRMED']
            
            num_planets = min(len(confirmed_planets), self.max_planets)
            planet_exists = np.zeros(self.max_planets, dtype=np.float32)
            planet_exists[:num_planets] = 1.0
            
            batch_features.append(features)
            batch_planet_counts.append(num_planets)
            batch_planet_exists.append(planet_exists)
        
        return {
            'light_curve': np.stack(batch_features),
            'planet_count': np.array(batch_planet_counts, dtype=np.int32),
            'planet_exists': np.stack(batch_planet_exists)
        }
    
    def split_data(self, train_frac: float = 0.7, val_frac: float = 0.15):
        """
        Split KIC IDs into train/val/test sets.
        
        Args:
            train_frac: Fraction for training
            val_frac: Fraction for validation
            
        Returns:
            Dict with 'train', 'validation', 'test' KIC ID lists
        """
        # Get unique KIC IDs
        unique_kics = self.koi_df['kic_id'].unique()
        
        # Shuffle
        self.rng.shuffle(unique_kics)
        
        # Split
        n = len(unique_kics)
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))
        
        return {
            'train': unique_kics[:train_end],
            'validation': unique_kics[train_end:val_end],
            'test': unique_kics[val_end:]
        }
    
    def create_data_iterator(self, kic_ids: np.ndarray, shuffle: bool = True) -> Iterator[Dict]:
        """
        Create iterator for a set of KIC IDs.
        
        Args:
            kic_ids: Array of KIC IDs
            shuffle: Whether to shuffle
            
        Yields:
            Batch dictionaries
        """
        if shuffle:
            kic_ids = kic_ids.copy()
            self.rng.shuffle(kic_ids)
        
        for i in range(0, len(kic_ids), self.batch_size):
            batch_kics = kic_ids[i:i + self.batch_size]
            
            # Skip incomplete batches for training
            if len(batch_kics) < self.batch_size and shuffle:
                continue
                
            yield self.prepare_batch(batch_kics)


@gin.configurable
def load_koi_data(koi_csv_path: str = "dataset/koi.csv",
                  cache_dir: str = "dataset/kepler_cache",
                  batch_size: int = 32,
                  disposition_filter: Optional[List[str]] = None):
    """
    Load KOI data with smart caching.
    
    Args:
        koi_csv_path: Path to KOI CSV
        cache_dir: Cache directory
        batch_size: Batch size
        disposition_filter: Filter by disposition
        
    Returns:
        Dict with train/validation/test iterators
    """
    # Create loader
    loader = KOIDataLoader(
        koi_csv_path=koi_csv_path,
        cache_dir=cache_dir,
        batch_size=batch_size,
        disposition_filter=disposition_filter or ['CONFIRMED']
    )
    
    # Split data
    splits = loader.split_data()
    
    logging.info(f"Dataset splits - Train: {len(splits['train'])}, "
                f"Val: {len(splits['validation'])}, Test: {len(splits['test'])}")
    
    # Create iterators
    return {
        'train': lambda: loader.create_data_iterator(splits['train'], shuffle=True),
        'validation': lambda: loader.create_data_iterator(splits['validation'], shuffle=False),
        'test': lambda: loader.create_data_iterator(splits['test'], shuffle=False),
        'loader': loader  # Keep reference to loader for metadata access
    }
