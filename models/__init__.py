"""
Models package for golf tournament predictions.
"""

from pathlib import Path

MODELS_DIR = Path(__file__).parent
SAVED_MODELS_DIR = MODELS_DIR / 'saved_models'

__all__ = ['train_baseline_model', 'predict_tournament']
