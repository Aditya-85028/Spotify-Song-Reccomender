"""
Spotify Song Recommender Package

A machine learning-based music recommendation system that analyzes
Spotify playlists and suggests new songs based on listening preferences.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .spotify_client import SpotifyClient
from .data_processor import DataProcessor
from .recommender import SpotifyRecommender
from .visualizer import DataVisualizer

__all__ = [
    'SpotifyClient',
    'DataProcessor', 
    'SpotifyRecommender',
    'DataVisualizer'
] 