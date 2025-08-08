"""
Configuration template for Spotify Song Recommender

Copy this file to config.py and fill in your actual credentials.
Never commit config.py to version control!
"""

# Spotify API Credentials
# Get these from https://developer.spotify.com/dashboard
SPOTIFY_CLIENT_ID = 'your_client_id_here'
SPOTIFY_CLIENT_SECRET = 'your_client_secret_here'
SPOTIFY_REDIRECT_URI = 'http://localhost:7777/callback'
SPOTIFY_USERNAME = 'your_spotify_username'

# Data settings
DATA_DIR = 'data/'
MAX_PLAYLISTS = 50
MAX_TRACKS_PER_PLAYLIST = 100

# Model settings
KNN_NEIGHBORS = 20
RECOMMENDATION_COUNT = 10

# Visualization settings
FIGURE_SIZE = (12, 8)
DPI = 300 