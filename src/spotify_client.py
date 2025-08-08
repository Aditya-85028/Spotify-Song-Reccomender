"""
Spotify API Client Module

Handles authentication and data fetching from Spotify Web API.
"""

import os
import sys
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from config import (
        SPOTIFY_CLIENT_ID, 
        SPOTIFY_CLIENT_SECRET, 
        SPOTIFY_REDIRECT_URI, 
        SPOTIFY_USERNAME,
        MAX_PLAYLISTS,
        MAX_TRACKS_PER_PLAYLIST
    )
except ImportError:
    logger.error("config.py not found. Please copy config_template.py to config.py and fill in your credentials.")
    sys.exit(1)


class SpotifyClient:
    """
    Client for interacting with Spotify Web API.
    
    Handles authentication, playlist fetching, and audio feature extraction.
    """
    
    def __init__(self):
        """Initialize Spotify client with authentication."""
        self.client_id = SPOTIFY_CLIENT_ID
        self.client_secret = SPOTIFY_CLIENT_SECRET
        self.redirect_uri = SPOTIFY_REDIRECT_URI
        self.username = SPOTIFY_USERNAME
        self.sp = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Spotify API."""
        try:
            scope = 'playlist-read-private user-library-read'
            token = util.prompt_for_user_token(
                username=self.username,
                scope=scope,
                client_id=self.client_id,
                client_secret=self.client_secret,
                redirect_uri=self.redirect_uri
            )
            
            if token:
                self.sp = spotipy.Spotify(auth=token)
                logger.info("Successfully authenticated with Spotify API")
            else:
                raise Exception(f"Could not get token for user: {self.username}")
                
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise
    
    def get_user_playlists(self) -> List[Dict]:
        """
        Fetch user's playlists.
        
        Returns:
            List of playlist dictionaries
        """
        try:
            results = self.sp.current_user_playlists(
                limit=MAX_PLAYLISTS, 
                offset=0
            )
            playlists = results['items']
            logger.info(f"Retrieved {len(playlists)} playlists")
            return playlists
        except Exception as e:
            logger.error(f"Failed to fetch user playlists: {str(e)}")
            raise
    
    def get_playlist_tracks(self, playlist_id: str) -> List[Dict]:
        """
        Fetch tracks from a specific playlist.
        
        Args:
            playlist_id: Spotify playlist ID
            
        Returns:
            List of track dictionaries
        """
        try:
            tracks = self.sp.playlist_items(
                playlist_id, 
                fields=None, 
                limit=MAX_TRACKS_PER_PLAYLIST, 
                offset=0, 
                market=None, 
                additional_types=('track', 'episode')
            )
            return tracks['items']
        except Exception as e:
            logger.error(f"Failed to fetch tracks from playlist {playlist_id}: {str(e)}")
            return []
    
    def get_audio_features(self, track_id: str) -> Optional[Dict]:
        """
        Get audio features for a specific track.
        
        Args:
            track_id: Spotify track ID
            
        Returns:
            Dictionary of audio features or None if failed
        """
        try:
            features = self.sp.audio_features(track_id)
            return features[0] if features and features[0] else None
        except Exception as e:
            logger.warning(f"Failed to get audio features for track {track_id}: {str(e)}")
            return None
    
    def get_spotify_playlists(self) -> List[str]:
        """
        Get Spotify's public playlists for building recommendation database.
        
        Returns:
            List of playlist IDs
        """
        try:
            spotify_playlists = self.sp.user_playlists('spotify')
            playlist_ids = [item['id'] for item in spotify_playlists['items']]
            logger.info(f"Retrieved {len(playlist_ids)} Spotify playlists")
            return playlist_ids
        except Exception as e:
            logger.error(f"Failed to fetch Spotify playlists: {str(e)}")
            return []
    
    def extract_track_data(self, track_item: Dict, liked: int = 1) -> Optional[List]:
        """
        Extract track data and audio features from a track item.
        
        Args:
            track_item: Track item from playlist
            liked: Whether the track is liked (1) or not (0)
            
        Returns:
            List of track data or None if extraction failed
        """
        try:
            track = track_item['track']
            if track is None:
                return None
            
            # Get basic track info
            track_id = track['id']
            track_name = track['name']
            artist_name = track['artists'][0]['name'] if track['artists'] else 'Unknown'
            
            # Get audio features
            features = self.get_audio_features(track_id)
            if features is None:
                return None
            
            # Extract feature values
            feature_list = [
                features['acousticness'],
                features['danceability'],
                features['energy'],
                features['instrumentalness'],
                features['liveness'],
                features['loudness'],
                features['speechiness'],
                features['tempo'],
                features['time_signature'],
                liked
            ]
            
            return [track_id, track_name, artist_name] + feature_list
            
        except Exception as e:
            logger.warning(f"Failed to extract track data: {str(e)}")
            return None
    
    def playlist_to_dataframe(self, playlist_ids: List[str], liked: int = 1) -> pd.DataFrame:
        """
        Convert playlists to a pandas DataFrame with audio features.
        
        Args:
            playlist_ids: List of playlist IDs
            liked: Whether tracks are liked (1) or not (0)
            
        Returns:
            DataFrame with track data and audio features
        """
        data = []
        columns = [
            'id', 'Track Name', 'Artist', 'Acousticness', 'Danceability',
            'Energy', 'Instrumentalness', 'Liveness', 'Loudness',
            'Speechiness', 'Tempo', 'Time Signature', 'Liked'
        ]
        
        for playlist_id in playlist_ids:
            logger.info(f"Processing playlist: {playlist_id}")
            tracks = self.get_playlist_tracks(playlist_id)
            
            for track_item in tracks:
                track_data = self.extract_track_data(track_item, liked)
                if track_data:
                    data.append(track_data)
        
        df = pd.DataFrame(data, columns=columns)
        logger.info(f"Created DataFrame with {len(df)} tracks")
        return df 