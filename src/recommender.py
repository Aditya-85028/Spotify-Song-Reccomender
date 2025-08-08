"""
Recommendation Engine Module

Implements K-Nearest Neighbors algorithm for song recommendations based on audio features.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

try:
    from config import KNN_NEIGHBORS, RECOMMENDATION_COUNT
except ImportError:
    KNN_NEIGHBORS = 20
    RECOMMENDATION_COUNT = 10
    logger.warning("Using default configuration values. Create config.py for custom settings.")


class SpotifyRecommender:
    """
    Recommendation engine using K-Nearest Neighbors algorithm.
    
    Analyzes audio features to find similar songs and generate recommendations.
    """
    
    def __init__(self, n_neighbors: int = KNN_NEIGHBORS):
        """
        Initialize the recommendation engine.
        
        Args:
            n_neighbors: Number of neighbors to consider for recommendations
        """
        self.n_neighbors = n_neighbors
        self.knn_model = None
        self.user_tracks = None
        self.unlistened_tracks = None
        self.song_data = None
        self.feature_columns = [
            'Acousticness', 'Danceability', 'Energy', 'Instrumentalness',
            'Liveness', 'Loudness', 'Speechiness', 'Tempo', 'Time Signature'
        ]
        logger.info(f"Initialized SpotifyRecommender with {n_neighbors} neighbors")
    
    def fit(self, user_tracks: pd.DataFrame, unlistened_tracks: pd.DataFrame, 
            song_data: pd.DataFrame) -> None:
        """
        Fit the recommendation model with training data.
        
        Args:
            user_tracks: DataFrame of user's liked tracks
            unlistened_tracks: DataFrame of tracks not yet listened to
            song_data: Feature-only DataFrame for unlistened tracks
        """
        logger.info("Fitting recommendation model...")
        
        self.user_tracks = user_tracks.copy()
        self.unlistened_tracks = unlistened_tracks.copy()
        self.song_data = song_data.copy()
        
        # Initialize and fit KNN model
        self.knn_model = NearestNeighbors(
            metric='cosine',
            algorithm='brute',
            n_neighbors=self.n_neighbors,
            n_jobs=-1
        )
        
        self.knn_model.fit(self.song_data)
        logger.info("Recommendation model fitted successfully")
    
    def get_recommendations(self, song_name: str, artist: str, 
                          num_recommendations: int = RECOMMENDATION_COUNT) -> pd.DataFrame:
        """
        Get song recommendations based on a specific song.
        
        Args:
            song_name: Name of the reference song
            artist: Artist of the reference song
            num_recommendations: Number of recommendations to return
            
        Returns:
            DataFrame with recommended songs
        """
        if self.knn_model is None:
            raise ValueError("Model not fitted. Call fit() method first.")
        
        logger.info(f"Getting recommendations for '{song_name}' by {artist}")
        
        # Find the reference song in user tracks
        reference_track = self.user_tracks[
            (self.user_tracks['Track Name'] == song_name) & 
            (self.user_tracks['Artist'] == artist)
        ]
        
        if reference_track.empty:
            raise ValueError(f"Song '{song_name}' by {artist} not found in user tracks")
        
        # Extract features and normalize
        metadata = reference_track[self.feature_columns].copy()
        metadata_normalized = tf.keras.utils.normalize(metadata, axis=1)
        
        # Get nearest neighbors
        distances, indices = self.knn_model.kneighbors(
            metadata_normalized, 
            n_neighbors=num_recommendations + 1
        )
        
        # Create recommendations DataFrame
        recommendations = []
        for idx, distance in zip(indices[0][1:], distances[0][1:]):  # Skip first (self)
            track = self.unlistened_tracks.iloc[idx]
            recommendations.append({
                'Title': track['Track Name'],
                'Artist': track['Artist'],
                'Similarity Score': 1 - distance,  # Convert distance to similarity
                'Acousticness': track['Acousticness'],
                'Danceability': track['Danceability'],
                'Energy': track['Energy'],
                'Tempo': track['Tempo']
            })
        
        recommendations_df = pd.DataFrame(recommendations)
        recommendations_df = recommendations_df.sort_values('Similarity Score', ascending=False)
        
        logger.info(f"Generated {len(recommendations_df)} recommendations")
        return recommendations_df
    
    def get_playlist_recommendations(self, song_dict: Dict[str, str], 
                                   num_recommendations: int = RECOMMENDATION_COUNT) -> pd.DataFrame:
        """
        Get recommendations for multiple songs (playlist-style).
        
        Args:
            song_dict: Dictionary mapping song names to artists
            num_recommendations: Number of recommendations per song
            
        Returns:
            DataFrame with all recommendations
        """
        logger.info(f"Getting playlist recommendations for {len(song_dict)} songs")
        
        all_recommendations = []
        
        for song_name, artist in song_dict.items():
            try:
                song_recs = self.get_recommendations(song_name, artist, num_recommendations)
                song_recs['Reference Song'] = f"{song_name} - {artist}"
                all_recommendations.append(song_recs)
            except ValueError as e:
                logger.warning(f"Skipping {song_name} by {artist}: {str(e)}")
                continue
        
        if not all_recommendations:
            raise ValueError("No valid recommendations could be generated")
        
        # Combine all recommendations
        combined_recs = pd.concat(all_recommendations, ignore_index=True)
        
        # Remove duplicates and sort by similarity
        combined_recs = combined_recs.drop_duplicates(subset=['Title', 'Artist'])
        combined_recs = combined_recs.sort_values('Similarity Score', ascending=False)
        
        logger.info(f"Generated {len(combined_recs)} unique playlist recommendations")
        return combined_recs
    
    def get_similar_songs(self, track_id: str, num_recommendations: int = RECOMMENDATION_COUNT) -> pd.DataFrame:
        """
        Get similar songs based on track ID.
        
        Args:
            track_id: Spotify track ID
            num_recommendations: Number of recommendations to return
            
        Returns:
            DataFrame with similar songs
        """
        if self.knn_model is None:
            raise ValueError("Model not fitted. Call fit() method first.")
        
        # Find track in unlistened tracks
        track_data = self.unlistened_tracks[self.unlistened_tracks['id'] == track_id]
        
        if track_data.empty:
            raise ValueError(f"Track with ID {track_id} not found in dataset")
        
        # Get features and find similar songs
        features = track_data[self.feature_columns].values
        distances, indices = self.knn_model.kneighbors(features, n_neighbors=num_recommendations + 1)
        
        recommendations = []
        for idx, distance in zip(indices[0][1:], distances[0][1:]):
            track = self.unlistened_tracks.iloc[idx]
            recommendations.append({
                'Title': track['Track Name'],
                'Artist': track['Artist'],
                'Similarity Score': 1 - distance,
                'Track ID': track['id']
            })
        
        return pd.DataFrame(recommendations)
    
    def get_feature_based_recommendations(self, features: Dict[str, float], 
                                        num_recommendations: int = RECOMMENDATION_COUNT) -> pd.DataFrame:
        """
        Get recommendations based on specific audio features.
        
        Args:
            features: Dictionary of audio features
            num_recommendations: Number of recommendations to return
            
        Returns:
            DataFrame with recommendations
        """
        if self.knn_model is None:
            raise ValueError("Model not fitted. Call fit() method first.")
        
        # Create feature vector
        feature_vector = []
        for feature in self.feature_columns:
            if feature in features:
                feature_vector.append(features[feature])
            else:
                feature_vector.append(0.5)  # Default value
        
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        # Normalize features
        feature_vector_normalized = tf.keras.utils.normalize(feature_vector, axis=1)
        
        # Get recommendations
        distances, indices = self.knn_model.kneighbors(
            feature_vector_normalized, 
            n_neighbors=num_recommendations
        )
        
        recommendations = []
        for idx, distance in zip(indices[0], distances[0]):
            track = self.unlistened_tracks.iloc[idx]
            recommendations.append({
                'Title': track['Track Name'],
                'Artist': track['Artist'],
                'Similarity Score': 1 - distance
            })
        
        return pd.DataFrame(recommendations)
    
    def get_user_profile_recommendations(self, num_recommendations: int = RECOMMENDATION_COUNT) -> pd.DataFrame:
        """
        Get recommendations based on user's overall listening profile.
        
        Args:
            num_recommendations: Number of recommendations to return
            
        Returns:
            DataFrame with recommendations
        """
        if self.knn_model is None or self.user_tracks is None:
            raise ValueError("Model not fitted. Call fit() method first.")
        
        # Calculate average user profile
        user_profile = self.user_tracks[self.feature_columns].mean().values.reshape(1, -1)
        
        # Normalize profile
        user_profile_normalized = tf.keras.utils.normalize(user_profile, axis=1)
        
        # Get recommendations
        distances, indices = self.knn_model.kneighbors(
            user_profile_normalized, 
            n_neighbors=num_recommendations
        )
        
        recommendations = []
        for idx, distance in zip(indices[0], distances[0]):
            track = self.unlistened_tracks.iloc[idx]
            recommendations.append({
                'Title': track['Track Name'],
                'Artist': track['Artist'],
                'Similarity Score': 1 - distance
            })
        
        return pd.DataFrame(recommendations)
    
    def evaluate_recommendations(self, test_songs: List[Tuple[str, str]], 
                               num_recommendations: int = 5) -> Dict[str, float]:
        """
        Evaluate recommendation quality using test songs.
        
        Args:
            test_songs: List of (song_name, artist) tuples to test
            num_recommendations: Number of recommendations per song
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating recommendation quality...")
        
        total_recommendations = 0
        successful_recommendations = 0
        
        for song_name, artist in test_songs:
            try:
                recommendations = self.get_recommendations(song_name, artist, num_recommendations)
                total_recommendations += len(recommendations)
                successful_recommendations += len(recommendations)
            except ValueError:
                logger.warning(f"Could not generate recommendations for {song_name} by {artist}")
        
        success_rate = successful_recommendations / total_recommendations if total_recommendations > 0 else 0
        
        evaluation_metrics = {
            'total_test_songs': len(test_songs),
            'successful_recommendations': successful_recommendations,
            'total_recommendations': total_recommendations,
            'success_rate': success_rate
        }
        
        logger.info(f"Evaluation completed: {success_rate:.2%} success rate")
        return evaluation_metrics 