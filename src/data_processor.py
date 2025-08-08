"""
Data Processing Module

Handles data cleaning, preprocessing, and feature scaling for the recommendation system.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import logging
from typing import Tuple, Optional
import os

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Handles data processing operations for the Spotify recommendation system.
    
    Includes data cleaning, feature scaling, and dataset preparation.
    """
    
    def __init__(self):
        """Initialize the data processor."""
        self.scaler = MinMaxScaler()
        self.feature_columns = [
            'Acousticness', 'Danceability', 'Energy', 'Instrumentalness',
            'Liveness', 'Loudness', 'Speechiness', 'Tempo', 'Time Signature'
        ]
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by removing duplicates and handling missing values.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning process...")
        
        # Remove duplicates based on track name and artist
        initial_count = len(df)
        df = df.drop_duplicates(subset=['Track Name', 'Artist'], keep='first')
        df = df.reset_index(drop=True)
        
        removed_count = initial_count - len(df)
        logger.info(f"Removed {removed_count} duplicate tracks")
        
        # Handle missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(f"Found missing values:\n{missing_counts[missing_counts > 0]}")
            
            # Fill missing values with median for numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if df[col].isnull().sum() > 0:
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    logger.info(f"Filled missing values in {col} with median: {median_val}")
        
        logger.info(f"Data cleaning completed. Final dataset has {len(df)} tracks")
        return df
    
    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale audio features using Min-Max scaling.
        
        Args:
            df: DataFrame with audio features
            
        Returns:
            DataFrame with scaled features
        """
        logger.info("Scaling audio features...")
        
        # Create a copy to avoid modifying original data
        df_scaled = df.copy()
        
        # Scale only the feature columns
        feature_data = df_scaled[self.feature_columns]
        scaled_features = self.scaler.fit_transform(feature_data)
        
        # Update the DataFrame with scaled features
        df_scaled[self.feature_columns] = scaled_features
        
        logger.info("Feature scaling completed")
        return df_scaled
    
    def prepare_datasets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare datasets for training and recommendation.
        
        Args:
            df: Complete dataset with scaled features
            
        Returns:
            Tuple of (user_tracks, unlistened_tracks, song_data)
        """
        logger.info("Preparing datasets for recommendation system...")
        
        # Split into liked and unliked tracks
        user_tracks = df.loc[df['Liked'] == 1].copy()
        unlistened = df.loc[df['Liked'] == 0].copy()
        
        # Create feature-only dataset for unlistened tracks
        song_data = unlistened[self.feature_columns].copy()
        
        logger.info(f"Dataset preparation completed:")
        logger.info(f"  - User tracks (liked): {len(user_tracks)}")
        logger.info(f"  - Unlistened tracks: {len(unlistened)}")
        logger.info(f"  - Feature dataset: {song_data.shape}")
        
        return user_tracks, unlistened, song_data
    
    def get_feature_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get statistical summary of audio features.
        
        Args:
            df: DataFrame with audio features
            
        Returns:
            Statistical summary DataFrame
        """
        return df[self.feature_columns].describe()
    
    def save_data(self, df: pd.DataFrame, filename: str, data_dir: str = 'data/') -> None:
        """
        Save DataFrame to CSV file.
        
        Args:
            df: DataFrame to save
            filename: Name of the file
            data_dir: Directory to save the file
        """
        os.makedirs(data_dir, exist_ok=True)
        filepath = os.path.join(data_dir, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
    
    def load_data(self, filename: str, data_dir: str = 'data/') -> Optional[pd.DataFrame]:
        """
        Load DataFrame from CSV file.
        
        Args:
            filename: Name of the file to load
            data_dir: Directory containing the file
            
        Returns:
            Loaded DataFrame or None if file doesn't exist
        """
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            logger.info(f"Data loaded from {filepath}")
            return df
        else:
            logger.warning(f"File not found: {filepath}")
            return None
    
    def combine_datasets(self, user_tracks: pd.DataFrame, spotify_tracks: pd.DataFrame) -> pd.DataFrame:
        """
        Combine user tracks and Spotify tracks into a single dataset.
        
        Args:
            user_tracks: DataFrame of user's liked tracks
            spotify_tracks: DataFrame of Spotify tracks
            
        Returns:
            Combined DataFrame
        """
        logger.info("Combining user and Spotify datasets...")
        
        # Combine datasets
        combined_df = pd.concat([user_tracks, spotify_tracks], ignore_index=True)
        
        # Clean the combined dataset
        combined_df = self.clean_data(combined_df)
        
        logger.info(f"Combined dataset created with {len(combined_df)} tracks")
        return combined_df
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate the dataset for required columns and data types.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_columns = [
            'id', 'Track Name', 'Artist', 'Acousticness', 'Danceability',
            'Energy', 'Instrumentalness', 'Liveness', 'Loudness',
            'Speechiness', 'Tempo', 'Time Signature', 'Liked'
        ]
        
        # Check if all required columns exist
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check data types
        expected_types = {
            'id': 'object',
            'Track Name': 'object',
            'Artist': 'object',
            'Liked': 'int64'
        }
        
        for col, expected_type in expected_types.items():
            if df[col].dtype != expected_type:
                logger.warning(f"Column {col} has type {df[col].dtype}, expected {expected_type}")
        
        # Check for reasonable value ranges
        for feature in self.feature_columns:
            if feature in df.columns:
                min_val = df[feature].min()
                max_val = df[feature].max()
                if min_val < 0 or max_val > 1:
                    logger.warning(f"Feature {feature} has values outside [0,1] range: [{min_val}, {max_val}]")
        
        logger.info("Data validation completed")
        return True 