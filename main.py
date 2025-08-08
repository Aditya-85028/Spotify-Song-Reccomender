#!/usr/bin/env python3
"""
Spotify Song Recommender - Main Application

A command-line interface for the Spotify recommendation system that allows users
to get personalized song recommendations based on their Spotify playlists.

Usage:
    python main.py --song "Song Name" --artist "Artist Name"
    python main.py --profile
    python main.py --interactive
    python main.py --analyze
"""

import argparse
import sys
import os
import logging
import pandas as pd
from typing import Optional, Dict, List
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.spotify_client import SpotifyClient
from src.data_processor import DataProcessor
from src.recommender import SpotifyRecommender
from src.visualizer import DataVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('spotify_recommender.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SpotifyRecommenderApp:
    """
    Main application class for the Spotify Song Recommender.
    
    Handles the complete workflow from data collection to recommendation generation.
    """
    
    def __init__(self):
        """Initialize the application components."""
        self.spotify_client = None
        self.data_processor = DataProcessor()
        self.recommender = SpotifyRecommender()
        self.visualizer = DataVisualizer()
        self.user_tracks = None
        self.unlistened_tracks = None
        self.song_data = None
        self.is_model_trained = False
        
        logger.info("Spotify Recommender App initialized")
    
    def setup_spotify_client(self) -> bool:
        """
        Initialize and authenticate with Spotify API.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Setting up Spotify client...")
            self.spotify_client = SpotifyClient()
            logger.info("✅ Spotify client setup successful")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to setup Spotify client: {str(e)}")
            return False
    
    def collect_data(self, force_refresh: bool = False) -> bool:
        """
        Collect and process user and Spotify data.
        
        Args:
            force_refresh: Whether to force data collection even if cached data exists
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if processed data already exists
            if not force_refresh:
                user_tracks = self.data_processor.load_data('user_tracks_processed.csv')
                spotify_tracks = self.data_processor.load_data('spotify_tracks_processed.csv')
                combined_data = self.data_processor.load_data('combined_dataset.csv')
                
                if all(data is not None for data in [user_tracks, spotify_tracks, combined_data]):
                    logger.info("📊 Using cached data...")
                    self.user_tracks = user_tracks
                    self.unlistened_tracks = spotify_tracks
                    self.song_data = spotify_tracks[self.data_processor.feature_columns].copy()
                    return True
            
            logger.info("📊 Collecting fresh data from Spotify...")
            
            # Get user playlists
            user_playlists = self.spotify_client.get_user_playlists()
            playlist_ids = [playlist['id'] for playlist in user_playlists]
            
            # Extract user tracks
            logger.info("🎵 Extracting user tracks...")
            user_tracks = self.spotify_client.playlist_to_dataframe(playlist_ids, liked=1)
            
            # Get Spotify dataset
            logger.info("🎵 Collecting Spotify dataset...")
            spotify_playlist_ids = self.spotify_client.get_spotify_playlists()
            spotify_tracks = self.spotify_client.playlist_to_dataframe(spotify_playlist_ids, liked=0)
            
            # Process and combine data
            logger.info("🧹 Processing and cleaning data...")
            user_tracks_clean = self.data_processor.clean_data(user_tracks)
            spotify_tracks_clean = self.data_processor.clean_data(spotify_tracks)
            combined_dataset = self.data_processor.combine_datasets(user_tracks_clean, spotify_tracks_clean)
            
            # Scale features
            logger.info("⚖️ Scaling features...")
            scaled_dataset = self.data_processor.scale_features(combined_dataset)
            
            # Prepare final datasets
            self.user_tracks, self.unlistened_tracks, self.song_data = \
                self.data_processor.prepare_datasets(scaled_dataset)
            
            # Save processed data
            logger.info("💾 Saving processed data...")
            self.data_processor.save_data(self.user_tracks, 'user_tracks_processed.csv')
            self.data_processor.save_data(self.unlistened_tracks, 'spotify_tracks_processed.csv')
            self.data_processor.save_data(scaled_dataset, 'combined_dataset.csv')
            
            logger.info("✅ Data collection completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Data collection failed: {str(e)}")
            return False
    
    def train_model(self) -> bool:
        """
        Train the recommendation model.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.user_tracks is None or self.unlistened_tracks is None or self.song_data is None:
                logger.error("❌ No data available for training. Run data collection first.")
                return False
            
            logger.info("🤖 Training recommendation model...")
            self.recommender.fit(self.user_tracks, self.unlistened_tracks, self.song_data)
            self.is_model_trained = True
            
            logger.info("✅ Model training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {str(e)}")
            return False
    
    def get_song_recommendations(self, song_name: str, artist: str, 
                               num_recommendations: int = 10) -> Optional[pd.DataFrame]:
        """
        Get recommendations for a specific song.
        
        Args:
            song_name: Name of the song
            artist: Name of the artist
            num_recommendations: Number of recommendations to return
            
        Returns:
            DataFrame with recommendations or None if failed
        """
        try:
            if not self.is_model_trained:
                logger.error("❌ Model not trained. Run training first.")
                return None
            
            logger.info(f"🎵 Getting recommendations for '{song_name}' by {artist}...")
            recommendations = self.recommender.get_recommendations(
                song_name, artist, num_recommendations
            )
            
            logger.info(f"✅ Generated {len(recommendations)} recommendations")
            return recommendations
            
        except ValueError as e:
            logger.error(f"❌ Song not found in user library: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"❌ Failed to get recommendations: {str(e)}")
            return None
    
    def get_profile_recommendations(self, num_recommendations: int = 15) -> Optional[pd.DataFrame]:
        """
        Get recommendations based on user's overall profile.
        
        Args:
            num_recommendations: Number of recommendations to return
            
        Returns:
            DataFrame with recommendations or None if failed
        """
        try:
            if not self.is_model_trained:
                logger.error("❌ Model not trained. Run training first.")
                return None
            
            logger.info("🎵 Getting profile-based recommendations...")
            recommendations = self.recommender.get_user_profile_recommendations(num_recommendations)
            
            logger.info(f"✅ Generated {len(recommendations)} profile recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Failed to get profile recommendations: {str(e)}")
            return None
    
    def display_recommendations(self, recommendations: pd.DataFrame, title: str = "Recommendations"):
        """
        Display recommendations in a formatted way.
        
        Args:
            recommendations: DataFrame with recommendations
            title: Title for the display
        """
        if recommendations is None or recommendations.empty:
            print("❌ No recommendations available.")
            return
        
        print(f"\n🎵 {title}")
        print("=" * 80)
        
        # Display recommendations in a table format
        for i, (_, rec) in enumerate(recommendations.iterrows(), 1):
            print(f"{i:2d}. {rec['Title']:<40} - {rec['Artist']:<30}")
            if 'Similarity Score' in rec:
                print(f"    Similarity: {rec['Similarity Score']:.3f}")
            if 'Danceability' in rec and 'Energy' in rec:
                print(f"    Danceability: {rec['Danceability']:.2f} | Energy: {rec['Energy']:.2f}")
            print()
        
        # Save recommendations
        filename = f"recommendations_{title.lower().replace(' ', '_')}.csv"
        recommendations.to_csv(filename, index=False)
        print(f"💾 Recommendations saved to: {filename}")
    
    def show_user_stats(self):
        """Display user statistics and profile information."""
        if self.user_tracks is None:
            print("❌ No user data available.")
            return
        
        print("\n📊 USER PROFILE STATISTICS")
        print("=" * 50)
        print(f"Total tracks in library: {len(self.user_tracks):,}")
        print(f"Unique artists: {self.user_tracks['Artist'].nunique():,}")
        
        # Top artists
        top_artists = self.user_tracks['Artist'].value_counts().head(5)
        print(f"\n🎵 Top 5 Artists:")
        for artist, count in top_artists.items():
            print(f"   {artist}: {count} tracks")
        
        # Audio feature averages
        feature_columns = self.data_processor.feature_columns
        avg_features = self.user_tracks[feature_columns].mean()
        print(f"\n🎼 Average Audio Features:")
        for feature, value in avg_features.items():
            print(f"   {feature}: {value:.3f}")
    
    def interactive_mode(self):
        """Run the application in interactive mode."""
        print("\n🎵 SPOTIFY SONG RECOMMENDER - INTERACTIVE MODE")
        print("=" * 60)
        
        # Setup and collect data
        if not self.setup_spotify_client():
            return
        
        if not self.collect_data():
            return
        
        if not self.train_model():
            return
        
        # Show user stats
        self.show_user_stats()
        
        while True:
            print("\n" + "=" * 60)
            print("Choose an option:")
            print("1. Get song recommendations")
            print("2. Get profile recommendations")
            print("3. Show user statistics")
            print("4. List available songs")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                song_name = input("Enter song name: ").strip()
                artist = input("Enter artist name: ").strip()
                num_recs = input("Number of recommendations (default 10): ").strip()
                num_recs = int(num_recs) if num_recs.isdigit() else 10
                
                recommendations = self.get_song_recommendations(song_name, artist, num_recs)
                self.display_recommendations(recommendations, f"Recommendations for '{song_name}'")
                
            elif choice == '2':
                num_recs = input("Number of recommendations (default 15): ").strip()
                num_recs = int(num_recs) if num_recs.isdigit() else 15
                
                recommendations = self.get_profile_recommendations(num_recs)
                self.display_recommendations(recommendations, "Profile-Based Recommendations")
                
            elif choice == '3':
                self.show_user_stats()
                
            elif choice == '4':
                if self.user_tracks is not None:
                    print(f"\n📋 Available songs in your library (showing first 20):")
                    available_songs = self.user_tracks[['Track Name', 'Artist']].head(20)
                    for i, (_, song) in enumerate(available_songs.iterrows(), 1):
                        print(f"{i:2d}. {song['Track Name']:<40} - {song['Artist']}")
                else:
                    print("❌ No user data available.")
                    
            elif choice == '5':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please try again.")


def main():
    """Main function to run the Spotify recommender application."""
    parser = argparse.ArgumentParser(
        description="Spotify Song Recommender - Get personalized music recommendations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --song "Woman" --artist "Doja Cat"
  python main.py --song "Bohemian Rhapsody" --artist "Queen" --count 15
  python main.py --profile --count 20
  python main.py --interactive
  python main.py --analyze
        """
    )
    
    # Recommendation options
    parser.add_argument('--song', type=str, help='Song name for recommendations')
    parser.add_argument('--artist', type=str, help='Artist name for recommendations')
    parser.add_argument('--count', type=int, default=10, help='Number of recommendations (default: 10)')
    
    # Mode options
    parser.add_argument('--profile', action='store_true', help='Get profile-based recommendations')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--analyze', action='store_true', help='Show user profile analysis')
    
    # Data options
    parser.add_argument('--refresh', action='store_true', help='Force refresh of data collection')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching and collect fresh data')
    
    # Output options
    parser.add_argument('--save', action='store_true', help='Save recommendations to CSV file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize application
    app = SpotifyRecommenderApp()
    
    try:
        # Setup Spotify client
        if not app.setup_spotify_client():
            sys.exit(1)
        
        # Collect data
        force_refresh = args.refresh or args.no_cache
        if not app.collect_data(force_refresh=force_refresh):
            sys.exit(1)
        
        # Train model
        if not app.train_model():
            sys.exit(1)
        
        # Handle different modes
        if args.interactive:
            app.interactive_mode()
            
        elif args.analyze:
            app.show_user_stats()
            
        elif args.profile:
            recommendations = app.get_profile_recommendations(args.count)
            app.display_recommendations(recommendations, "Profile-Based Recommendations")
            
        elif args.song and args.artist:
            recommendations = app.get_song_recommendations(args.song, args.artist, args.count)
            app.display_recommendations(recommendations, f"Recommendations for '{args.song}'")
            
        else:
            # Default: show help
            parser.print_help()
            print("\n💡 Try running with --interactive for an interactive experience!")
    
    except KeyboardInterrupt:
        print("\n\n👋 Application interrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Application error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 