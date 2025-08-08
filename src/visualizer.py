"""
Data Visualization Module

Provides visualization tools for analyzing audio features and recommendation results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from config import FIGURE_SIZE, DPI
except ImportError:
    FIGURE_SIZE = (12, 8)
    DPI = 300
    logger.warning("Using default visualization settings. Create config.py for custom settings.")


class DataVisualizer:
    """
    Visualization tools for Spotify recommendation system data analysis.
    
    Provides methods to plot audio features, user preferences, and recommendation results.
    """
    
    def __init__(self, style: str = 'default'):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        self.style = style
        self.figure_size = FIGURE_SIZE
        self.dpi = DPI
        
        # Set up plotting style
        plt.style.use(style)
        sns.set_palette("husl")
        
        self.feature_columns = [
            'Acousticness', 'Danceability', 'Energy', 'Instrumentalness',
            'Liveness', 'Loudness', 'Speechiness', 'Tempo', 'Time Signature'
        ]
        
        logger.info("DataVisualizer initialized")
    
    def plot_feature_distributions(self, df: pd.DataFrame, title: str = "Audio Feature Distributions") -> None:
        """
        Plot distributions of audio features.
        
        Args:
            df: DataFrame with audio features
            title: Plot title
        """
        fig, axes = plt.subplots(3, 3, figsize=self.figure_size, dpi=self.dpi)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        for i, feature in enumerate(self.feature_columns):
            row = i // 3
            col = i % 3
            
            axes[row, col].hist(df[feature], bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
            axes[row, col].set_title(feature, fontweight='bold')
            axes[row, col].set_xlabel('Value')
            axes[row, col].set_ylabel('Density')
            axes[row, col].grid(True, alpha=0.3)
        
        # Remove empty subplot if needed
        if len(self.feature_columns) < 9:
            fig.delaxes(axes[2, 2])
        
        plt.tight_layout()
        plt.show()
        
        logger.info("Feature distribution plot created")
    
    def plot_feature_correlations(self, df: pd.DataFrame, title: str = "Audio Feature Correlations") -> None:
        """
        Plot correlation matrix of audio features.
        
        Args:
            df: DataFrame with audio features
            title: Plot title
        """
        correlation_matrix = df[self.feature_columns].corr()
        
        plt.figure(figsize=(10, 8), dpi=self.dpi)
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(
            correlation_matrix, 
            mask=mask, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={"shrink": .8}
        )
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
        
        logger.info("Feature correlation plot created")
    
    def plot_user_vs_spotify_features(self, user_df: pd.DataFrame, spotify_df: pd.DataFrame) -> None:
        """
        Compare user preferences with Spotify dataset features.
        
        Args:
            user_df: DataFrame of user's tracks
            spotify_df: DataFrame of Spotify tracks
        """
        fig, axes = plt.subplots(3, 3, figsize=self.figure_size, dpi=self.dpi)
        fig.suptitle("User Preferences vs Spotify Dataset", fontsize=16, fontweight='bold')
        
        for i, feature in enumerate(self.feature_columns):
            row = i // 3
            col = i % 3
            
            # Plot histograms
            axes[row, col].hist(user_df[feature], bins=30, alpha=0.6, label='User Tracks', color='blue', density=True)
            axes[row, col].hist(spotify_df[feature], bins=30, alpha=0.6, label='Spotify Dataset', color='red', density=True)
            
            axes[row, col].set_title(feature, fontweight='bold')
            axes[row, col].set_xlabel('Value')
            axes[row, col].set_ylabel('Density')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        # Remove empty subplot if needed
        if len(self.feature_columns) < 9:
            fig.delaxes(axes[2, 2])
        
        plt.tight_layout()
        plt.show()
        
        logger.info("User vs Spotify features comparison plot created")
    
    def plot_recommendation_results(self, recommendations: pd.DataFrame, 
                                  title: str = "Song Recommendations") -> None:
        """
        Visualize recommendation results.
        
        Args:
            recommendations: DataFrame with recommendation results
            title: Plot title
        """
        if recommendations.empty:
            logger.warning("No recommendations to plot")
            return
        
        # Create a summary plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=self.dpi)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Plot similarity scores
        ax1.bar(range(len(recommendations)), recommendations['Similarity Score'], 
                color='lightcoral', alpha=0.7)
        ax1.set_xlabel('Recommendation Rank')
        ax1.set_ylabel('Similarity Score')
        ax1.set_title('Similarity Scores by Rank')
        ax1.grid(True, alpha=0.3)
        
        # Plot top artists
        top_artists = recommendations['Artist'].value_counts().head(10)
        ax2.barh(range(len(top_artists)), top_artists.values, color='lightblue', alpha=0.7)
        ax2.set_yticks(range(len(top_artists)))
        ax2.set_yticklabels(top_artists.index)
        ax2.set_xlabel('Number of Recommendations')
        ax2.set_title('Top Recommended Artists')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        logger.info("Recommendation results plot created")
    
    def plot_feature_radar(self, track_features: pd.Series, track_name: str = "Track") -> None:
        """
        Create a radar plot for a single track's audio features.
        
        Args:
            track_features: Series with audio features
            track_name: Name of the track
        """
        # Number of features
        N = len(self.feature_columns)
        
        # Create angles for each feature
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Get feature values
        values = [track_features[feature] for feature in self.feature_columns]
        values += values[:1]  # Complete the circle
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'), dpi=self.dpi)
        
        ax.plot(angles, values, 'o-', linewidth=2, color='red', alpha=0.7)
        ax.fill(angles, values, alpha=0.25, color='red')
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(self.feature_columns)
        ax.set_ylim(0, 1)
        
        # Add title
        plt.title(f"Audio Features: {track_name}", fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.show()
        
        logger.info(f"Radar plot created for {track_name}")
    
    def plot_user_profile(self, user_df: pd.DataFrame) -> None:
        """
        Create a comprehensive user profile visualization.
        
        Args:
            user_df: DataFrame of user's tracks
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12), dpi=self.dpi)
        fig.suptitle("User Music Profile Analysis", fontsize=18, fontweight='bold')
        
        # 1. Average feature profile
        avg_features = user_df[self.feature_columns].mean()
        ax1.bar(range(len(avg_features)), avg_features.values, color='lightgreen', alpha=0.7)
        ax1.set_xticks(range(len(avg_features)))
        ax1.set_xticklabels(avg_features.index, rotation=45, ha='right')
        ax1.set_ylabel('Average Value')
        ax1.set_title('Average Audio Feature Profile')
        ax1.grid(True, alpha=0.3)
        
        # 2. Top artists
        top_artists = user_df['Artist'].value_counts().head(10)
        ax2.barh(range(len(top_artists)), top_artists.values, color='lightcoral', alpha=0.7)
        ax2.set_yticks(range(len(top_artists)))
        ax2.set_yticklabels(top_artists.index)
        ax2.set_xlabel('Number of Tracks')
        ax2.set_title('Top Artists in Library')
        ax2.grid(True, alpha=0.3)
        
        # 3. Feature correlation heatmap
        correlation_matrix = user_df[self.feature_columns].corr()
        im = ax3.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
        ax3.set_xticks(range(len(self.feature_columns)))
        ax3.set_yticks(range(len(self.feature_columns)))
        ax3.set_xticklabels(self.feature_columns, rotation=45, ha='right')
        ax3.set_yticklabels(self.feature_columns)
        ax3.set_title('Feature Correlations')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Correlation Coefficient')
        
        # 4. Track count over time (if available)
        if 'added_at' in user_df.columns:
            user_df['added_at'] = pd.to_datetime(user_df['added_at'])
            track_counts = user_df.groupby(user_df['added_at'].dt.date).size()
            ax4.plot(track_counts.index, track_counts.values, marker='o', color='purple', alpha=0.7)
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Number of Tracks Added')
            ax4.set_title('Tracks Added Over Time')
            ax4.grid(True, alpha=0.3)
        else:
            # Alternative: Energy vs Danceability scatter
            ax4.scatter(user_df['Energy'], user_df['Danceability'], alpha=0.6, color='orange')
            ax4.set_xlabel('Energy')
            ax4.set_ylabel('Danceability')
            ax4.set_title('Energy vs Danceability')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        logger.info("User profile analysis plot created")
    
    def plot_recommendation_comparison(self, recommendations_list: List[pd.DataFrame], 
                                     labels: List[str]) -> None:
        """
        Compare multiple recommendation sets.
        
        Args:
            recommendations_list: List of recommendation DataFrames
            labels: Labels for each recommendation set
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=self.dpi)
        fig.suptitle("Recommendation Comparison", fontsize=18, fontweight='bold')
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        # 1. Average similarity scores
        avg_scores = [rec['Similarity Score'].mean() for rec in recommendations_list]
        axes[0, 0].bar(labels, avg_scores, color=colors[:len(labels)], alpha=0.7)
        axes[0, 0].set_ylabel('Average Similarity Score')
        axes[0, 0].set_title('Average Similarity Scores')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Number of recommendations
        rec_counts = [len(rec) for rec in recommendations_list]
        axes[0, 1].bar(labels, rec_counts, color=colors[:len(labels)], alpha=0.7)
        axes[0, 1].set_ylabel('Number of Recommendations')
        axes[0, 1].set_title('Number of Recommendations')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Feature distributions comparison
        for i, (rec, label, color) in enumerate(zip(recommendations_list, labels, colors)):
            if 'Energy' in rec.columns:
                axes[1, 0].hist(rec['Energy'], alpha=0.5, label=label, color=color, bins=20)
        axes[1, 0].set_xlabel('Energy')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Energy Distribution Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Danceability comparison
        for i, (rec, label, color) in enumerate(zip(recommendations_list, labels, colors)):
            if 'Danceability' in rec.columns:
                axes[1, 1].hist(rec['Danceability'], alpha=0.5, label=label, color=color, bins=20)
        axes[1, 1].set_xlabel('Danceability')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Danceability Distribution Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        logger.info("Recommendation comparison plot created")
    
    def save_plots(self, filename: str, dpi: int = 300) -> None:
        """
        Save the current figure.
        
        Args:
            filename: Name of the file to save
            dpi: Resolution for saving
        """
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        logger.info(f"Plot saved as {filename}")
    
    def create_summary_report(self, user_df: pd.DataFrame, recommendations: pd.DataFrame) -> str:
        """
        Create a text summary report of the analysis.
        
        Args:
            user_df: DataFrame of user's tracks
            recommendations: DataFrame of recommendations
            
        Returns:
            Summary report as string
        """
        report = []
        report.append("=" * 50)
        report.append("SPOTIFY RECOMMENDATION SYSTEM SUMMARY REPORT")
        report.append("=" * 50)
        report.append("")
        
        # User profile summary
        report.append("USER PROFILE SUMMARY:")
        report.append(f"- Total tracks in library: {len(user_df)}")
        report.append(f"- Unique artists: {user_df['Artist'].nunique()}")
        report.append("")
        
        # Top artists
        top_artists = user_df['Artist'].value_counts().head(5)
        report.append("TOP 5 ARTISTS:")
        for artist, count in top_artists.items():
            report.append(f"- {artist}: {count} tracks")
        report.append("")
        
        # Audio feature averages
        avg_features = user_df[self.feature_columns].mean()
        report.append("AVERAGE AUDIO FEATURES:")
        for feature, value in avg_features.items():
            report.append(f"- {feature}: {value:.3f}")
        report.append("")
        
        # Recommendations summary
        report.append("RECOMMENDATION SUMMARY:")
        report.append(f"- Total recommendations: {len(recommendations)}")
        if 'Similarity Score' in recommendations.columns:
            report.append(f"- Average similarity score: {recommendations['Similarity Score'].mean():.3f}")
            report.append(f"- Highest similarity score: {recommendations['Similarity Score'].max():.3f}")
        report.append("")
        
        # Top recommended artists
        if 'Artist' in recommendations.columns:
            top_rec_artists = recommendations['Artist'].value_counts().head(5)
            report.append("TOP 5 RECOMMENDED ARTISTS:")
            for artist, count in top_rec_artists.items():
                report.append(f"- {artist}: {count} recommendations")
        
        report.append("")
        report.append("=" * 50)
        
        return "\n".join(report) 