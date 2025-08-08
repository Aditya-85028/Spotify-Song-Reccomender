# Spotify Song Recommender

A machine learning-based music recommendation system that analyzes your Spotify playlists and suggests new songs based on your listening preferences.

## 🎵 Features

- **Personalized Recommendations**: Analyzes your Spotify playlists to understand your music taste
- **Audio Feature Analysis**: Uses Spotify's audio features (danceability, energy, acousticness, etc.) for accurate recommendations
- **K-Nearest Neighbors Algorithm**: Implements cosine similarity-based recommendation engine
- **Interactive Interface**: Easy-to-use functions to get song recommendations
- **Data Visualization**: Visualize your music preferences and audio feature distributions

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- Spotify account
- Spotify Developer credentials

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/spotify-song-recommender.git
cd spotify-song-recommender
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Spotify API credentials:
   - Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
   - Create a new application
   - Copy your `Client ID` and `Client Secret`
   - Add `http://localhost:7777/callback` to your redirect URIs

4. Configure your credentials:
   - Copy `config_template.py` to `config.py`
   - Fill in your Spotify credentials in `config.py`

### Usage

#### Option 1: Command Line Interface (Recommended)

The easiest way to use the recommender is through the command-line interface:

```bash
# Get recommendations for a specific song
python main.py --song "Woman" --artist "Doja Cat"

# Get recommendations with custom count
python main.py --song "Bohemian Rhapsody" --artist "Queen" --count 15

# Get profile-based recommendations
python main.py --profile --count 20

# Run in interactive mode
python main.py --interactive

# Show user profile analysis
python main.py --analyze

# Force refresh data collection
python main.py --song "Song Name" --artist "Artist Name" --refresh
```

#### Option 2: Jupyter Notebook

For more detailed analysis and visualization:

1. Run the main notebook:
```bash
jupyter notebook spotify_recommender.ipynb
```

2. Follow the notebook cells to:
   - Authenticate with Spotify
   - Load your playlists
   - Generate recommendations

## 📁 Project Structure

```
spotify-song-recommender/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config_template.py        # Template for configuration
├── main.py                   # Command-line interface
├── src/
│   ├── __init__.py
│   ├── spotify_client.py     # Spotify API client
│   ├── data_processor.py     # Data processing utilities
│   ├── recommender.py        # Recommendation engine
│   └── visualizer.py         # Data visualization
├── data/                     # Data storage
│   ├── user_tracks.csv
│   └── spotify_tracks.csv
```

## 🔧 Configuration

Create a `config.py` file with your Spotify credentials:

```python
# Spotify API Credentials
SPOTIFY_CLIENT_ID = 'your_client_id_here'
SPOTIFY_CLIENT_SECRET = 'your_client_secret_here'
SPOTIFY_REDIRECT_URI = 'http://localhost:7777/callback'
SPOTIFY_USERNAME = 'your_spotify_username'

# Data settings
DATA_DIR = 'data/'
MAX_PLAYLISTS = 50
MAX_TRACKS_PER_PLAYLIST = 100
```

## 📊 How It Works

1. **Data Collection**: Fetches your Spotify playlists and extracts audio features for each track
2. **Feature Engineering**: Normalizes audio features using Min-Max scaling
3. **Model Training**: Uses K-Nearest Neighbors with cosine similarity
4. **Recommendation Generation**: Finds similar songs based on audio feature similarity

### Audio Features Used

- **Acousticness**: How acoustic the song is
- **Danceability**: How suitable the song is for dancing
- **Energy**: Perceived energy level
- **Instrumentalness**: Amount of vocals vs. instrumental content
- **Liveness**: Presence of audience in the recording
- **Loudness**: Overall loudness in dB
- **Speechiness**: Presence of spoken words
- **Tempo**: Estimated tempo in BPM
- **Time Signature**: Estimated time signature

## 🎯 Example Usage

### Command Line Interface

```bash
# Get recommendations for a specific song
python main.py --song "Woman" --artist "Doja Cat"

# Get profile-based recommendations
python main.py --profile --count 15

# Interactive mode for multiple recommendations
python main.py --interactive
```

### Programmatic Usage

```python
from src.recommender import SpotifyRecommender

# Initialize recommender
recommender = SpotifyRecommender()

# Get recommendations based on a song
recommendations = recommender.get_recommendations(
    song_name="Woman",
    artist="Doja Cat",
    num_recommendations=10
)

print(recommendations)
```

## 📈 Data Visualization

The project includes visualization tools to analyze:
- Audio feature distributions
- User listening patterns
- Feature correlations
- Recommendation quality metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Spotify Web API](https://developer.spotify.com/documentation/web-api/) for providing music data
- [Spotipy](https://spotipy.readthedocs.io/) for Python Spotify API wrapper
- [scikit-learn](https://scikit-learn.org/) for machine learning algorithms


---
