import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

class SpotifyDataLoader:
    def __init__(self, file_path: str='data/spotify_songs.csv'):
        """
            initialize the data loader with the path to the 30000 Spotify songs
        """
        self.file_path = file_path
        self.data = None
        self.audio_features = ['danceability', 'energy', 'acousticness', 'valence', 'tempo']
        self.metadata_features = ['track_id', 'track_name', 'track_artist',
                                'track_album_release_date', 'playlist_genre',
                                'track_popularity', 'duration_ms']

    def load_data(self) -> None:
        self.data = pd.read_csv(self.file_path)
        self._clean_data()

    def _clean_data(self) -> None:
        self.data['track_album_release_date'] = pd.to_datetime(
            self.data['track_album_release_date'], errors='coerce'
        )

        self.data['release_year'] = self.data['track_album_release_date'].dt.year
        self.data = self.data.dropna(subset=self.audio_features + ['track_id', 'track_name'])
        for feature in self.audio_features:
            self.data[feature] = self.data[feature].astype(float)

    def get_song_by_id(self, track_id: str) -> Dict:
        '''
            get a song by id: super useful when we know the ID and need the features
        '''
        if track_id not in self.data['track_id'].values:
            return None

        song = self.data[self.data['track_id'] == track_id].iloc[0]
        return self._format_song_dict(song)

    def get_random_songs(self, n: int = 1) -> List[Dict]:
        '''
            gets a number of random songs from the dataset - great for recommendations!
        '''
        songs = self.data.sample(n=n)
        return [self._format_song_dict(song) for _, song in songs.iterrows()]

    def get_songs_by_genre(self, genre: str, n: int = 1) -> List[Dict]:
        '''
            gets a number of songs by genre
        '''
        genre_songs = self.data[self.data['playlist_genre'] == genre]
        if len(genre_songs) == 0:
            return []

        songs = genre_songs.sample(n=min(n, len(genre_songs)))
        return [self._format_song_dict(song) for _, song in songs.iterrows()]

    def get_songs_by_year(self, year: int, n: int = 1) -> List[Dict]:
        '''
            gets a bunch of songs by year
        '''
        year_songs = self.data[self.data['release_year'] == year]
        if len(year_songs) == 0:
            return []

        songs = year_songs.sample(n=min(n, len(year_songs)))
        return [self._format_song_dict(song) for _, song in songs.iterrows()]

    def get_songs_by_artist(self, artist: str, n: int = 1) -> List[Dict]:
        '''
            gets a number of songs by artist - not so useful
        '''
        artist_songs = self.data[self.data['track_artist'] == artist]
        if len(artist_songs) == 0:
            return []

        songs = artist_songs.sample(n=min(n, len(artist_songs)))
        return [self._format_song_dict(song) for _, song in songs.iterrows()]

    def _format_song_dict(self, song: pd.Series) -> Dict:
        '''
            dumps song into a nice dictionary with all of the important features we generated
        '''
        return {
            'id': song['track_id'],
            'name': song['track_name'],
            'artist': song['track_artist'],
            'genre': song['playlist_genre'],
            'release_year': song['release_year'],
            'popularity': song['track_popularity'],
            'duration_ms': song['duration_ms'],
            'danceability': song['danceability'],
            'energy': song['energy'],
            'acousticness': song['acousticness'],
            'valence': song['valence'],
            'tempo': song['tempo'],
        }

    def get_unique_genres(self) -> List[str]:
        return self.data['playlist_genre'].unique().tolist()

    def get_unique_artists(self) -> List[str]:
        return self.data['track_artist'].unique().tolist()

    def get_year_range(self) -> tuple:
        return (self.data['release_year'].min(),
                self.data['release_year'].max())

    def get_similar_songs_by_features(self, song_id: str, n: int = 5) -> List[Dict]:
        '''
            this is the content based filtering!
            using cosine similarity to measure song-feature-vector-distances
        '''
        if song_id not in self.data['track_id'].values:
            return []

        base_song = self.data[self.data['track_id'] == song_id].iloc[0]
        feature_vector = base_song[self.audio_features].values.reshape(1, -1)

        all_features = self.data[self.audio_features].values
        similarities = cosine_similarity(feature_vector, all_features)[0]

        similar_indices = np.argsort(similarities)[::-1][1:n+1]
        similar_songs = self.data.iloc[similar_indices]

        return [self._format_song_dict(song) for _, song in similar_songs.iterrows()]

    def get_similar_songs_by_metadata(self, song_id: str, n: int = 5) -> List[Dict]:
        if song_id not in self.data['track_id'].values:
            return []

        base_song = self.data[self.data['track_id'] == song_id].iloc[0]

        same_artist = self.data['track_artist'] == base_song['track_artist']
        same_genre = self.data['playlist_genre'] == base_song['playlist_genre']
        same_era = abs(self.data['release_year'] - base_song['release_year']) <= 5

        similarity_score = (same_artist.astype(int) * 3 +
                            same_genre.astype(int) * 2 +
                            same_era.astype(int))

        self.data['temp_score'] = similarity_score
        similar_songs = self.data.nlargest(n + 1, 'temp_score')
        similar_songs = similar_songs[similar_songs['track_id'] != song_id].head(n)
        self.data = self.data.drop('temp_score', axis=1)

        return [self._format_song_dict(song) for _, song in similar_songs.iterrows()]

    def get_top_songs_by_genre(self, genre: str, n: int = 5) -> List[Dict]:
        genre_songs = self.data[self.data['playlist_genre'] == genre]
        top_songs = genre_songs.nlargest(n, 'track_popularity')
        return [self._format_song_dict(song) for _, song in top_songs.iterrows()]

    def get_artist_top_songs(self, artist: str, n: int = 5) -> List[Dict]:
        artist_songs = self.data[self.data['track_artist'] == artist]
        top_songs = artist_songs.nlargest(n, 'track_popularity')
        return [self._format_song_dict(song) for _, song in top_songs.iterrows()]

    def get_popular_songs_in_timeframe(self, start_year: int, end_year: int, n: int = 5) -> List[Dict]:
        timeframe_songs = self.data[
            (self.data['release_year'] >= start_year) &
            (self.data['release_year'] <= end_year)
        ]
        top_songs = timeframe_songs.nlargest(n, 'track_popularity')
        return [self._format_song_dict(song) for _, song in top_songs.iterrows()]

    def get_genre_distribution(self) -> Dict[str, int]:
        return self.data['playlist_genre'].value_counts().to_dict()

    def get_feature_vector(self, song_id: str) -> np.ndarray:
        if song_id not in self.data['track_id'].values:
            return None
        return self.data[self.data['track_id'] == song_id][self.audio_features].values[0]

    def get_feature_matrix(self) -> np.ndarray:
        return self.data[self.audio_features].values
