import uuid
import random
import numpy as np
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime, timedelta

class GenrePreference(BaseModel):
    '''
        denotes the genre preference of the agent with a weight
    '''
    genre: str
    weight: float = Field(ge=0, le=1)

class AudioFeaturePreference(BaseModel):
    '''
        denotes the audio feature that we'll weight the heaviest
    '''
    feature_name: str
    preferred_range: tuple[float, float]
    weight: float = Field(ge=0, le=1)

class ListeningBehavior(BaseModel):
    '''
        denotes the listening behavior, probabilistic since we are workign with agents
    '''
    avg_session_length: int = Field(ge=10, le=180)
    skip_probability: float = Field(ge=0, le=1)
    rating_probability: float = Field(ge=0, le=1)
    playlist_creation_frequency: float = Field(ge=0, le=1)

class Playlist(BaseModel):
    '''
        denotes a playlist created by the agent
    '''
    playlist_id: str
    name: str
    description: str
    songs: List[str] = []
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    genre_focus: Optional[str] = None

class MusicAgent(BaseModel):
    agent_id: str
    archetype: str

    genre_preferences: List[GenrePreference]
    feature_preferences: List[AudioFeaturePreference]
    behavior: ListeningBehavior

    active_hours: List[int] = Field(default_factory=lambda: list(range(9, 23)))
    playlists: List[Playlist] = Field(default_factory=list)

    def should_listen_to_song(self, song_features: Dict) -> bool:
        """
            decides if agent would be interested in a song based on preferences
        """
        score = 0
        total_weight = 0

        # is the current song genre a preferenced genre?
        if song_features.get('genre') in [gp.genre for gp in self.genre_preferences]:
            genre_pref = next(gp for gp in self.genre_preferences if gp.genre == song_features['genre'])
            score += genre_pref.weight
            total_weight += 1

        # do the features of the song align with the agent's preferences?
        for feature_pref in self.feature_preferences:
            if feature_pref.feature_name in song_features:
                feature_value = song_features[feature_pref.feature_name]
                min_val, max_val = feature_pref.preferred_range
                if min_val <= feature_value <= max_val:
                    score += feature_pref.weight
                total_weight += 1

        return (score / total_weight) > 0.6 if total_weight > 0 else False

    def generate_rating(self, song_features: Dict) -> Optional[int]:
        """
            generates a rating based on song features and agent
            the idea here is: if a user is giving a rating; they either really liked or disliked
            so we have strong heuristics for either
        """
        if random.random() > self.behavior.rating_probability:
            return None

        base_score = 3
        score_adjustments = 0
        count = 0

        # if preferred genre, simulate a better rating
        if song_features.get('genre') in [gp.genre for gp in self.genre_preferences]:
            genre_pref = next(gp for gp in self.genre_preferences if gp.genre == song_features['genre'])
            score_adjustments += 2 * genre_pref.weight
            count += 1

        # if preferred feature values align, then simulate a better rating
        for feature_pref in self.feature_preferences:
            if feature_pref.feature_name in song_features:
                feature_value = song_features[feature_pref.feature_name]
                min_val, max_val = feature_pref.preferred_range
                if min_val <= feature_value <= max_val:
                    score_adjustments += 2 * feature_pref.weight
                else:
                    score_adjustments -= 2 * feature_pref.weight
                count += 1

        if count > 0:
            final_score = base_score + (score_adjustments / count)
            return max(1, min(5, round(final_score)))
        return base_score

    def generate_playlist_name(self) -> str:
        """
            returns playlist name as well as metadata about the playlist
            This is actually very similar to spotify's naming scheme
        """
        time_contexts = ["Morning", "Evening", "Late Night", "Weekend"]
        mood_contexts = ["Chill", "Energetic", "Focus", "Relaxing", "Upbeat"]
        activity_contexts = ["Working", "Studying", "Workout", "Drive"]

        if self.archetype == "classical_connoisseur":
            templates = [
                "Classical {mood}",
                "Orchestra Essentials",
                "Sophisticated {time}",
                "{mood} Orchestral Mix"
            ]
        elif self.archetype == "eclectic_explorer":
            templates = [
                "Eclectic {mood} Mix",
                "Genre-Bending {activity}",
                "Global Sounds",
                "Discovery {time}"
            ]
        else:
            templates = [
                "{mood} {time} Mix",
                "{activity} Essentials",
                "My {mood} Playlist",
                "{time} {activity} Mix"
            ]

        template = random.choice(templates)
        return template.format(
            time=random.choice(time_contexts),
            mood=random.choice(mood_contexts),
            activity=random.choice(activity_contexts)
        )
    def create_playlist(self, available_songs: List[Dict], playlist_type: str = "genre") -> Optional[Playlist]:
        '''
            returns a playlist based on the playlist type parameter
        '''
        if random.random() > self.behavior.playlist_creation_frequency:
                return None

        genre_weights = [pref.weight for pref in self.genre_preferences]
        primary_genres = [pref.genre for pref in self.genre_preferences]

        if playlist_type == "genre":
            # creartes playlist based on the top genre of the agent
            chosen_genre = random.choices(primary_genres, weights=genre_weights)[0]
            name = f"My {chosen_genre.title()} Mix Vol.{len([p for p in self.playlists if p.genre_focus == chosen_genre]) + 1}"
            description = f"A curated collection of {chosen_genre} tracks"
            genre_focus = chosen_genre
        else:
            # creates mixed genre playlist; still weighing agent's preferred
            name = self.generate_playlist_metadata()
            description = "A personalized mix based on my favorite genres"
            genre_focus = random.choices(primary_genres + ['mixed'], weights=genre_weights + [0.3])[0]

        # if song is something the agent would listen to: add to suitable songs
        suitable_songs = [
            song for song in available_songs
            if self.should_listen_to_song(song)
        ]

        # Select songs for playlist
        if suitable_songs:
            playlist_size = random.randint(10, 30)
            selected_songs = random.sample(
                suitable_songs,
                min(playlist_size, len(suitable_songs))
            )

            new_playlist = Playlist(
                playlist_id=f"pl_{uuid.uuid4().hex[:8]}",
                name=name,
                description=description,
                songs=[song['id'] for song in selected_songs],
                genre_focus=genre_focus
            )
            self.playlists.append(new_playlist)
            return new_playlist
        return None

def create_genre_focused_agent(agent_id: str, primary_genre: str,
                             secondary_genres: List[str]) -> MusicAgent:
    """
        agent preference builder
    """
    genre_prefs = [
        GenrePreference(genre=primary_genre, weight=0.8)
    ] + [
        GenrePreference(genre=g, weight=0.4)
        for g in secondary_genres
    ]

    feature_prefs = [
        AudioFeaturePreference(
            feature_name="danceability",
            preferred_range=(0.6, 1.0),
            weight=0.7
        ),
        AudioFeaturePreference(
            feature_name="energy",
            preferred_range=(0.5, 0.9),
            weight=0.6
        ),
        AudioFeaturePreference(
            feature_name="valence",
            preferred_range=(0.4, 0.8),
            weight=0.5
        )
    ]

    behavior = ListeningBehavior(
        avg_session_length=45,
        skip_probability=0.3,
        rating_probability=0.2,
        playlist_creation_frequency=0.1
    )

    return MusicAgent(
        agent_id=agent_id,
        archetype=f"{primary_genre}_enthusiast",
        genre_preferences=genre_prefs,
        feature_preferences=feature_prefs,
        behavior=behavior
    )

def create_pop_agent(agent_id: str) -> MusicAgent:
    return create_genre_focused_agent(
        agent_id=agent_id,
        primary_genre="pop",
        secondary_genres=["dance", "electronic"]
    )

def create_classical_agent(agent_id: str) -> MusicAgent:
    return MusicAgent(
        agent_id=agent_id,
        archetype="classical_connoisseur",
        genre_preferences=[
            GenrePreference(genre="classical", weight=0.9),
            GenrePreference(genre="orchestra", weight=0.7)
        ],
        feature_preferences=[
            AudioFeaturePreference(
                feature_name="acousticness",
                preferred_range=(0.8, 1.0),
                weight=0.9
            ),
            AudioFeaturePreference(
                feature_name="instrumentalness",
                preferred_range=(0.8, 1.0),
                weight=0.8
            )
        ],
        behavior=ListeningBehavior(
            avg_session_length=60,
            skip_probability=0.1,
            rating_probability=0.4,
            playlist_creation_frequency=0.05
        )
    )

def create_general_agent(agent_id: str) -> MusicAgent:
    return MusicAgent(
        agent_id=agent_id,
        archetype="general_explorer",
        genre_preferences=[
            GenrePreference(genre=genre, weight=0.4)
            for genre in ["indie", "experimental", "world", "jazz", "electronic"]
        ],
        feature_preferences=[
            AudioFeaturePreference(
                feature_name="acousticness",
                preferred_range=(0.2, 0.8),
                weight=0.5
            ),
            AudioFeaturePreference(
                feature_name="energy",
                preferred_range=(0.3, 0.7),
                weight=0.5
            ),
            AudioFeaturePreference(
                feature_name="valence",
                preferred_range=(0.3, 0.7),
                weight=0.5
            )
        ],
        behavior=ListeningBehavior(
            avg_session_length=90,
            skip_probability=0.4,
            rating_probability=0.3,
            playlist_creation_frequency=0.2
        )
    )
