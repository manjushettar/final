from typing import Dict, List
import random
import pandas as pd
from datetime import datetime, timedelta
from agents import (
    MusicAgent,
    create_pop_agent,
    create_classical_agent,
    create_genre_focused_agent
)
from dataloader import SpotifyDataLoader

class OnboardingManager:
    def __init__(self, data_loader: SpotifyDataLoader):
        self.data_loader = data_loader
        self.onboarding_histories = {}

    def onboard_agent(self, agent: MusicAgent) -> Dict:
        '''
            onboard an agent and record the history
        '''
        selected_songs = self._select_onboarding_songs(agent)
        ratings = self._collect_ratings(agent, selected_songs)
        history = self._create_onboarding_history(agent, ratings)
        self.onboarding_histories[agent.agent_id] = history
        return history

    def _select_onboarding_songs(self, agent: MusicAgent) -> List[Dict]:
        '''
            show 5 random songs to the agent from a random year range
        '''
        selected_songs = []

        primary_genres = [pref.genre for pref in agent.genre_preferences]
        for genre in primary_genres[:2]:
            genre_songs = self.data_loader.get_top_songs_by_genre(genre, n=2)
            selected_songs.extend(genre_songs)

        year_range = self.data_loader.get_year_range()
        recent_songs = self.data_loader.get_popular_songs_in_timeframe(
            year_range[1] - 2,
            year_range[1],
            n=1
        )
        selected_songs.extend(recent_songs)

        return selected_songs[:5]

    def _collect_ratings(self, agent: MusicAgent, songs: List[Dict]) -> List[Dict]:
        '''
            collect the ratings for the agent's reactions to the songs
        '''
        ratings = []
        current_time = datetime.now()

        for idx, song in enumerate(songs):
            rating = agent.generate_rating(song)
            skip = random.random() < agent.behavior.skip_probability

            interaction = {
                'agent_id': agent.agent_id,
                'song_id': song['id'],
                'rating': rating,
                'skipped': skip,
                'timestamp': current_time - timedelta(minutes=30*(5-idx)),
                'type': 'onboarding'
            }
            ratings.append(interaction)

        return ratings

    def _create_onboarding_history(self, agent: MusicAgent, ratings: List[Dict]) -> Dict:
        '''
            creates a onboarding history by collecting the genre preferences
            and feature preferences of each agent
        '''
        avg_rating_by_genre = {}
        feature_preferences = {
            'danceability': [],
            'energy': [],
            'acousticness': [],
            'valence': [],
            'tempo': []
        }


        for rating in ratings:
            song = self.data_loader.get_song_by_id(rating['song_id'])
            if song:
                genre = song['genre']
                if genre not in avg_rating_by_genre:
                    avg_rating_by_genre[genre] = []
                if rating['rating']:
                    avg_rating_by_genre[genre].append(rating['rating'])

                if rating['rating'] and rating['rating'] >= 4:
                    for feature in feature_preferences.keys():
                        feature_preferences[feature].append(song[feature])

        genre_preferences = {
            genre: sum(ratings)/len(ratings)
            for genre, ratings in avg_rating_by_genre.items()
            if ratings
        }

        for feature in feature_preferences:
            if feature_preferences[feature]:
                feature_preferences[feature] = sum(feature_preferences[feature]) / len(feature_preferences[feature])
            else:
                feature_preferences[feature] = 0.5

        return {
            'agent_id': agent.agent_id,
            'ratings': ratings,
            'genre_preferences': genre_preferences,
            'feature_preferences': feature_preferences,
            'onboarding_timestamp': datetime.now()
        }

    def get_agent_history(self, agent_id: str) -> Dict:
        return self.onboarding_histories.get(agent_id)
