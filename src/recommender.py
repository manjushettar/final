from typing import List, Dict, Optional
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity

from agents import MusicAgent, create_pop_agent, create_classical_agent

from dataloader import SpotifyDataLoader
from onboarding import OnboardingManager

class RecommenderManager:
    def __init__(self, data_loader: SpotifyDataLoader, onboarding_histories: Dict):
        self.data_loader = data_loader
        self.onboarding_histories = onboarding_histories
        self.interaction_histories = {}

    def add_interaction(self, agent_id: str, song_id: str, rating: Optional[int], timestamp: datetime):
        if agent_id not in self.interaction_histories:
            self.interaction_histories[agent_id] = []

        self.interaction_histories[agent_id].append({
            'song_id': song_id,
            'rating': rating,
            'timestamp': timestamp
        })

    def get_recommendations(self, agent: MusicAgent, n_recommendations: int = 5) -> List[Dict]:
        content_recs = self._get_content_based_recommendations(agent, n_recommendations)
        collab_recs = self._get_collaborative_recommendations(agent, n_recommendations)

        combined_recs = self._combine_recommendations(
            content_recs,
            collab_recs,
            agent
        )

        return combined_recs[:n_recommendations]

    def _get_content_based_recommendations(self, agent: MusicAgent, n: int) -> List[Dict]:
        if agent.agent_id not in self.onboarding_histories:
            return self.data_loader.get_random_songs(n)

        history = self.onboarding_histories[agent.agent_id]
        feature_preferences = history['feature_preferences']

        all_songs = self.data_loader.data
        feature_scores = []

        for _, song in all_songs.iterrows():
            feature_similarity = sum(
                abs(1 - abs(song[feature] - pref_value))
                for feature, pref_value in feature_preferences.items()
            ) / len(feature_preferences)

            genre_bonus = 0
            if song['playlist_genre'] in history['genre_preferences']:
                genre_bonus = history['genre_preferences'][song['playlist_genre']] / 5

            feature_scores.append(feature_similarity + genre_bonus)

        feature_scores = np.array(feature_scores)
        top_indices = np.argsort(feature_scores)[-n*2:][::-1]
        recommended_songs = []

        for idx in top_indices:
            song_data = self.data_loader._format_song_dict(all_songs.iloc[idx])
            if not self._is_recently_recommended(agent.agent_id, song_data['id']):
                recommended_songs.append(song_data)

        return recommended_songs[:n]

    def _get_collaborative_recommendations(self, agent: MusicAgent, n: int) -> List[Dict]:
        if len(self.interaction_histories) < 2:
            return []

        agent_ratings = self._get_agent_ratings(agent.agent_id)
        if not agent_ratings:
            return []

        similar_agents = self._find_similar_agents(agent.agent_id)
        recommendations = {}

        for similar_agent, similarity in similar_agents:
            similar_ratings = self._get_agent_ratings(similar_agent)

            for song_id, rating in similar_ratings.items():
                if song_id not in agent_ratings:
                    if song_id not in recommendations:
                        recommendations[song_id] = 0
                    recommendations[song_id] += rating * similarity

        sorted_recs = sorted(
            recommendations.items(),
            key=lambda x: x[1],
            reverse=True
        )

        recommended_songs = []
        for song_id, _ in sorted_recs[:n]:
            song = self.data_loader.get_song_by_id(song_id)
            if song:
                recommended_songs.append(song)

        return recommended_songs

    def _combine_recommendations(
        self,
        content_recs: List[Dict],
        collab_recs: List[Dict],
        agent: MusicAgent
    ) -> List[Dict]:
        all_recs = {}

        content_weight = 0.7
        collab_weight = 0.3

        for i, rec in enumerate(content_recs):
            score = content_weight * (1 - i/len(content_recs))
            all_recs[rec['id']] = {
                'song': rec,
                'score': score
            }

        for i, rec in enumerate(collab_recs):
            score = collab_weight * (1 - i/len(collab_recs))
            if rec['id'] in all_recs:
                all_recs[rec['id']]['score'] += score
            else:
                all_recs[rec['id']] = {
                    'song': rec,
                    'score': score
                }

        sorted_recs = sorted(
            all_recs.values(),
            key=lambda x: x['score'],
            reverse=True
        )

        return [rec['song'] for rec in sorted_recs]

    def _get_agent_ratings(self, agent_id: str) -> Dict[str, float]:
        ratings = {}

        if agent_id in self.onboarding_histories:
            for interaction in self.onboarding_histories[agent_id]['ratings']:
                if interaction['rating']:
                    ratings[interaction['song_id']] = interaction['rating']

        if agent_id in self.interaction_histories:
            for interaction in self.interaction_histories[agent_id]:
                if interaction['rating']:
                    ratings[interaction['song_id']] = interaction['rating']

        return ratings

    def _find_similar_agents(self, agent_id: str) -> List[tuple]:
        if agent_id not in self.onboarding_histories:
            return []

        agent_ratings = self._get_agent_ratings(agent_id)
        similarities = []

        for other_id in self.onboarding_histories.keys():
            if other_id != agent_id:
                other_ratings = self._get_agent_ratings(other_id)
                similarity = self._calculate_rating_similarity(
                    agent_ratings,
                    other_ratings
                )
                similarities.append((other_id, similarity))

        return sorted(similarities, key=lambda x: x[1], reverse=True)

    def _calculate_rating_similarity(
        self,
        ratings1: Dict[str, float],
        ratings2: Dict[str, float]
    ) -> float:
        common_songs = set(ratings1.keys()) & set(ratings2.keys())
        if not common_songs:
            return 0

        vector1 = [ratings1[song] for song in common_songs]
        vector2 = [ratings2[song] for song in common_songs]

        return float(cosine_similarity([vector1], [vector2])[0, 0])

    def _is_recently_recommended(
        self,
        agent_id: str,
        song_id: str,
        hours: int = 24
    ) -> bool:
        if agent_id not in self.interaction_histories:
            return False

        recent_cutoff = datetime.now() - timedelta(hours=hours)
        recent_interactions = [
            interaction for interaction in self.interaction_histories[agent_id]
            if interaction['timestamp'] > recent_cutoff
        ]

        return any(interaction['song_id'] == song_id
                  for interaction in recent_interactions)
