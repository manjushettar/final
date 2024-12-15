import random
import pandas as pd
from typing import List, Dict
from datetime import datetime, timedelta
from dataloader import SpotifyDataLoader
from agents import (
    MusicAgent,
    create_pop_agent,
    create_classical_agent,
    create_genre_focused_agent
)

class SimulationRunner:
    def __init__(
        self,
        songs_data: List[Dict],
        n_pop_agents: int = 1,
        n_classical_agents: int = 1,
        n_general_agents: int = 1,
        simulation_days: int = 30
    ):
        self.songs_data = songs_data
        self.simulation_days = simulation_days
        self.start_date = datetime.now() - timedelta(days=simulation_days)
        self.agents = self._initialize_agents(
            n_pop_agents,
            n_classical_agents,
            n_general_agents
        )
        self.interaction_history = []

    def _initialize_agents(
        self,
        n_pop: int,
        n_classical: int,
        n_general: int
    ) -> List[MusicAgent]:
        agents = []


        for i in range(n_pop):
            agent = create_pop_agent(f"pop_agent_{i}")
            agents.append(agent)

        for i in range(n_classical):
            agent = create_classical_agent(f"classical_agent_{i}")
            agents.append(agent)

        genres = ["rock", "indie", "jazz", "electronic", "hip-hop"]
        for i in range(n_general):
            primary_genre = random.choice(genres)
            secondary_genres = random.sample([g for g in genres if g != primary_genre], 2)
            agent = create_genre_focused_agent(
                f"general_agent_{i}",
                primary_genre,
                secondary_genres
            )
            agents.append(agent)

        return agents

    def run_single_day(self, current_date: datetime) -> List[Dict]:
        '''
            this is essentially the most detailed simulation of a single day listening history i could think of
            returns the daily interactions made by the agent
        '''
        daily_interactions = []

        for agent in self.agents:
            active_hours = agent.active_hours
            current_hour = random.choice(active_hours)

            # for now, this is hardcoded behavior (songs per session)
            session_length = agent.behavior.avg_session_length
            songs_per_session = session_length // 3

            # each agent gets to interact with the song (no sense of recommendation yet)
            for _ in range(songs_per_session):
                song = random.choice(self.songs_data)

                if agent.should_listen_to_song(song):
                    interaction = {
                        'agent_id': agent.agent_id,
                        'song_id': song['id'],
                        'timestamp': current_date + timedelta(hours=current_hour),
                        'skipped': random.random() < agent.behavior.skip_probability,
                        'rating': agent.generate_rating(song),
                        'archetype': agent.archetype,
                        'type': 'song-interaction'
                    }
                    daily_interactions.append(interaction)

            # create playlist based on the likelihood of the agent
            if random.random() < agent.behavior.playlist_creation_frequency:
                playlist = agent.create_playlist(self.songs_data)
                if playlist:
                    daily_interactions.append({
                        'agent_id': agent.agent_id,
                        'type': 'playlist_creation',
                        'playlist_id': playlist.playlist_id,
                        'playlist_name': playlist.name,
                        'timestamp': current_date + timedelta(hours=current_hour),
                        'songs': playlist.songs,
                        'archetype': agent.archetype
                    })

        return daily_interactions

    def run_simulation(self) -> pd.DataFrame:
        '''
            runs the sim and dumps results into a dataframe
        '''
        all_interactions = []

        for day in range(self.simulation_days):
            current_date = self.start_date + timedelta(days=day)
            daily_interactions = self.run_single_day(current_date)
            all_interactions.extend(daily_interactions)

        df = pd.DataFrame(all_interactions)
        self.interaction_history = df
        return df

    def get_agent_statistics(self) -> Dict:
        '''
            returns some statistics about interaction history like average rating and skip rate
        '''
        if self.interaction_history is None or len(self.interaction_history) == 0:
            return {}

        stats = {
            'total_interactions': len(self.interaction_history),
            'interactions_by_archetype': self.interaction_history.groupby('archetype').size().to_dict(),
            'playlists_created': len(self.interaction_history[
                self.interaction_history['type'] == 'playlist_creation'
            ]),

            'average_rating': self.interaction_history['rating'].mean(),
            'skip_rate': (
                self.interaction_history['skipped'].sum() /
                len(self.interaction_history)
            )
        }
        return stats

    def get_agent_playlists(self) -> pd.DataFrame:
        """
            dumps all agent playlists
        """
        playlists = []

        for agent in self.agents:
            for playlist in agent.playlists:
                playlist_info = {
                    'agent_id': agent.agent_id,
                    'agent_archetype': agent.archetype,
                    'playlist_id': playlist.playlist_id,
                    'playlist_name': playlist.name,
                    'description': playlist.description,
                    'genre_focus': playlist.genre_focus,
                    'n_songs': len(playlist.songs),
                    'created_at': playlist.created_at
                }
                playlists.append(playlist_info)
        if not playlists:
                return pd.DataFrame(columns=[
                    'agent_id', 'agent_archetype', 'playlist_id', 'playlist_name',
                    'description', 'genre_focus', 'n_songs', 'created_at'
                ])
        return pd.DataFrame(playlists)
