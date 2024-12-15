import os
from datetime import datetime
from typing import Dict, TextIO
import pandas as pd

from agents import (
    MusicAgent,
    create_pop_agent,
    create_classical_agent,
    create_genre_focused_agent
)
from dataloader import SpotifyDataLoader
from onboarding import OnboardingManager
from recommender import RecommenderManager

def create_test_agents():
    return [
        create_pop_agent("pop_enthusiast"),
        create_classical_agent("classical_lover"),
        create_genre_focused_agent("rock_indie_mix", "rock", ["indie", "pop"]),
        create_genre_focused_agent("jazz_electronic_mix", "jazz", ["electronic", "ambient"]),
        create_genre_focused_agent("hiphop_rnb_mix", "hip-hop", ["r-b", "pop"])
    ]

def setup_results_directory() -> Dict[str, TextIO]:
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return {
        'onboarding': open(f'results/onboarding_{timestamp}.txt', 'w'),
        'profiles': open(f'results/onboarding_profiles_{timestamp}.txt', 'w'),
        'recommendations': open(f'results/recommender_results_{timestamp}.txt', 'w')
    }

def write_onboarding_results(agent: MusicAgent, history: Dict, file: TextIO):
    file.write(f"\n{'='*80}\n")
    file.write(f"Onboarding Results for {agent.agent_id}\n")
    file.write(f"{'='*80}\n")

    total_ratings = len(history['ratings'])
    if total_ratings == 0:
        return

    valid_ratings = [interaction['rating'] for interaction in history['ratings'] if interaction['rating'] is not None]
    total_valid_ratings = len(valid_ratings)

    skips = sum(1 for interaction in history['ratings'] if interaction['skipped'])
    skip_frequency = skips / total_ratings if total_ratings else 0

    total_score = sum(valid_ratings)
    average_rating = total_score / total_valid_ratings if total_valid_ratings else 0

    file.write(f"Skip Frequency: {skip_frequency:.2f}\n")
    file.write(f"Average Rating: {average_rating:.2f}\n")

    file.write("\nInitial Song Ratings:\n")
    file.write("-" * 40 + "\n")
    for interaction in history['ratings']:
        song = loader.get_song_by_id(interaction['song_id'])
        if song:
            file.write(f"Song: {song['name']} by {song['artist']}\n")
            file.write(f"Genre: {song['genre']}\n")
            rating = interaction['rating'] if interaction['rating'] is not None else 'N/A'
            file.write(f"Rating: {rating}\n")
            file.write(f"Skipped: {interaction['skipped']}\n\n")

def write_user_profile(agent: MusicAgent, history: Dict, file: TextIO):
    file.write(f"\n{'='*80}\n")
    file.write(f"User Profile for {agent.agent_id}\n")
    file.write(f"{'='*80}\n")

    file.write("\nGenre Preferences:\n")
    file.write("-" * 40 + "\n")
    for genre, score in sorted(history['genre_preferences'].items(),
                             key=lambda x: x[1], reverse=True):
        file.write(f"{genre}: {score:.2f}\n")

    file.write("\nAudio Feature Preferences:\n")
    file.write("-" * 40 + "\n")
    for feature, value in history['feature_preferences'].items():
        file.write(f"{feature}: {value:.2f}\n")

def write_recommendation_results(
    agent: MusicAgent,
    recommendations: list,
    playlist: Dict,
    file: TextIO
):
    file.write(f"\n{'='*80}\n")
    file.write(f"Recommendations for {agent.agent_id}\n")
    file.write(f"{'='*80}\n")

    file.write("\nFeature Preferences Summary:\n")
    file.write("-" * 40 + "\n")

    for feature_pref in agent.feature_preferences:
        min_val, max_val = feature_pref.preferred_range
        file.write(f"{feature_pref.feature_name}: {min_val:.2f} - {max_val:.2f} (Weight: {feature_pref.weight:.2f})\n")

    file.write("\nRecommended Songs:\n")
    file.write("-" * 40 + "\n")

    for i, rec in enumerate(recommendations, 1):
        rating = agent.generate_rating(rec)

        file.write(f"\n{i}. {rec['name']} by {rec['artist']}\n")
        file.write(f"Genre: {rec['genre']}\n")
        file.write(f"Rating given: {rating}\n")
        file.write("Audio Features:\n")
        for feature in ['danceability', 'energy', 'acousticness', 'valence', 'tempo']:
            file.write(f"- {feature}: {rec[feature]:.2f}\n")
        file.write("\n")

    if playlist:
        file.write("\nCreated Playlist:\n")
        file.write("-" * 40 + "\n")
        file.write(f"Name: {playlist.name}\n")
        file.write(f"Description: {playlist.description}\n")
        file.write(f"Genre Focus: {playlist.genre_focus}\n")
        file.write(f"Number of Songs: {len(playlist.songs)}\n")

def main():
    print("Starting Music Recommendation System Simulation...")

    files = setup_results_directory()
    global loader
    loader = SpotifyDataLoader('data/spotify_songs.csv')
    loader.load_data()

    print("Initializing agents...")
    agents = create_test_agents()

    print("Starting onboarding phase...")
    onboarder = OnboardingManager(loader)
    onboarding_histories = {}

    for agent in agents:
        print(f"Onboarding {agent.agent_id}...")
        history = onboarder.onboard_agent(agent)
        onboarding_histories[agent.agent_id] = history

        write_onboarding_results(agent, history, files['onboarding'])
        write_user_profile(agent, history, files['profiles'])

    # Recommendation Phase
    print("\nStarting recommendation phase...")
    recommender = RecommenderManager(loader, onboarding_histories)

    for agent in agents:
        print(f"Generating recommendations for {agent.agent_id}...")
        recommendations = recommender.get_recommendations(agent, n_recommendations=5)

        # Create playlist from highly rated recommendations
        playlist_songs = []
        for rec in recommendations:
            rating = agent.generate_rating(rec)
            if rating and rating >= 4:
                playlist_songs.append(rec)

        playlist = None
        if playlist_songs:
            playlist = agent.create_playlist(playlist_songs)
            print(f"Created playlist for {agent.agent_id} with {len(playlist_songs)} songs")

        write_recommendation_results(
            agent,
            recommendations,
            playlist,
            files['recommendations']
        )

    # Cleanup
    for f in files.values():
        f.close()

    print("\nSimulation complete! Results have been written to the 'results' directory.")

if __name__ == "__main__":
    main()
