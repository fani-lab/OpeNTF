from basketball_reference_web_scraper import client
from basketball_reference_web_scraper.data import OutputType, Team

import re
import json
from collections import defaultdict

# Fetch play-by-play log
pbp_log = client.play_by_play(
    home_team=Team.PHOENIX_SUNS, 
    year=2018, month=11, day=8, 
    output_type=OutputType.JSON
    output_file_path="2018_11_08_PHO_PBP.json"
)

# Define regex patterns for player names and substitutions
player_name = r"([A-Z]\. [A-Z][a-z]+)"  # Add capturing group for player name
substitution = player_name + r" enters the game for " + player_name  # Use capturing groups

# Parse play-by-play log
pbp_log_json = json.loads(pbp_log)

# Track active players and substitution events
active_players = defaultdict(lambda: None)  # Tracks when a player is active
results_by_player = defaultdict(list)

# Process play-by-play events
for i, event in enumerate(pbp_log_json):
    match = re.match(substitution, event['description'])
    if match:
        entering_player = match.group(1)  # Player entering the game
        exiting_player = match.group(2)  # Player exiting the game

        # Calculate score change for the exiting player
        if active_players[exiting_player] is not None:
            start_index = active_players[exiting_player]
            start_score = pbp_log_json[start_index]['home_score'] - pbp_log_json[start_index]['away_score']
            end_score = event['home_score'] - event['away_score']
            score_change = end_score - start_score

            # Determine success or failure
            if score_change > 0:
                result = "Success"
            elif score_change < 0:
                result = "Failure"
            else:
                result = "Neutral"

            # Record the result for the exiting player
            results_by_player[exiting_player].append({
                "description": pbp_log_json[start_index]['description'],
                "start_score": start_score,
                "end_score": end_score,
                "score_change": score_change,
                "result": result
            })

        # Mark the entering player as active
        active_players[entering_player] = i

# Output results organized by player
for player, events in results_by_player.items():
    print(f"Player: {player}")
    for event in events:
        print(f"  - {event['description']}: {event['result']} (Score Change: {event['score_change']})")