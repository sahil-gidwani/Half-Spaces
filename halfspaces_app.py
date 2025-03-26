import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import io
import base64
import fsspec
import os
import pyarrow.parquet as pq

# Caching the data loading functions
@st.cache_data(ttl=1800, max_entries=2)
def load_data(data_path: str, columns=None):
    if data_path.startswith("http"):  # Hugging Face Parquet file
        with fsspec.open(data_path) as file:
            df = pd.read_parquet(file, engine="pyarrow", columns=columns)
    elif data_path.endswith(".csv"):  # Local CSV file
        df = pd.read_csv(data_path)
    elif data_path.endswith(".parquet"):  # Local Parquet file
        df = pd.read_parquet(data_path, engine="pyarrow", columns=columns)
    else:
        raise ValueError("Unsupported file format! Only CSV and Parquet are allowed.")

    return df
    

@st.cache_data
def add_carries(_game_df):
    game_df = _game_df.copy()
    min_dribble_length: float = 0.0
    max_dribble_length: float = 100.0
    max_dribble_duration: float = 20.0
    game_df['time_seconds'] = game_df['minute']*60+game_df['second']
    next_actions = game_df.shift(-1)
    same_game = game_df.gameId == next_actions.gameId
    same_team = game_df.teamId == next_actions.teamId
    dx = game_df.endX - next_actions.x
    dy = game_df.endY - next_actions.y
    far_enough = dy**2 + dy**2 >= min_dribble_length**2
    not_too_far = dx**2 + dy**2 <= max_dribble_length**2
    dt = next_actions.time_seconds - game_df.time_seconds
    same_phase = dt < max_dribble_duration
    same_period = game_df.period == next_actions.period
    dribble_idx = same_team & far_enough & not_too_far & same_phase & same_period & same_game
    dribbles = pd.DataFrame()
    prev = game_df[dribble_idx]
    nex = next_actions[dribble_idx]
    dribbles['gameId'] = nex.gameId
    dribbles['period'] = nex.period
    dribbles['expandedMinute'] = nex.expandedMinute
    dribbles['passKey'] = [True for _ in range(len(dribbles))]
    dribbles['assist'] = [True for _ in range(len(dribbles))]
    dribbles['isTouch'] = [True for _ in range(len(dribbles))]
    
    for cols in ['playerId', 'team', 'player']:
        dribbles[cols] = nex[cols]
    
    dribbles['time_seconds'] = (prev.time_seconds + nex.time_seconds) / 2
    dribbles['teamId'] = nex.teamId
    dribbles['x'] = prev.endX
    dribbles['y'] = prev.endY
    dribbles['endX'] = nex.x
    dribbles['endY'] = nex.y
    dribbles['type'] = ['Carry' for _ in range(len(dribbles))]
    dribbles['outcomeType'] = ['Successful' for _ in range(len(dribbles))]
    
    game_df = pd.concat([game_df, dribbles], ignore_index=True, sort=False)
    game_df = game_df.sort_values(['gameId', 'period']).reset_index(drop=True)
    game_df['action_id'] = range(len(game_df))
    return game_df

@st.cache_data
def prepare_data(data):
    # Adjust coordinates
    data = data.copy()
    data['x'] = data['x']*1.2
    data['y'] = data['y']*.8
    data['endX'] = data['endX']*1.2
    data['endY'] = data['endY']*.8
    
    # Extract carries and passes
    data_carries = data[data['type'] == 'Carry']
    data_passes = data[(data['type'] == 'Pass') & (data['outcomeType'] == 'Successful') & (data['x'] <= 119.2)]
    
    # Half-space definitions
    left_halfspace_x_min, left_halfspace_x_max = 60, 102
    left_halfspace_y_min, left_halfspace_y_max = 50, 62
    right_halfspace_x_min, right_halfspace_x_max = 60, 102
    right_halfspace_y_min, right_halfspace_y_max = 18, 30
    
    # Add half-space columns
    data_passes['in_rhs'] = (data_passes['x'].between(right_halfspace_x_min, right_halfspace_x_max) & 
                             data_passes['y'].between(right_halfspace_y_min, right_halfspace_y_max))
    data_passes['in_lhs'] = (data_passes['x'].between(left_halfspace_x_min, left_halfspace_x_max) & 
                             data_passes['y'].between(left_halfspace_y_min, left_halfspace_y_max))
    data_passes['into_rhs'] = (data_passes['endX'].between(right_halfspace_x_min, right_halfspace_x_max) & 
                               data_passes['endY'].between(right_halfspace_y_min, right_halfspace_y_max) & 
                               (data_passes['in_rhs'] == False))
    data_passes['into_lhs'] = (data_passes['endX'].between(left_halfspace_x_min, left_halfspace_x_max) & 
                               data_passes['endY'].between(left_halfspace_y_min, left_halfspace_y_max) & 
                               (data_passes['in_lhs'] == False))
    
    data_carries['in_rhs'] = (data_carries['x'].between(right_halfspace_x_min, right_halfspace_x_max) & 
                              data_carries['y'].between(right_halfspace_y_min, right_halfspace_y_max))
    data_carries['in_lhs'] = (data_carries['x'].between(left_halfspace_x_min, left_halfspace_x_max) & 
                              data_carries['y'].between(left_halfspace_y_min, left_halfspace_y_max))
    data_carries['into_rhs'] = (data_carries['endX'].between(right_halfspace_x_min, right_halfspace_x_max) & 
                                data_carries['endY'].between(right_halfspace_y_min, right_halfspace_y_max) & 
                                (data_carries['in_rhs'] == False))
    data_carries['into_lhs'] = (data_carries['endX'].between(left_halfspace_x_min, left_halfspace_x_max) & 
                                data_carries['endY'].between(left_halfspace_y_min, left_halfspace_y_max) & 
                                (data_carries['in_lhs'] == False))
    
    return data_passes, data_carries

@st.cache_data
def calculate_progressive_actions(df):
    df_prog = df.copy()
    df_prog['beginning'] = np.sqrt(np.square(120 - df_prog['x']) + np.square(40 - df_prog['y']))
    df_prog['end'] = np.sqrt(np.square(120 - df_prog['endX']) + np.square(40 - df_prog['endY']))
    df_prog['progressive'] = (df_prog['end'] / df_prog['beginning']) < 0.75
    return df_prog[df_prog['progressive']]

@st.cache_data
def process_halfspace_data(data_passes, data_carries, mins_data, league_data):
    # Progressive Actions in Half-Spaces
    prog_rhs_passes = calculate_progressive_actions(data_passes[data_passes['in_rhs']])
    prog_lhs_passes = calculate_progressive_actions(data_passes[data_passes['in_lhs']])
    prog_rhs_carries = calculate_progressive_actions(data_carries[data_carries['in_rhs']])
    prog_lhs_carries = calculate_progressive_actions(data_carries[data_carries['in_lhs']])
    
    # Group and aggregate
    prog_rhs_passes_grouped = prog_rhs_passes.groupby(['playerId', 'player', 'team']).size().reset_index(name='prog_rhs_passes')
    prog_lhs_passes_grouped = prog_lhs_passes.groupby(['playerId', 'player', 'team']).size().reset_index(name='prog_lhs_passes')
    prog_rhs_carries_grouped = prog_rhs_carries.groupby(['playerId', 'player', 'team']).size().reset_index(name='prog_rhs_carries')
    prog_lhs_carries_grouped = prog_lhs_carries.groupby(['playerId', 'player', 'team']).size().reset_index(name='prog_lhs_carries')
    
    # Merge data
    prog_result_df_rhs = pd.merge(prog_rhs_passes_grouped, prog_rhs_carries_grouped, on=['playerId', 'player', 'team'], how='outer').fillna(0)
    prog_result_df_rhs['prog_rhs_actions'] = prog_result_df_rhs['prog_rhs_passes'] + prog_result_df_rhs['prog_rhs_carries']
    
    prog_result_df_lhs = pd.merge(prog_lhs_passes_grouped, prog_lhs_carries_grouped, on=['playerId', 'player', 'team'], how='outer').fillna(0)
    prog_result_df_lhs['prog_lhs_actions'] = prog_result_df_lhs['prog_lhs_passes'] + prog_result_df_lhs['prog_lhs_carries']
    
    # Combine data
    combined_prog_df = pd.merge(prog_result_df_rhs, prog_result_df_lhs, on=['playerId', 'player', 'team'], how='outer').fillna(0)
    combined_prog_df['prog_HS_actions'] = combined_prog_df['prog_rhs_actions'] + combined_prog_df['prog_lhs_actions']
    
    # Add minutes played
    mins_data['90s'] = mins_data['Mins'] / 90
    combined_prog_df = pd.merge(combined_prog_df, mins_data, on=['player', 'team'], how='left')
    
    # Add league information from league_data
    combined_prog_df = pd.merge(combined_prog_df, 
                                league_data[['player', 'team', 'league', 'season', 'position']].drop_duplicates(), 
                                on=['player', 'team'], 
                                how='left')
    
    # Calculate per 90 metrics
    combined_prog_df['prog_act_HS_p90'] = combined_prog_df['prog_HS_actions'] / combined_prog_df['90s']
    combined_prog_df['prog_rhs_act_p90'] = combined_prog_df['prog_rhs_actions'] / combined_prog_df['90s']
    combined_prog_df['prog_lhs_act_p90'] = combined_prog_df['prog_lhs_actions'] / combined_prog_df['90s']
    
    # Filter players
    combined_prog_df = combined_prog_df[
        (combined_prog_df['90s'] >= 15) & 
        (combined_prog_df['position'] != 'GK')
    ]
    
    # Drop duplicates
    combined_prog_df = combined_prog_df.drop_duplicates(subset=['player'])
    
    return combined_prog_df, prog_rhs_passes, prog_lhs_passes, prog_rhs_carries, prog_lhs_carries

@st.cache_data
def plot_player_halfspace_actions(player_data, player_id, prog_rhs_passes, prog_lhs_passes, 
                                   prog_rhs_carries, prog_lhs_carries, action_type):
    fig, ax = plt.subplots(figsize=(15, 10), facecolor='#1e1e1e')
    
    # Pitch setup
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#1e1e1e', line_color='#FFFFFF')
    pitch.draw(ax=ax)

    ax.text(60, 78, '@pranav_m28', fontsize=17, color='white', alpha=0.7, ha='center', va='center')
    
    # Filter player-specific data based on action type
    if action_type == "Right Half-Space Actions":
        player_prog_passes = prog_rhs_passes[prog_rhs_passes['playerId'] == player_id]
        player_prog_carries = prog_rhs_carries[prog_rhs_carries['playerId'] == player_id]
    elif action_type == "Left Half-Space Actions":
        player_prog_passes = prog_lhs_passes[prog_lhs_passes['playerId'] == player_id]
        player_prog_carries = prog_lhs_carries[prog_lhs_carries['playerId'] == player_id]
    else:  # All Half-Space Actions
        player_prog_rhs_passes = prog_rhs_passes[prog_rhs_passes['playerId'] == player_id]
        player_prog_rhs_carries = prog_rhs_carries[prog_rhs_carries['playerId'] == player_id]
        player_prog_lhs_passes = prog_lhs_passes[prog_lhs_passes['playerId'] == player_id]
        player_prog_lhs_carries = prog_lhs_carries[prog_lhs_carries['playerId'] == player_id]
    
    if action_type in ["Right Half-Space Actions", "Left Half-Space Actions"]:
        # Plot Progressive Passes
        pitch.lines(player_prog_passes.x, player_prog_passes.y, 
                    player_prog_passes.endX, player_prog_passes.endY,
                    lw=3, transparent=True, comet=True, color='#24a8ff', ax=ax)
        pitch.scatter(player_prog_passes.endX, player_prog_passes.endY, 
                      s=40, marker='o', edgecolors='none', c='#24a8ff', ax=ax, alpha=1)
        
        # Plot Progressive Carries
        pitch.lines(player_prog_carries.x, player_prog_carries.y, 
                    player_prog_carries.endX, player_prog_carries.endY, 
                    ls='dashed', lw=2, transparent=False, comet=False, color='#FF5959', ax=ax)
        pitch.scatter(player_prog_carries.endX, player_prog_carries.endY, 
                      s=40, marker='o', edgecolors='none', c='#FF5959', ax=ax, alpha=1)
    else:
        # Plot RHS Progressive Passes
        pitch.lines(player_prog_rhs_passes.x, player_prog_rhs_passes.y, 
                    player_prog_rhs_passes.endX, player_prog_rhs_passes.endY,
                    lw=3, transparent=True, comet=True, color='#24a8ff', ax=ax)
        pitch.scatter(player_prog_rhs_passes.endX, player_prog_rhs_passes.endY, 
                      s=40, marker='o', edgecolors='none', c='#24a8ff', ax=ax, alpha=1)
        
        # Plot RHS Progressive Carries
        pitch.lines(player_prog_rhs_carries.x, player_prog_rhs_carries.y, 
                    player_prog_rhs_carries.endX, player_prog_rhs_carries.endY, 
                    ls='dashed', lw=2, transparent=False, comet=False, color='#FF5959', ax=ax)
        pitch.scatter(player_prog_rhs_carries.endX, player_prog_rhs_carries.endY, 
                      s=40, marker='o', edgecolors='none', c='#FF5959', ax=ax, alpha=1)
        
        # Plot LHS Progressive Passes
        pitch.lines(player_prog_lhs_passes.x, player_prog_lhs_passes.y, 
                    player_prog_lhs_passes.endX, player_prog_lhs_passes.endY,
                    lw=3, transparent=True, comet=True, color='#24a8ff', ax=ax)
        pitch.scatter(player_prog_lhs_passes.endX, player_prog_lhs_passes.endY, 
                      s=40, marker='o', edgecolors='none', c='#24a8ff', ax=ax, alpha=1)
        
        # Plot LHS Progressive Carries
        pitch.lines(player_prog_lhs_carries.x, player_prog_lhs_carries.y, 
                    player_prog_lhs_carries.endX, player_prog_lhs_carries.endY, 
                    ls='dashed', lw=2, transparent=False, comet=False, color='#FF5959', ax=ax)
        pitch.scatter(player_prog_lhs_carries.endX, player_prog_lhs_carries.endY, 
                      s=40, marker='o', edgecolors='none', c='#FF5959', ax=ax, alpha=1)
    
    ax.invert_yaxis()
    
    # Title and annotations
    if action_type == "Right Half-Space Actions":
        title = f'{player_data["player"]} - Right Half-Space Progressive Actions\nRight Half-Space Actions p90: {player_data["prog_rhs_act_p90"]:.2f}'
    elif action_type == "Left Half-Space Actions":
        title = f'{player_data["player"]} - Left Half-Space Progressive Actions\nLeft Half-Space Actions p90: {player_data["prog_lhs_act_p90"]:.2f}'
    else:
        title = f'{player_data["player"]} - Half-Space Progressive Actions\nTotal Half-Space Actions p90: {player_data["prog_act_HS_p90"]:.2f}'
    
    ax.set_title(title, fontsize=28, color='white', fontweight='bold')
    
    # Convert plot to base64 for Streamlit
    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png', facecolor='#1e1e1e', edgecolor='none', dpi=300)
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    
    return plot_data

def main():
    # Set page configuration
    st.set_page_config(page_title="Half-Spaces Progressive Actions", layout="wide")
    
    # Load the main dataset and minutes data using cached function
    hf_url = "https://huggingface.co/datasets/pranavm28/Top_5_Leagues_23_24/resolve/main/Top_5_Leagues_23_24.parquet"
    data = load_data(hf_url, columns = [
        "league", "season", "gameId", "period", "minute", "second", "expandedMinute",  
        "type", "outcomeType", "teamId", "team", "playerId", "player",  
        "x", "y", "endX", "endY"
    ])
    
    # Load minutes data
    mins_data = load_data("T5 Leagues Mins 23-24.csv")
    
    # Streamlit App
    st.title("Top 5 Leagues Half-Spaces Progressive Actions")
    
    # Sidebar for filtering
    st.sidebar.header("Filters")
    
    # Season and League Selection
    seasons = sorted(data['season'].unique())
    leagues = sorted(data['league'].unique())
    
    selected_season = st.sidebar.selectbox("Select Season", seasons)
    selected_league = st.sidebar.selectbox("Select League", leagues)
    
    # Filter data based on season and league selection
    filtered_data = data[
        (data['season'] == selected_season) & 
        (data['league'] == selected_league)
    ]
    
    # 90s Slider
    min_90s = float(mins_data['Mins'].min() / 90)
    max_90s = float(mins_data['Mins'].max() / 90)
    default_min_90s = 15.0  # Default minimum 90s
    
    selected_90s = st.sidebar.slider(
        "Minimum 90s Played", 
        min_value=min_90s, 
        max_value=max_90s, 
        value=default_min_90s, 
        step=0.5
    )
    
    # Add carries to the filtered data
    filtered_data = add_carries(filtered_data)
    
    # Prepare data for the selected season and league
    data_passes, data_carries = prepare_data(filtered_data)
    
    # Merge data from minutes file based on shared columns (player and team)
    filtered_mins_data = mins_data.copy()
    
    # Prepare league data for merging
    league_data = filtered_data[['player', 'team', 'league', 'season']].drop_duplicates()
    
    # Process halfspace data
    combined_prog_df, prog_rhs_passes, prog_lhs_passes, prog_rhs_carries, prog_lhs_carries = process_halfspace_data(
        data_passes, data_carries, filtered_mins_data, league_data
    )
    
    # Filter players by selected 90s and league
    combined_prog_df = combined_prog_df[
        (combined_prog_df['90s'] >= selected_90s) & 
        (combined_prog_df['league'] == selected_league) &
        (combined_prog_df['season'] == selected_season)
    ]
    
    # Team selection
    # Filter available teams based on the selected season, league, and 90s played
    available_teams = sorted(combined_prog_df['team'].unique())
    
    # Use multiselect with the filtered teams
    selected_teams = st.sidebar.multiselect("Select Teams", available_teams, default=available_teams)
    
    # Filtered DataFrame
    filtered_df = combined_prog_df[combined_prog_df['team'].isin(selected_teams)]
    
    # Half-Space Action Type Selection
    action_type = st.sidebar.radio("Action Type", 
                                   ["All Half-Space Actions", 
                                    "Right Half-Space Actions", 
                                    "Left Half-Space Actions"])
    
    # Sorting and Display
    if action_type == "Right Half-Space Actions":
        sorted_df = filtered_df.sort_values("prog_rhs_act_p90", ascending=False)
    elif action_type == "Left Half-Space Actions":
        sorted_df = filtered_df.sort_values("prog_lhs_act_p90", ascending=False)
    else:
        sorted_df = filtered_df.sort_values("prog_act_HS_p90", ascending=False)
    
    st.write("### Half-Space Progressive Actions per 90")
    
    # Display top players table
    display_columns = ['player', 'team', 'prog_act_HS_p90', 'prog_rhs_act_p90', 'prog_lhs_act_p90', '90s']
    st.dataframe(sorted_df[display_columns], use_container_width=True)
    
    # Player Selection for Visualization
    st.write("### Player Half-Space Actions Visualization")
    selected_player = st.selectbox("Select a Player", sorted_df['player'])
    
    # Get selected player data
    player_data = sorted_df[sorted_df['player'] == selected_player]

    if player_data.empty:
        st.error("No data found for the selected player.")
    
        return  # Exit the function to prevent further errors

    player_data = player_data.iloc[0]
    player_id = player_data['playerId']

    
    # Plot player's half-space actions
    plot_data = plot_player_halfspace_actions(
        player_data, player_id, 
        prog_rhs_passes, prog_lhs_passes, 
        prog_rhs_carries, prog_lhs_carries,
        action_type
    )

    with st.sidebar:
        st.markdown("### Connect with me")
        st.markdown("- 🐦 [Twitter](https://twitter.com/pranav_m28)")
        st.markdown("- 🔗 [GitHub](https://github.com/pranavm28)")
        st.markdown("-Contibute: [BuyMeACoffee](https://buymeacoffee.com/pranav_m28)")
    
    # Display plot
    st.image(f"data:image/png;base64,{plot_data}")

if __name__ == "__main__":
    main()