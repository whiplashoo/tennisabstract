from flask import Flask, render_template, request, jsonify
import pandas as pd
from mod import (
    get_player_matches, 
    calculate_yearly_stats,
    calculate_career_stats, 
    calculate_recent_form,
    format_h2h_matches,
    career
)
import json
from datetime import datetime

app = Flask(__name__)

def dataframe_to_dict(df):
    """Convert DataFrame to dictionary format for easier frontend handling"""
    if df.index.name:
        df = df.reset_index()
    return df.to_dict('records')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['GET', 'POST'])
def compare_players():
    if request.method == 'POST':
        player1 = request.form.get('player1', '').strip()
        player2 = request.form.get('player2', '').strip()
        year = int(request.form.get('year', datetime.now().year))
        
        if not player1 or not player2:
            return jsonify({'error': 'Please enter both player names'}), 400
        
        # Get matches for both players
        p1_matches = get_player_matches(player1)
        p2_matches = get_player_matches(player2)
        
        if p1_matches is None or p2_matches is None:
            return jsonify({'error': 'One or both players not found'}), 404
        
        # Calculate stats for all surfaces
        all_surfaces_stats = {}
        surfaces = ['All', 'Hard', 'Clay', 'Grass', 'Carpet']
        
        for surface in surfaces:
            surface_filter = None if surface == 'All' else surface
            try:
                p1_stats = calculate_yearly_stats(p1_matches, surface_filter)
                p2_stats = calculate_yearly_stats(p2_matches, surface_filter)
                
                if year in p1_stats.index and year in p2_stats.index:
                    p1_data = p1_stats.loc[year]
                    p2_data = p2_stats.loc[year]
                    
                    # Convert to structured data
                    stats_data = []
                    for stat in p1_data.index:
                        if stat != 'surface':  # Skip surface indicator
                            stats_data.append({
                                'stat': stat,
                                'player1_value': p1_data[stat],
                                'player2_value': p2_data[stat],
                                'player1_name': player1,
                                'player2_name': player2
                            })
                    
                    all_surfaces_stats[surface] = stats_data
            except:
                continue
        
        if not all_surfaces_stats:
            return jsonify({'error': 'No data available for the selected year'}), 404
        
        return jsonify({
            'all_surfaces_stats': all_surfaces_stats,
            'player1': player1,
            'player2': player2,
            'year': year
        })
    
    return render_template('compare.html', current_year=datetime.now().year)

@app.route('/h2h', methods=['GET', 'POST'])
def head_to_head():
    if request.method == 'POST':
        player1 = request.form.get('player1', '').strip()
        player2 = request.form.get('player2', '').strip()
        
        if not player1 or not player2:
            return jsonify({'error': 'Please enter both player names'}), 400
        
        matches_df = get_player_matches(player1)
        if matches_df is None:
            return jsonify({'error': f'Player {player1} not found'}), 404
        
        h2h_data = format_h2h_matches(matches_df, player1, player2)
        
        if h2h_data.empty:
            return jsonify({'error': 'No head-to-head matches found between these players'}), 404
        
        # Calculate H2H summary
        p1_wins = len(h2h_data[h2h_data.winner_name == player1])
        p2_wins = len(h2h_data[h2h_data.winner_name == player2])
        
        # Get surface breakdown
        surface_stats = h2h_data.groupby('surface')['winner_name'].value_counts().unstack(fill_value=0)
        surface_breakdown = {}
        for surface in surface_stats.index:
            surface_breakdown[surface] = {
                player1: int(surface_stats.loc[surface].get(player1, 0)),
                player2: int(surface_stats.loc[surface].get(player2, 0))
            }
        
        # Convert matches to list of dicts
        matches_list = h2h_data.to_dict('records')
        for match in matches_list:
            match['match_date'] = str(match['match_date'])
        
        return jsonify({
            'matches': matches_list,
            'summary': {
                'player1': player1,
                'player2': player2,
                'p1_wins': p1_wins,
                'p2_wins': p2_wins,
                'total_matches': len(h2h_data),
                'surface_breakdown': surface_breakdown
            }
        })
    
    return render_template('h2h.html')

@app.route('/career', methods=['GET', 'POST'])
def career_stats():
    if request.method == 'POST':
        player_name = request.form.get('player', '').strip()
        
        if not player_name:
            return jsonify({'error': 'Please enter a player name'}), 400
        
        # Get player matches
        matches = get_player_matches(player_name)
        if matches is None:
            return jsonify({'error': f'Player {player_name} not found'}), 404
        
        career_data = career(player_name)
        
        if career_data is None:
            return jsonify({'error': f'No data available for {player_name}'}), 404
        
        # Get recent form
        recent_form = calculate_recent_form(matches)
        
        # Get surface breakdown with yearly details
        surface_breakdown = {}
        for surface in ['Hard', 'Clay', 'Grass', 'Carpet']:
            surface_stats = calculate_yearly_stats(matches, surface)
            if not surface_stats.empty:
                # Convert to dict with years as keys
                surface_yearly = {}
                for year in surface_stats.index:
                    year_data = surface_stats.loc[year].to_dict()
                    year_data['year'] = int(year)
                    surface_yearly[str(year)] = year_data
                
                # Calculate surface career totals
                surface_matches = matches[matches['surf'] == surface]
                surface_career = calculate_career_stats(surface_matches)
                if not surface_career.empty:
                    surface_breakdown[surface] = {
                        'yearly': surface_yearly,
                        'career': surface_career.iloc[0].to_dict()
                    }
        
        # Separate yearly stats from career summary
        yearly_stats = career_data[career_data.index != 'career'].copy()
        career_summary = career_data[career_data.index == 'career'].iloc[0].to_dict()
        
        # Convert yearly stats to list of dicts and sort by year descending
        yearly_data = []
        for year, row in yearly_stats.iterrows():
            year_dict = row.to_dict()
            year_dict['year'] = int(year)
            yearly_data.append(year_dict)
        
        # Sort by year in descending order
        yearly_data.sort(key=lambda x: x['year'], reverse=True)
        
        return jsonify({
            'yearly_stats': yearly_data,
            'career_summary': career_summary,
            'recent_form': recent_form,
            'surface_breakdown': surface_breakdown,
            'player': player_name
        })
    
    return render_template('career.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
