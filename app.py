from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from mod import (
    get_player_matches, 
    calculate_yearly_stats,
    calculate_career_stats, 
    calculate_recent_form,
    format_h2h_matches,
    career,
    get_players,
    suggest_players
)
import json
from datetime import datetime, date
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

def to_py(val):
    """Convert numpy/pandas types and NaN/Timestamp to JSON-safe Python primitives."""
    if pd.isna(val):
        return None
    # Datetime-like
    if isinstance(val, (pd.Timestamp, datetime, date)):
        try:
            return str(val.date()) if hasattr(val, "date") else str(val)
        except Exception:
            return str(val)
    # Numpy to Python
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    return val

def series_to_py_dict(s):
    """Convert a pandas Series to a JSON-safe dict."""
    out = {}
    for k, v in s.to_dict().items():
        out[str(k)] = to_py(v)
    return out

def dataframe_to_dict(df):
    """Convert DataFrame to JSON-safe list of dicts."""
    if df is None or len(df) == 0:
        return []
    if df.index.name:
        df = df.reset_index()
    records = df.to_dict('records')
    return [{k: to_py(v) for k, v in rec.items()} for rec in records]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/players', methods=['GET'])
def players():
    try:
        all_players = get_players(tour=None)
        atp_players = all_players.get('ATP', [])
        wta_players = all_players.get('WTA', [])
        return render_template('players.html', atp_players=atp_players, wta_players=wta_players)
    except Exception as e:
        logging.exception(f"/players error: {e}")
        return render_template('players.html', atp_players=[], wta_players=[])

@app.route('/api/players', methods=['GET'])
def players_api():
    tour = request.args.get('tour')
    try:
        def transform(p, default_tour=None):
            return {
                'rank': to_py(p.get('rank')),
                'name': to_py(p.get('name')),
                'tour': p.get('tour', default_tour),
                'country': p.get('country'),
                'birthdate': p.get('birthdate'),
                'age': round(p.get('age'), 1) if p.get('age') is not None else None,  # Round to 1 decimal
            }
        
        if tour:
            tour = tour.upper()
            if tour not in ('ATP', 'WTA'):
                return jsonify({'error': "Invalid tour; must be 'ATP' or 'WTA'"}), 400
            lst = get_players(tour=tour)
            return jsonify({'players': [transform(p, default_tour=tour) for p in lst]})
        else:
            all_players = get_players(tour=None)
            return jsonify({
                'ATP': [transform(p, default_tour='ATP') for p in all_players.get('ATP', [])],
                'WTA': [transform(p, default_tour='WTA') for p in all_players.get('WTA', [])]
            })
    except Exception as e:
        logging.exception(f"/api/players error: {e}")
        return jsonify({'error': 'Failed to load players'}), 500

@app.route('/api/player-suggest', methods=['GET'])
def player_suggest():
    q = request.args.get('q', '').strip()
    if not q or len(q) < 2:
        return jsonify({'suggestions': []})
    try:
        try:
            limit = int(request.args.get('limit', 10))
        except Exception:
            limit = 10
        tour = request.args.get('tour')
        suggestions = suggest_players(q, limit=limit, tour=tour)
        return jsonify({'suggestions': [{'label': to_py(s.get('label')), 'value': to_py(s.get('value')), 'tour': s.get('tour')} for s in suggestions]})
    except Exception as e:
        logging.exception(f"/api/player-suggest error: {e}")
        return jsonify({'suggestions': []}), 500

@app.route('/compare', methods=['GET', 'POST'])
def compare_players():
    if request.method == 'POST':
        player1 = request.form.get('player1', '').strip()
        player2 = request.form.get('player2', '').strip()
        try:
            year = int(request.form.get('year', datetime.now().year))
        except Exception:
            year = datetime.now().year
        
        if not player1 or not player2:
            return jsonify({'error': 'Please enter both player names'}), 400
        
        # Get matches for both players
        p1_matches = get_player_matches(player1)
        p2_matches = get_player_matches(player2)
        
        if p1_matches is None or p2_matches is None:
            return jsonify({'error': 'One or both players not found'}), 404
        
        all_surfaces_stats = {}
        surfaces = ['All', 'Hard', 'Clay', 'Grass', 'Carpet']
        
        for surface in surfaces:
            surface_filter = None if surface == 'All' else surface
            try:
                p1_stats = calculate_yearly_stats(p1_matches, surface_filter)
                p2_stats = calculate_yearly_stats(p2_matches, surface_filter)
                
                if p1_stats is None or p2_stats is None or p1_stats.empty or p2_stats.empty:
                    continue
                
                if year in p1_stats.index and year in p2_stats.index:
                    p1_row = p1_stats.loc[year]
                    p2_row = p2_stats.loc[year]
                    if isinstance(p1_row, pd.DataFrame):
                        p1_row = p1_row.iloc[0]
                    if isinstance(p2_row, pd.DataFrame):
                        p2_row = p2_row.iloc[0]
                    
                    # Union of stat keys so new fields (e.g., tb_win%, top10_win%) are not lost
                    p1_keys = set(map(str, p1_row.index))
                    p2_keys = set(map(str, p2_row.index))
                    all_keys = sorted(p1_keys.union(p2_keys))
                    
                    # Exclude non-stat helper labels if present
                    exclude = {'surface', 'year'}
                    
                    stats_data = []
                    for stat in all_keys:
                        if stat in exclude:
                            continue
                        p1_val = to_py(p1_row.get(stat, None))
                        p2_val = to_py(p2_row.get(stat, None))
                        # Allow strings (e.g., 'W-L'), numeric percentages, etc.
                        stats_data.append({
                            'stat': stat,
                            'player1_value': p1_val,
                            'player2_value': p2_val,
                            'player1_name': player1,
                            'player2_name': player2
                        })
                    
                    # Only add surface if we have at least one stat
                    if stats_data:
                        all_surfaces_stats[surface] = stats_data
            except Exception as e:
                logging.exception(f"Compare error for surface={surface}: {e}")
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
        
        try:
            matches_df = get_player_matches(player1)
            if matches_df is None:
                return jsonify({'error': f'Player {player1} not found'}), 404
            
            h2h_data = format_h2h_matches(matches_df, player1, player2)
            if h2h_data is None or h2h_data.empty:
                return jsonify({'error': 'No head-to-head matches found between these players'}), 404
            
            # H2H summary
            p1_wins = int((h2h_data['winner_name'] == player1).sum())
            p2_wins = int((h2h_data['winner_name'] == player2).sum())
            
            # Surface breakdown
            surface_stats = h2h_data.groupby('surface')['winner_name'].value_counts().unstack(fill_value=0)
            surface_breakdown = {}
            for surface in surface_stats.index:
                surface_breakdown[str(surface)] = {
                    player1: int(surface_stats.loc[surface].get(player1, 0)),
                    player2: int(surface_stats.loc[surface].get(player2, 0))
                }
            
            # Convert matches to list of dicts, ensure JSON-safe types
            matches_list = []
            for rec in h2h_data.to_dict('records'):
                # Rename fields if needed; ensure match_date is string
                if 'match_date' in rec:
                    rec['match_date'] = to_py(rec['match_date'])
                for k, v in list(rec.items()):
                    rec[k] = to_py(v)
                matches_list.append(rec)
            
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
        except Exception as e:
            logging.exception(f"H2H error: {e}")
            return jsonify({'error': 'An error occurred while computing H2H'}), 500
    
    return render_template('h2h.html')

@app.route('/career', methods=['GET', 'POST'])
def career_stats():
    if request.method == 'POST':
        player_name = request.form.get('player', '').strip()
        
        if not player_name:
            return jsonify({'error': 'Please enter a player name'}), 400
        
        try:
            # Get player matches
            matches = get_player_matches(player_name)
            if matches is None:
                return jsonify({'error': f'Player {player_name} not found'}), 404
            
            career_data = career(player_name)
            if career_data is None or (isinstance(career_data, pd.DataFrame) and career_data.empty):
                return jsonify({'error': f'No data available for {player_name}'}), 404
            
            # Recent form
            recent_form = calculate_recent_form(matches) or {}
            # Make recent_form JSON-safe
            if isinstance(recent_form, dict):
                if 'matches' in recent_form and isinstance(recent_form['matches'], list):
                    for m in recent_form['matches']:
                        for k, v in list(m.items()):
                            m[k] = to_py(v)
                for k, v in list(recent_form.items()):
                    if k != 'matches':
                        recent_form[k] = to_py(v)
            
            # Surface breakdown with yearly details
            surface_breakdown = {}
            for surface in ['Hard', 'Clay', 'Grass', 'Carpet']:
                try:
                    surface_stats = calculate_yearly_stats(matches, surface)
                except Exception as e:
                    logging.exception(f"calculate_yearly_stats failed on surface={surface}: {e}")
                    surface_stats = None
                
                if surface_stats is not None and not surface_stats.empty:
                    # Yearly dict
                    surface_yearly = {}
                    for yr in surface_stats.index:
                        row = surface_stats.loc[yr]
                        if isinstance(row, pd.DataFrame):
                            row = row.iloc[0]
                        row_dict = series_to_py_dict(row)
                        row_dict['year'] = int(yr)
                        surface_yearly[str(int(yr))] = row_dict
                    
                    # Surface career totals
                    try:
                        surface_matches = matches[matches['surf'] == surface]
                        surface_career = calculate_career_stats(surface_matches)
                    except Exception as e:
                        logging.exception(f"calculate_career_stats failed on surface={surface}: {e}")
                        surface_career = None
                    
                    if surface_career is not None and not surface_career.empty:
                        career_row = surface_career.iloc[0]
                        surface_breakdown[surface] = {
                            'yearly': surface_yearly,
                            'career': series_to_py_dict(career_row)
                        }
            
            # Separate yearly stats from career summary; accept DF or dict
            if isinstance(career_data, pd.DataFrame):
                if 'career' in career_data.index:
                    career_summary_row = career_data.loc['career']
                    if isinstance(career_summary_row, pd.DataFrame):
                        career_summary_row = career_summary_row.iloc[0]
                    career_summary = series_to_py_dict(career_summary_row)
                else:
                    # Fallback: last row as career if no explicit 'career'
                    career_summary = series_to_py_dict(career_data.iloc[-1])
                
                yearly_stats_df = career_data[career_data.index != 'career'].copy()
                yearly_data = []
                for yr, row in yearly_stats_df.iterrows():
                    row_dict = series_to_py_dict(row)
                    try:
                        row_dict['year'] = int(yr)
                    except Exception:
                        # If index is string, try to coerce
                        try:
                            row_dict['year'] = int(str(yr).strip())
                        except Exception:
                            row_dict['year'] = to_py(yr)
                    yearly_data.append(row_dict)
                yearly_data.sort(key=lambda x: (x.get('year') is None, x.get('year')), reverse=True)
            elif isinstance(career_data, dict):
                # Support dict-based career() if mod.py changed
                yearly_data = career_data.get('yearly_stats', [])
                career_summary = career_data.get('career_summary', {})
            else:
                return jsonify({'error': f'Unexpected career data format for {player_name}'}), 500
            
            return jsonify({
                'yearly_stats': yearly_data,
                'career_summary': career_summary,
                'recent_form': recent_form,
                'surface_breakdown': surface_breakdown,
                'player': player_name
            })
        except Exception as e:
            logging.exception(f"Career error: {e}")
            return jsonify({'error': 'An error occurred while computing career stats'}), 500
    
    return render_template('career.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
