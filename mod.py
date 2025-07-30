import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
import demjson3 as demjson
import time
import numpy as np
import re
import ast
from functools import lru_cache
from datetime import datetime, timedelta
import pickle
import os
from config import BASE_URL, REQUEST_DELAY, REQUEST_RETRIES

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TennisDataCache:
    """Simple file-based cache for player data"""
    def __init__(self, cache_dir='cache', expire_hours=24):
        self.cache_dir = cache_dir
        self.expire_hours = expire_hours
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, player_name):
        return os.path.join(self.cache_dir, f"{player_name.replace(' ', '_')}.pkl")
    
    def get(self, player_name):
        cache_path = self._get_cache_path(player_name)
        if os.path.exists(cache_path):
            mod_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if datetime.now() - mod_time < timedelta(hours=self.expire_hours):
                try:
                    with open(cache_path, 'rb') as f:
                        return pickle.load(f)
                except:
                    pass
        return None
    
    def set(self, player_name, data):
        cache_path = self._get_cache_path(player_name)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)

# Initialize cache
cache = TennisDataCache()

class TennisDataScraper:
    def __init__(self):
        self.session = self._init_session()
    
    def _init_session(self):
        """Initialize session with headers"""
        session = requests.Session()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        session.headers.update(headers)
        return session
    
    def _make_request(self, url, retries=REQUEST_RETRIES, delay=REQUEST_DELAY):
        """Make HTTP request with retry logic"""
        for attempt in range(retries):
            try:
                time.sleep(delay * (attempt + 1))
                response = self.session.get(url)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                logging.error(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                if attempt == retries - 1:
                    raise
    
    def get_player_matches(self, player_name):
        """Get all matches for a player with caching"""
        # Check cache first
        cached_data = cache.get(player_name)
        if cached_data is not None:
            logging.info(f"Using cached data for {player_name}")
            return cached_data
        
        # If not in cache, fetch from web
        data = self._fetch_player_matches(player_name)
        if data is not None:
            cache.set(player_name, data)
        return data
    
    def _fetch_player_matches(self, player_name):
        """Fetch player matches from web"""
        player_name = player_name.replace(' ', '')
        all_matches = []
        
        # Try HTML page first
        try:
            html_url = f'{BASE_URL}/cgi-bin/player-classic.cgi?p={player_name}'
            response = self._make_request(html_url)
            
            if "Benoit Paire" not in response.text[:4000]:
                if "No player found" in response.text:
                    logging.error(f"Player {player_name} not found")
                    return None
                
                matches = self._parse_matches_from_html(response.text)
                if matches:
                    all_matches.extend(matches)
                    logging.info(f"Found matches in HTML for {player_name}")
        
        except Exception as e:
            logging.warning(f"Could not get matches from HTML for {player_name}: {str(e)}")
        
        # If no matches found, try JS files
        if not all_matches:
            js_urls = [
                f"{BASE_URL}/jsmatches/{player_name}.js",
                f"{BASE_URL}/jsmatches/{player_name}Career.js"
            ]
            for url in js_urls:
                try:
                    response = self._make_request(url)
                    if response.status_code == 200:
                        matches = self._parse_matches_from_js(response.text)
                        if matches:
                            all_matches.extend(matches)
                            logging.info(f"Found matches in JS file: {url}")
                except Exception as e:
                    logging.warning(f"Could not get matches from {url}: {str(e)}")
        
        if all_matches:
            return self._create_matches_dataframe(all_matches)
        
        logging.warning(f"No matches found for {player_name}")
        return None
    
    def _parse_matches_from_html(self, html_content):
        """Extract matches data from HTML content"""
        try:
            start_marker = 'var matchmx = ['
            end_marker = '];'
            
            start_pos = html_content.find(start_marker)
            if start_pos == -1:
                return None
            
            start_pos += len(start_marker) - 1
            end_pos = html_content.find(end_marker, start_pos)
            if end_pos == -1:
                return None
            
            matches_str = html_content[start_pos:end_pos + 1]
            matches_str = matches_str.replace('null', 'None')
            return ast.literal_eval(matches_str)
            
        except Exception as e:
            logging.error(f"Error parsing matches from HTML: {str(e)}")
            return None
    
    def _parse_matches_from_js(self, js_content):
        """Parse matches data from JavaScript content"""
        try:
            if 'matchmx = [' in js_content:
                matches_str = js_content.split('matchmx = [')[1].split('];')[0]
                matches_str = '[' + matches_str + ']'
                return demjson.decode(matches_str)
            return []
        except Exception as e:
            logging.error(f"Error parsing matches from JS: {str(e)}")
            return []
    
    def _create_matches_dataframe(self, matches):
        """Create a DataFrame from matches data"""
        essential_columns = {
            0: 'date', 1: 'tourn', 2: 'surf', 4: 'wl', 8: 'round',
            9: 'score', 11: 'opp', 12: 'orank', 21: 'aces', 22: 'dfs',
            23: 'pts', 24: 'firsts', 25: 'fwon', 26: 'swon', 27: 'games',
            28: 'saved', 29: 'chances', 30: 'oaces', 31: 'odfs', 32: 'opts',
            33: 'ofirsts', 34: 'ofwon', 35: 'oswon', 36: 'ogames',
            37: 'osaved', 38: 'ochances'
        }
        
        df = pd.DataFrame(matches)
        df = df[list(essential_columns.keys())].rename(columns=essential_columns)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        
        # Filter out walkovers and empty scores
        df = df[~df['score'].isin(['W/O', '', None])]
        df = df[df['score'].notna()]
        
        df = df.sort_values('date')
        
        return df


class TennisStatsCalculator:
    @staticmethod
    def calculate_yearly_stats(df):
        """Calculate tennis statistics by year"""
        if df is None or df.empty:
            return pd.DataFrame()
        
        numeric_columns = ['pts', 'aces', 'dfs', 'firsts', 'fwon', 'swon', 
                        'saved', 'chances', 'games', 'ogames', 'osaved', 
                        'ochances', 'orank']
        
        stats_df = df.copy()
        stats_df['year'] = stats_df['date'].dt.year
        
        # Filter out walkovers and empty scores (additional safety check)
        stats_df = stats_df[~stats_df['score'].isin(['W/O', '', None])]
        stats_df = stats_df[stats_df['score'].notna()]
        
        for col in numeric_columns:
            stats_df[col] = pd.to_numeric(stats_df[col], errors='coerce')
        
        # Calculate win-loss record for all valid matches
        wl_record = stats_df.groupby('year')['wl'].agg(
            wins=lambda x: sum(x == 'W'),
            losses=lambda x: sum(x == 'L')
        )
        wl_record['total'] = wl_record['wins'] + wl_record['losses']
        wl_record['win%'] = (wl_record['wins'] / wl_record['total'] * 100).round(1)
        wl_record['W-L'] = wl_record.apply(lambda x: f"{int(x['wins'])}-{int(x['losses'])}", axis=1)
        
        # Filter for matches with stats
        stats_df = stats_df[stats_df['pts'].notna()]
        
        # Calculate yearly sums
        yearly_sums = stats_df.groupby('year')[numeric_columns].agg({
            'aces': 'sum', 'dfs': 'sum', 'pts': 'sum', 'firsts': 'sum',
            'fwon': 'sum', 'swon': 'sum', 'saved': 'sum', 'chances': 'sum',
            'games': 'sum', 'ogames': 'sum', 'osaved': 'sum', 'ochances': 'sum',
            'orank': 'mean'
        }).fillna(0)
        
        # Calculate percentages
        yearly_stats = pd.DataFrame(index=yearly_sums.index)
        yearly_stats['W-L'] = wl_record['W-L']
        yearly_stats['win%'] = wl_record['win%']
        yearly_stats['ace%'] = (yearly_sums['aces'] / yearly_sums['pts'] * 100).round(2)
        yearly_stats['df%'] = (yearly_sums['dfs'] / yearly_sums['pts'] * 100).round(2)
        yearly_stats['1st_in%'] = (yearly_sums['firsts'] / yearly_sums['pts'] * 100).round(1)
        yearly_stats['1st_win%'] = (yearly_sums['fwon'] / yearly_sums['firsts'] * 100).round(1)
        
        second_serves = yearly_sums['pts'] - yearly_sums['firsts']
        yearly_stats['2nd_win%'] = (yearly_sums['swon'] / second_serves * 100).round(1)
        yearly_stats['bp_saved%'] = (yearly_sums['saved'] / yearly_sums['chances'] * 100).round(1)
        yearly_stats['hold%'] = 100 - ((yearly_sums['chances'] - yearly_sums['saved']) / yearly_sums['games'] * 100).round(1)
        yearly_stats['break%'] = ((yearly_sums['ochances'] - yearly_sums['osaved']) / yearly_sums['ogames'] * 100).round(1)
        yearly_stats['avg_opp_rank'] = yearly_sums['orank'].round(1)
        yearly_stats['matches_with_stats%'] = (stats_df.groupby('year').size() / wl_record['total'] * 100).round(1)
        
        yearly_stats = yearly_stats.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return yearly_stats

    
    @staticmethod
    def calculate_career_stats(df):
        """Calculate career statistics"""
        if df is None or df.empty:
            return pd.DataFrame()
        
        stats_df = df.copy()
        numeric_columns = ['pts', 'aces', 'dfs', 'firsts', 'fwon', 'swon', 
                        'saved', 'chances', 'games', 'ogames', 'osaved', 
                        'ochances', 'orank']
        
        for col in numeric_columns:
            stats_df[col] = pd.to_numeric(stats_df[col], errors='coerce')
        
        # Filter out walkovers and empty scores
        stats_df = stats_df[~stats_df['score'].isin(['W/O', '', None])]
        stats_df = stats_df[stats_df['score'].notna()]
        
        total_matches = len(stats_df)
        if total_matches == 0:
            return pd.DataFrame()
        
        wins = sum(stats_df['wl'] == 'W')
        losses = total_matches - wins
        win_pct = round((wins / total_matches * 100), 1)
        wl_record = f"{wins}-{losses}"
        
        matches_with_stats = stats_df[stats_df['pts'].notna()]
        stats_pct = round((len(matches_with_stats) / total_matches * 100), 1) if total_matches > 0 else 0
        
        career_sums = matches_with_stats[numeric_columns].agg({
            'aces': 'sum', 'dfs': 'sum', 'pts': 'sum', 'firsts': 'sum',
            'fwon': 'sum', 'swon': 'sum', 'saved': 'sum', 'chances': 'sum',
            'games': 'sum', 'ogames': 'sum', 'osaved': 'sum', 'ochances': 'sum',
            'orank': 'mean'
        }).fillna(0)
        
        career_stats = pd.DataFrame({
            'W-L': wl_record,
            'win%': win_pct,
            'ace%': (career_sums['aces'] / career_sums['pts'] * 100).round(2) if career_sums['pts'] > 0 else 0,
            'df%': (career_sums['dfs'] / career_sums['pts'] * 100).round(2) if career_sums['pts'] > 0 else 0,
            '1st_in%': (career_sums['firsts'] / career_sums['pts'] * 100).round(1) if career_sums['pts'] > 0 else 0,
            '1st_win%': (career_sums['fwon'] / career_sums['firsts'] * 100).round(1) if career_sums['firsts'] > 0 else 0,
            '2nd_win%': (career_sums['swon'] / (career_sums['pts'] - career_sums['firsts']) * 100).round(1) if (career_sums['pts'] - career_sums['firsts']) > 0 else 0,
            'bp_saved%': (career_sums['saved'] / career_sums['chances'] * 100).round(1) if career_sums['chances'] > 0 else 0,
            'hold%': 100 - ((career_sums['chances'] - career_sums['saved']) / career_sums['games'] * 100).round(1) if career_sums['games'] > 0 else 0,
            'break%': ((career_sums['ochances'] - career_sums['osaved']) / career_sums['ogames'] * 100).round(1) if career_sums['ogames'] > 0 else 0,
            'avg_opp_rank': career_sums['orank'].round(1),
            'matches_with_stats%': stats_pct
        }, index=['career'])
        
        return career_stats.replace([np.inf, -np.inf], np.nan).fillna(0)

    
    @staticmethod
    def format_h2h_matches(matches_df, player1, player2):
        """Format head-to-head matches"""
        if matches_df is None or matches_df.empty:
            return pd.DataFrame()
        
        # Filter out walkovers and empty scores before processing H2H
        matches_df_clean = matches_df[~matches_df['score'].isin(['W/O', '', None])]
        matches_df_clean = matches_df_clean[matches_df_clean['score'].notna()]
        
        h2h_matches = matches_df_clean[matches_df_clean['opp'] == player2][
            ['date', 'tourn', 'wl', 'surf', 'score', 'round']
        ].copy()
        
        if h2h_matches.empty:
            return pd.DataFrame()
        
        h2h_matches['winner_name'] = np.where(h2h_matches['wl'] == 'W', player1, player2)
        h2h_matches['loser_name'] = np.where(h2h_matches['wl'] == 'W', player2, player1)
        
        formatted_h2h = h2h_matches[['date', 'tourn', 'surf', 'winner_name', 'loser_name', 'score', 'round']].rename(columns={
            'date': 'match_date',
            'tourn': 'tournament',
            'surf': 'surface'
        })
        
        formatted_h2h['match_date'] = pd.to_datetime(formatted_h2h['match_date']).dt.date
        
        # Sort by date descending (latest first)
        formatted_h2h = formatted_h2h.sort_values('match_date', ascending=False)
        
        # Calculate running H2H (from earliest to latest for correct calculation)
        temp_df = formatted_h2h.sort_values('match_date', ascending=True)
        h2h_record = {player1: 0, player2: 0}
        h2h_column = []
        
        for _, row in temp_df.iterrows():
            h2h_record[row['winner_name']] += 1
            h2h_column.append(f"{h2h_record[player1]}-{h2h_record[player2]}")
        
        # Reverse the h2h_column to match the descending date order
        formatted_h2h['h2h'] = h2h_column[::-1]
        
        return formatted_h2h

# Initialize global instances
scraper = TennisDataScraper()
calculator = TennisStatsCalculator()

# Convenience functions for backward compatibility
def get_player_matches(player_name):
    return scraper.get_player_matches(player_name)

def calculate_yearly_stats(df):
    return calculator.calculate_yearly_stats(df)

def calculate_career_stats(df):
    return calculator.calculate_career_stats(df)

def format_h2h_matches(matches_df, player1, player2):
    return calculator.format_h2h_matches(matches_df, player1, player2)

def compare(p1, p2, year=2025):
    try:
        p1_data = calculate_yearly_stats(get_player_matches(p1)).loc[year]
        p2_data = calculate_yearly_stats(get_player_matches(p2)).loc[year]
        
        result = pd.concat([
            pd.DataFrame(p1_data).rename(columns={year: p1}),
            pd.DataFrame(p2_data).rename(columns={year: p2})
        ], axis=1).iloc[:-1]
        
        return result
    except (KeyError, AttributeError):
        return None

def career(player_name):
    try:
        matches = get_player_matches(player_name)
        if matches is None:
            return None
        
        yearly_stats = calculate_yearly_stats(matches)
        career_stats = calculate_career_stats(matches)
        
        return pd.concat([yearly_stats, career_stats])
    except Exception as e:
        logging.error(f"Error getting career stats for {player_name}: {str(e)}")
        return None
