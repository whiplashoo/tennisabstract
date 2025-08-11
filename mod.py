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
from config import (BASE_URL, REQUEST_DELAY, REQUEST_RETRIES,
                    CACHE_EXPIRE_HOURS, DATABASE_PATH)
from database import TennisDatabase

# Tournament round order for sorting
ROUND_SORT_ORDER = {
    'Q1': 0, 'Q2': 1, 'Q3': 2, 'R128': 3, 'R64': 4, 'ER': 5,
    'R32': 6, 'R16': 7, 'RR': 8, 'QF': 9, 'SF': 10, 'BR': 11, 'F': 12
}

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize database
db = TennisDatabase(DATABASE_PATH)


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
        """Get all matches for a player with database caching"""
        # Check database first
        cached_data = db.get_player_matches(player_name)
        if cached_data is not None:
            logging.info(f"Using database cached data for {player_name}")
            return cached_data

        # If not in database or expired, fetch from web
        data = self._fetch_player_matches(player_name)
        if data is not None:
            db.cache_player_matches(player_name, data)
        return data

    def _fetch_player_matches(self, player_name):
        """Fetch player matches from web"""
        player_name_url = player_name.replace(' ', '')
        all_matches = []

        # Try HTML page first
        try:
            html_url = f'{BASE_URL}/cgi-bin/player-classic.cgi?p={player_name_url}'
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
                f"{BASE_URL}/jsmatches/{player_name_url}.js",
                f"{BASE_URL}/jsmatches/{player_name_url}Career.js"
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
        # Include 'level' at index 3
        essential_columns = {
            0: 'date', 1: 'tourn', 2: 'surf', 3: 'level', 4: 'wl', 8: 'round',
            9: 'score', 11: 'opp', 12: 'orank', 21: 'aces', 22: 'dfs',
            23: 'pts', 24: 'firsts', 25: 'fwon', 26: 'swon', 27: 'games',
            28: 'saved', 29: 'chances', 30: 'oaces', 31: 'odfs', 32: 'opts',
            33: 'ofirsts', 34: 'ofwon', 35: 'oswon', 36: 'ogames',
            37: 'osaved', 38: 'ochances'
        }

        df = pd.DataFrame(matches)
        df = df[list(essential_columns.keys())].rename(columns=essential_columns)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')

        # Filter out walkovers and empty scores
        df = df[~df['score'].isin(['W/O', '', None])]
        df = df[df['score'].notna()]

        df = df.sort_values('date')

        return df


class TennisStatsCalculator:
    @staticmethod
    def _count_tiebreaks(score: str):
        """
        Count tiebreaks won/lost for the player from a match score string.
        Assumes the player's set score is the first number in each set (player pages).
        A tiebreak set is detected by set games 7-6 or 6-7 (parentheses optional).
        """
        if not isinstance(score, str) or not score:
            return 0, 0

        # Find all set scores like "7-6(5)" or "6-7(4)" or "7-6"
        pairs = re.findall(r'(\d+)\s*-\s*(\d+)(?:$[^)]+$)?', score)
        tb_won = 0
        tb_lost = 0
        for a, b in pairs:
            try:
                ga = int(a)
                gb = int(b)
            except ValueError:
                continue
            if ga == 7 and gb == 6:
                tb_won += 1
            elif ga == 6 and gb == 7:
                tb_lost += 1

        return tb_won, tb_lost

    @staticmethod
    def calculate_yearly_stats(df, surface=None):
        """Calculate tennis statistics by year and optionally by surface, including Top-N opponent records"""
        if df is None or df.empty:
            return pd.DataFrame()

        numeric_columns = ['pts', 'aces', 'dfs', 'firsts', 'fwon', 'swon',
                           'saved', 'chances', 'games', 'ogames', 'osaved',
                           'ochances', 'orank']
        top_brackets = (5, 10, 20, 50, 100)

        stats_df = df.copy()
        stats_df['year'] = stats_df['date'].dt.year

        # Filter by surface if specified
        if surface:
            stats_df = stats_df[stats_df['surf'] == surface]
            if stats_df.empty:
                return pd.DataFrame()

        # Filter out walkovers and empty scores
        valid_df = stats_df[~stats_df['score'].isin(['W/O', '', None])]
        valid_df = valid_df[valid_df['score'].notna()].copy()

        for col in numeric_columns:
            valid_df[col] = pd.to_numeric(valid_df[col], errors='coerce')

        # Win-loss record for all valid matches
        wl_record = valid_df.groupby('year')['wl'].agg(
            wins=lambda x: (x == 'W').sum(),
            losses=lambda x: (x == 'L').sum()
        )
        wl_record['total'] = wl_record['wins'] + wl_record['losses']
        wl_record['win%'] = (wl_record['wins'] / wl_record['total'] * 100).round(1)
        wl_record['W-L'] = wl_record.apply(lambda x: f"{int(x['wins'])}-{int(x['losses'])}", axis=1)

        # Filter for matches with stats
        with_pts = valid_df[valid_df['pts'].notna()]

        # Calculate yearly sums (where stats exist)
        yearly_sums = with_pts.groupby('year')[numeric_columns].agg({
            'aces': 'sum', 'dfs': 'sum', 'pts': 'sum', 'firsts': 'sum',
            'fwon': 'sum', 'swon': 'sum', 'saved': 'sum', 'chances': 'sum',
            'games': 'sum', 'ogames': 'sum', 'osaved': 'sum', 'ochances': 'sum',
            'orank': 'mean'
        }).fillna(0)

        # Build result frame indexed by all years where matches occurred
        years_index = wl_record.index.union(yearly_sums.index)
        yearly_stats = pd.DataFrame(index=years_index)

        # Basic records
        yearly_stats['W-L'] = wl_record['W-L'].reindex(yearly_stats.index)
        yearly_stats['win%'] = wl_record['win%'].reindex(yearly_stats.index)

        # Reindex sums to align
        ys = yearly_sums.reindex(yearly_stats.index).fillna(0)

        # Percentages
        with np.errstate(divide='ignore', invalid='ignore'):
            yearly_stats['ace%'] = (ys['aces'] / ys['pts'] * 100).round(2)
            yearly_stats['df%'] = (ys['dfs'] / ys['pts'] * 100).round(2)
            yearly_stats['1st_in%'] = (ys['firsts'] / ys['pts'] * 100).round(1)
            yearly_stats['1st_win%'] = (ys['fwon'] / ys['firsts'] * 100).round(1)

            second_serves = ys['pts'] - ys['firsts']
            yearly_stats['2nd_win%'] = (ys['swon'] / second_serves * 100).round(1)
            yearly_stats['bp_saved%'] = (ys['saved'] / ys['chances'] * 100).round(1)
            yearly_stats['hold%'] = (100 - ((ys['chances'] - ys['saved']) / ys['games'] * 100)).round(1)
            yearly_stats['break%'] = ((ys['ochances'] - ys['osaved']) / ys['ogames'] * 100).round(1)

        yearly_stats['avg_opp_rank'] = ys['orank'].round(1)
        yearly_stats['avg_opp_rank'] = yearly_stats['avg_opp_rank'].fillna(0)

        # Share of matches that have detailed stats
        counts_with_pts = with_pts.groupby('year').size()
        yearly_stats['matches_with_stats%'] = (
            (counts_with_pts.reindex(yearly_stats.index).fillna(0)) /
            (wl_record['total'].reindex(yearly_stats.index).fillna(0))
            * 100
        ).round(1)

        # Top-N opponent records (ignore matches with missing orank)
        orank_num = pd.to_numeric(valid_df['orank'], errors='coerce')
        for top_n in top_brackets:
            sub = valid_df[orank_num <= top_n]
            if sub.empty:
                yearly_stats[f'top{top_n}_W-L'] = '0-0'
                yearly_stats[f'top{top_n}_win%'] = 0.0
                continue
            g = sub.groupby('year')['wl'].agg(
                wins=lambda x: (x == 'W').sum(),
                losses=lambda x: (x == 'L').sum()
            )
            g = g.reindex(yearly_stats.index).fillna(0)
            totals = g['wins'] + g['losses']
            with np.errstate(divide='ignore', invalid='ignore'):
                winp = (g['wins'] / totals * 100).round(1)
            wl_str = (g['wins'].astype(int).astype(str) + '-' + g['losses'].astype(int).astype(str))
            yearly_stats[f'top{top_n}_W-L'] = wl_str
            yearly_stats[f'top{top_n}_win%'] = winp.fillna(0)

        # Tiebreak records (based on score, all valid matches)
        tb_pairs = valid_df['score'].apply(TennisStatsCalculator._count_tiebreaks)
        tb_df = pd.DataFrame(tb_pairs.tolist(), columns=['tb_won', 'tb_lost'])
        tb_df['year'] = valid_df['year'].values
        tb_year = tb_df.groupby('year')[['tb_won', 'tb_lost']].sum().reindex(yearly_stats.index).fillna(0)

        tb_totals = tb_year['tb_won'] + tb_year['tb_lost']
        with np.errstate(divide='ignore', invalid='ignore'):
            tb_winp = (tb_year['tb_won'] / tb_totals * 100).round(1)

        yearly_stats['tb_W-L'] = (tb_year['tb_won'].astype(int).astype(str)
                                  + '-' + tb_year['tb_lost'].astype(int).astype(str))
        yearly_stats['tb_win%'] = tb_winp.fillna(0)

        if surface:
            yearly_stats['surface'] = surface

        yearly_stats = yearly_stats.replace([np.inf, -np.inf], np.nan).fillna(0)

        return yearly_stats

    @staticmethod
    def calculate_surface_breakdown(df):
        """Calculate statistics broken down by surface"""
        if df is None or df.empty:
            return {}

        surfaces = df['surf'].unique()
        surface_stats = {}

        for surface in surfaces:
            if pd.notna(surface):
                stats = TennisStatsCalculator.calculate_yearly_stats(df, surface)
                if not stats.empty:
                    surface_stats[surface] = stats

        return surface_stats

    @staticmethod
    def calculate_recent_form(df, num_matches=10):
        """Calculate form over last N matches with proper tournament round sorting"""
        if df is None or df.empty:
            return {}

        # Get last N matches (excluding walkovers)
        valid_matches = df[~df['score'].isin(['W/O', '', None])]
        valid_matches = valid_matches[valid_matches['score'].notna()]

        # Sort by date first, then by tournament and round for same-date matches
        valid_matches = valid_matches.copy()
        valid_matches['round_order'] = valid_matches['round'].map(ROUND_SORT_ORDER).fillna(99)

        # Sort by date (descending), then by tournament name, then by round order (descending)
        valid_matches = valid_matches.sort_values(
            ['date', 'tourn', 'round_order'],
            ascending=[False, True, False]
        )

        recent = valid_matches.head(num_matches)

        if recent.empty:
            return {}

        # Calculate basic stats
        wins = sum(recent['wl'] == 'W')
        losses = sum(recent['wl'] == 'L')

        form_string = ''.join(recent['wl'].astype(str).tolist())

        # Calculate current streak from most recent match
        win_streak = 0
        loss_streak = 0
        for r in form_string:
            if r == 'W':
                if loss_streak == 0:
                    win_streak += 1
                else:
                    break
            elif r == 'L':
                if win_streak == 0:
                    loss_streak += 1
                else:
                    break

        form_stats = {
            'last_matches': len(recent),
            'wins': wins,
            'losses': losses,
            'win_pct': round((wins / len(recent) * 100), 1) if len(recent) > 0 else 0,
            'win_streak': win_streak,
            'loss_streak': loss_streak,
            'form_string': form_string,
            'avg_opp_rank': round(pd.to_numeric(recent['orank'], errors='coerce').mean(), 1)
        }

        # Surface breakdown
        surface_breakdown = recent.groupby('surf')['wl'].value_counts().unstack(fill_value=0)
        form_stats['surface_breakdown'] = surface_breakdown.to_dict('index')

        # Recent matches details (remove round_order from output)
        recent_matches = recent[['date', 'tourn', 'surf', 'opp', 'wl', 'score', 'round']].copy()
        recent_matches['date'] = recent_matches['date'].dt.strftime('%Y-%m-%d')
        form_stats['matches'] = recent_matches.to_dict('records')

        return form_stats

    @staticmethod
    def calculate_career_stats(df):
        """Calculate career statistics, including Top-N opponent records"""
        if df is None or df.empty:
            return pd.DataFrame()

        stats_df = df.copy()
        numeric_columns = ['pts', 'aces', 'dfs', 'firsts', 'fwon', 'swon',
                           'saved', 'chances', 'games', 'ogames', 'osaved',
                           'ochances', 'orank']
        top_brackets = (5, 10, 20, 50, 100)

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

        career_stats = {
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
            'avg_opp_rank': round(career_sums['orank'], 1) if not np.isnan(career_sums['orank']) else 0,
            'matches_with_stats%': stats_pct
        }

        # Top-N opponent records (ignore matches with missing orank)
        orank_num = pd.to_numeric(stats_df['orank'], errors='coerce')
        for top_n in top_brackets:
            sub = stats_df[orank_num <= top_n]
            w = (sub['wl'] == 'W').sum()
            l = (sub['wl'] == 'L').sum()
            t = w + l
            wl = f"{int(w)}-{int(l)}"
            wp = round((w / t * 100), 1) if t > 0 else 0.0
            career_stats[f'top{top_n}_W-L'] = wl
            career_stats[f'top{top_n}_win%'] = wp

        # Tiebreak records (career)
        tb_pairs = stats_df['score'].apply(TennisStatsCalculator._count_tiebreaks)
        tb_won = int(sum(w for w, _ in tb_pairs))
        tb_lost = int(sum(l for _, l in tb_pairs))
        tb_total = tb_won + tb_lost
        tb_winp = round(tb_won / tb_total * 100, 1) if tb_total > 0 else 0.0

        career_stats['tb_W-L'] = f"{tb_won}-{tb_lost}"
        career_stats['tb_win%'] = tb_winp

        out = pd.DataFrame(career_stats, index=['career'])
        return out.replace([np.inf, -np.inf], np.nan).fillna(0)

    @staticmethod
    def calculate_topn_records(df, surface=None, brackets=(5, 10, 20, 50, 100)):
        """Return per-year W-L and win% vs Top-N opponents based on orank"""
        if df is None or df.empty:
            return pd.DataFrame()

        data = df.copy()
        if surface:
            data = data[data['surf'] == surface]
            if data.empty:
                return pd.DataFrame()

        data = data[~data['score'].isin(['W/O', '', None])]
        data = data[data['score'].notna()].copy()
        data['year'] = data['date'].dt.year
        data['orank'] = pd.to_numeric(data['orank'], errors='coerce')

        if data.empty:
            return pd.DataFrame()

        years = sorted(data['year'].unique())
        out = pd.DataFrame(index=years)

        for top_n in brackets:
            sub = data[data['orank'] <= top_n]
            if sub.empty:
                out[f'top{top_n}_W-L'] = '0-0'
                out[f'top{top_n}_win%'] = 0.0
                continue
            g = sub.groupby('year')['wl'].agg(
                wins=lambda x: (x == 'W').sum(),
                losses=lambda x: (x == 'L').sum()
            )
            g = g.reindex(out.index).fillna(0)
            totals = g['wins'] + g['losses']
            with np.errstate(divide='ignore', invalid='ignore'):
                winp = (g['wins'] / totals * 100).round(1)
            wl_str = (g['wins'].astype(int).astype(str) + '-' + g['losses'].astype(int).astype(str))
            out[f'top{top_n}_W-L'] = wl_str
            out[f'top{top_n}_win%'] = winp.fillna(0)

        if surface:
            out['surface'] = surface
        return out

    @staticmethod
    def calculate_tiebreak_records(df, surface=None):
        """
        Return per-year tiebreak W-L and win% based on set scores (7-6/6-7).
        """
        if df is None or df.empty:
            return pd.DataFrame()

        data = df.copy()
        if surface:
            data = data[data['surf'] == surface]
            if data.empty:
                return pd.DataFrame()

        # Exclude walkovers/empty scores similarly to other stats
        data = data[~data['score'].isin(['W/O', '', None])]
        data = data[data['score'].notna()].copy()
        data['year'] = data['date'].dt.year

        if data.empty:
            return pd.DataFrame()

        tb_pairs = data['score'].apply(TennisStatsCalculator._count_tiebreaks)
        tb_df = pd.DataFrame(tb_pairs.tolist(), columns=['tb_won', 'tb_lost'])
        tb_df['year'] = data['year'].values

        out = tb_df.groupby('year')[['tb_won', 'tb_lost']].sum().sort_index()
        totals = out['tb_won'] + out['tb_lost']
        with np.errstate(divide='ignore', invalid='ignore'):
            winp = (out['tb_won'] / totals * 100).round(1)

        result = pd.DataFrame(index=out.index)
        result['tb_W-L'] = out['tb_won'].astype(int).astype(str) + '-' + out['tb_lost'].astype(int).astype(str)
        result['tb_win%'] = winp.fillna(0)

        if surface:
            result['surface'] = surface
        return result

    @staticmethod
    def format_h2h_matches(matches_df, player1, player2):
        """Format head-to-head matches"""
        if matches_df is None or matches_df.empty:
            return pd.DataFrame()

        # Filter out walkovers and empty scores before processing H2H
        matches_df_clean = matches_df[~matches_df['score'].isin(['W/O', '', None])]
        matches_df_clean = matches_df_clean[matches_df_clean['score'].notna()].copy()

        # Normalize opponent name for comparison
        player2_normalized = player2.lower().replace(' ', '').replace('-', '').replace("'", '')
        matches_df_clean['opp_normalized'] = (
            matches_df_clean['opp']
            .str.lower()
            .str.replace(' ', '', regex=False)
            .str.replace('-', '', regex=False)
            .str.replace("'", '', regex=False)
        )

        h2h_matches = matches_df_clean[matches_df_clean['opp_normalized'] == player2_normalized][
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

def calculate_yearly_stats(df, surface=None):
    return calculator.calculate_yearly_stats(df, surface)

def calculate_surface_breakdown(df):
    return calculator.calculate_surface_breakdown(df)

def calculate_recent_form(df, num_matches=20):
    return calculator.calculate_recent_form(df, num_matches)

def calculate_career_stats(df):
    return calculator.calculate_career_stats(df)

def calculate_topn_records(df, surface=None, brackets=(5, 10, 20, 50, 100)):
    return calculator.calculate_topn_records(df, surface=surface, brackets=brackets)

def calculate_tiebreak_records(df, surface=None):
    return calculator.calculate_tiebreak_records(df, surface=surface)

def format_h2h_matches(matches_df, player1, player2):
    return calculator.format_h2h_matches(matches_df, player1, player2)

def compare(p1, p2, year=2025, surface=None):
    try:
        p1_stats = calculate_yearly_stats(get_player_matches(p1), surface)
        p2_stats = calculate_yearly_stats(get_player_matches(p2), surface)

        p1_data = p1_stats.loc[year]
        p2_data = p2_stats.loc[year]

        result = pd.concat([
            p1_data.to_frame(p1),
            p2_data.to_frame(p2)
        ], axis=1)

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


# Example usage (safe to remove or comment out)
if __name__ == '__main__':
    try:
        player = "Daria Kasatkina"
        opponent = "Maya Joint"

        # Fetch matches (now includes 'level' column)
        matches_df = get_player_matches(player)
        print(f"Fetched matches for {player}. Columns: {list(matches_df.columns) if matches_df is not None else 'No data'}")

        # Yearly stats (all surfaces) with Top-N records and tiebreaks
        ys = calculate_yearly_stats(matches_df)
        print("\nYearly stats (tail):")
        print(ys.tail(3))

        # Yearly stats by surface with Top-N records and tiebreaks
        try:
            clay_ys = calculate_yearly_stats(matches_df, surface="Clay")
            if not clay_ys.empty:
                print("\nClay 2024 sample:")
                print(clay_ys.loc[2024, ['W-L', 'win%', 'top10_W-L', 'top10_win%', 'tb_W-L', 'tb_win%']])
        except Exception:
            pass

        # Dedicated Top-N records per year
        topn_all = calculate_topn_records(matches_df)
        print("\nTop-N per year (tail):")
        print(topn_all.tail(3))

        # Dedicated tiebreak records per year
        tb_yearly = calculate_tiebreak_records(matches_df)
        print("\nTiebreak records per year (tail):")
        print(tb_yearly.tail(3))

        # Career stats with Top-N records and tiebreaks
        car = career(player)
        if car is not None and 'career' in car.index:
            print("\nCareer stats (selected columns):")
            print(car.loc['career', ['W-L', 'win%', 'top50_W-L', 'top50_win%', 'tb_W-L', 'tb_win%']])

        # Recent form
        recent = calculate_recent_form(matches_df, num_matches=10)
        print("\nRecent form:")
        print({
            'wins': recent.get('wins'),
            'losses': recent.get('losses'),
            'form_string': recent.get('form_string'),
            'avg_opp_rank': recent.get('avg_opp_rank')
        })

        # H2H formatting
        h2h_df = format_h2h_matches(matches_df, player, opponent)
        print(f"\nH2H {player} vs {opponent} (head):")
        print(h2h_df.head())

        # Compare two players on a given year/surface
        cmp_df = compare(player, opponent, year=2024, surface='Hard')
        print("\nCompare on 2024 Hard:")
        print(cmp_df)

        # Tiebreak examples
        print("\nTiebreak examples:")
        
        # Per-year tiebreak records (all surfaces)
        tb_yearly = calculate_tiebreak_records(matches_df)
        if not tb_yearly.empty:
            print("Per-year tiebreaks (2024):")
            try:
                print(tb_yearly.loc[2024])
            except KeyError:
                print("No data for 2024")

        # Per-year tiebreak records on a surface
        tb_hard = calculate_tiebreak_records(matches_df, surface="Hard")
        if not tb_hard.empty:
            print("\nHard court tiebreaks (tail):")
            print(tb_hard.tail())

    except Exception as e:
        logging.error(f"Example usage error: {e}")
