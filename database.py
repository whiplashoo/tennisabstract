import sqlite3
from contextlib import contextmanager
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class TennisDatabase:
    def __init__(self, db_path='tennis_stats.db'):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database tables"""
        with self.get_db() as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS players (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    name_normalized TEXT,
                    current_rank INTEGER,
                    peak_rank INTEGER,
                    handedness TEXT,
                    backhand_style TEXT,
                    age_years REAL,
                    birthdate TEXT,
                    height_cm INTEGER,
                    coach_names TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id INTEGER,
                    date DATE,
                    tournament TEXT,
                    surface TEXT,
                    round TEXT,
                    opponent TEXT,
                    opponent_rank INTEGER,
                    result TEXT,
                    score TEXT,
                    aces INTEGER,
                    dfs INTEGER,
                    pts INTEGER,
                    firsts INTEGER,
                    fwon INTEGER,
                    swon INTEGER,
                    games INTEGER,
                    saved INTEGER,
                    chances INTEGER,
                    oaces INTEGER,
                    odfs INTEGER,
                    opts INTEGER,
                    ofirsts INTEGER,
                    ofwon INTEGER,
                    oswon INTEGER,
                    ogames INTEGER,
                    osaved INTEGER,
                    ochances INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (player_id) REFERENCES players (id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_matches_player_date 
                ON matches (player_id, date DESC);
                
                CREATE INDEX IF NOT EXISTS idx_matches_player_surface 
                ON matches (player_id, surface);
                
                CREATE INDEX IF NOT EXISTS idx_players_normalized 
                ON players (name_normalized);
                
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    player_id INTEGER PRIMARY KEY,
                    last_updated TIMESTAMP,
                    FOREIGN KEY (player_id) REFERENCES players (id)
                );
                
                CREATE TABLE IF NOT EXISTS player_list_cache (
                    id INTEGER PRIMARY KEY,
                    players_json TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            ''')

            # Ensure backward-compatible schema by adding missing columns on existing DBs
            try:
                cols = {row[1] for row in conn.execute("PRAGMA table_info(players)").fetchall()}
                alter_stmts = []
                if 'current_rank' not in cols:
                    alter_stmts.append("ALTER TABLE players ADD COLUMN current_rank INTEGER")
                if 'peak_rank' not in cols:
                    alter_stmts.append("ALTER TABLE players ADD COLUMN peak_rank INTEGER")
                if 'handedness' not in cols:
                    alter_stmts.append("ALTER TABLE players ADD COLUMN handedness TEXT")
                if 'backhand_style' not in cols:
                    alter_stmts.append("ALTER TABLE players ADD COLUMN backhand_style TEXT")
                if 'age_years' not in cols:
                    alter_stmts.append("ALTER TABLE players ADD COLUMN age_years REAL")
                if 'birthdate' not in cols:
                    alter_stmts.append("ALTER TABLE players ADD COLUMN birthdate TEXT")
                if 'height_cm' not in cols:
                    alter_stmts.append("ALTER TABLE players ADD COLUMN height_cm INTEGER")
                if 'coach_names' not in cols:
                    alter_stmts.append("ALTER TABLE players ADD COLUMN coach_names TEXT")
                for stmt in alter_stmts:
                    conn.execute(stmt)
            except Exception:
                # Best-effort migration; safe to ignore
                pass
    
    @contextmanager
    def get_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def normalize_name(self, name):
        """Normalize player name for consistent storage"""
        return name.lower().replace(' ', '').replace('-', '').replace("'", '')
    
    def get_or_create_player(self, name):
        """Get player ID or create new player"""
        normalized = self.normalize_name(name)
        
        with self.get_db() as conn:
            player = conn.execute(
                'SELECT * FROM players WHERE name_normalized = ?', (normalized,)
            ).fetchone()
            
            if not player:
                cursor = conn.execute(
                    'INSERT INTO players (name, name_normalized) VALUES (?, ?)', 
                    (name, normalized)
                )
                return cursor.lastrowid
            
            return player['id']

    def upsert_player_profile(self, name: str, profile: dict) -> None:
        """Insert or update player profile fields in players table.

        Expected keys in profile: current_rank, peak_rank, handedness,
        backhand_style, age_years, birthdate.
        """
        player_id = self.get_or_create_player(name)
        fields = {
            'current_rank': profile.get('current_rank'),
            'peak_rank': profile.get('peak_rank'),
            'handedness': profile.get('handedness'),
            'backhand_style': profile.get('backhand_style'),
            'age_years': profile.get('age_years'),
            'birthdate': profile.get('birthdate'),
            'height_cm': profile.get('height_cm'),
            'coach_names': json.dumps(profile.get('coach_names')) if profile.get('coach_names') else None
        }
        # Build dynamic SET clause
        set_clause = ', '.join([f"{k} = ?" for k, v in fields.items()])
        values = list(fields.values())
        values.append(player_id)
        with self.get_db() as conn:
            conn.execute(f"UPDATE players SET {set_clause} WHERE id = ?", values)
    
    def is_cache_valid(self, player_id, expire_hours=6):
        """Check if player cache is still valid"""
        with self.get_db() as conn:
            metadata = conn.execute(
                'SELECT last_updated FROM cache_metadata WHERE player_id = ?',
                (player_id,)
            ).fetchone()
            
            if not metadata:
                return False
            
            last_updated = datetime.fromisoformat(metadata['last_updated'])
            return datetime.now() - last_updated < timedelta(hours=expire_hours)
    
    def get_player_matches(self, player_name):
        """Get all matches for a player from database"""
        player_id = self.get_or_create_player(player_name)
        
        if not self.is_cache_valid(player_id):
            return None
        
        with self.get_db() as conn:
            matches = conn.execute('''
                SELECT * FROM matches 
                WHERE player_id = ? 
                ORDER BY date DESC
            ''', (player_id,)).fetchall()
            
            if not matches:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame([dict(match) for match in matches])
            df['date'] = pd.to_datetime(df['date'])
            
            # Drop database-specific columns
            df = df.drop(['id', 'player_id', 'created_at'], axis=1)
            
            # Rename columns to match original format
            df = df.rename(columns={
                'tournament': 'tourn',
                'surface': 'surf',
                'result': 'wl',
                'opponent': 'opp',
                'opponent_rank': 'orank'
            })
            
            return df
    
    def cache_player_matches(self, player_name, matches_df):
        """Store matches in database"""
        if matches_df is None or matches_df.empty:
            return
        
        player_id = self.get_or_create_player(player_name)
        
        with self.get_db() as conn:
            # Clear old matches
            conn.execute('DELETE FROM matches WHERE player_id = ?', (player_id,))
            
            # Insert new matches
            for _, match in matches_df.iterrows():
                conn.execute('''
                    INSERT INTO matches 
                    (player_id, date, tournament, surface, round, opponent, 
                     opponent_rank, result, score, aces, dfs, pts, firsts, 
                     fwon, swon, games, saved, chances, oaces, odfs, opts, 
                     ofirsts, ofwon, oswon, ogames, osaved, ochances)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    player_id, 
                    match['date'].strftime('%Y-%m-%d') if pd.notna(match['date']) else None,
                    match.get('tourn'), match.get('surf'), match.get('round'),
                    match.get('opp'), match.get('orank'), match.get('wl'),
                    match.get('score'), match.get('aces'), match.get('dfs'),
                    match.get('pts'), match.get('firsts'), match.get('fwon'),
                    match.get('swon'), match.get('games'), match.get('saved'),
                    match.get('chances'), match.get('oaces'), match.get('odfs'),
                    match.get('opts'), match.get('ofirsts'), match.get('ofwon'),
                    match.get('oswon'), match.get('ogames'), match.get('osaved'),
                    match.get('ochances')
                ))
            
            # Update cache metadata
            conn.execute('''
                INSERT OR REPLACE INTO cache_metadata (player_id, last_updated)
                VALUES (?, datetime('now'))
            ''', (player_id,))
    
    def get_all_players(self):
        """Get list of all players in database"""
        with self.get_db() as conn:
            players = conn.execute(
                'SELECT DISTINCT name FROM players ORDER BY name'
            ).fetchall()
            
            return [player['name'] for player in players]
    
    def cache_player_list(self, players):
        """Cache the complete player list"""
        with self.get_db() as conn:
            conn.execute('DELETE FROM player_list_cache')
            conn.execute(
                'INSERT INTO player_list_cache (players_json) VALUES (?)',
                (json.dumps(players),)
            )
    
    def get_cached_player_list(self, expire_days=7):
        """Get cached player list if valid"""
        with self.get_db() as conn:
            cache = conn.execute(
                'SELECT * FROM player_list_cache ORDER BY last_updated DESC LIMIT 1'
            ).fetchone()
            
            if not cache:
                return None
            
            last_updated = datetime.fromisoformat(cache['last_updated'])
            if datetime.now() - last_updated > timedelta(days=expire_days):
                return None
            
            return json.loads(cache['players_json'])
