import ast
import logging
import os
import pickle
import re
import time
import unicodedata
from datetime import date as dt_date
from datetime import datetime
from functools import lru_cache

import demjson3 as demjson
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from .config import (BASE_URL, CACHE_EXPIRE_HOURS, DATABASE_PATH,
                     REQUEST_DELAY, REQUEST_RETRIES, TA_REPORTS_BASE)
from .database import TennisDatabase

# Tournament round order for sorting
ROUND_SORT_ORDER = {
    'Q1': 0, 'Q2': 1, 'Q3': 2, 'R128': 3, 'R64': 4, 'ER': 5,
    'R32': 6, 'R16': 7, 'RR': 8, 'QF': 9, 'SF': 10, 'BR': 11, 'F': 12
}

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize database
db = TennisDatabase(DATABASE_PATH)

# ---------------- Score parsing helpers (tiebreaks, oriented by WL) ----------------
def _standardize_score_dashes(score: str) -> str:
    if not isinstance(score, str):
        return ''
    return score.replace('—', '-').replace('–', '-').replace('−', '-')

# Match set tokens like "7-6(5)" or "6-4" anywhere in the score string; parentheses optional
SET_TOKEN_RE = re.compile(r'(\d+)\s*[-–-]\s*(\d+)(?:\s*$(\d+)$)?')

def _count_tb_oriented(score: str, wl_flag: str) -> tuple[int, int]:
    """
    Count tie-breaks from a score string, oriented by player result.
    If wl_flag == 'W', player's set games are the first number in each set token.
    If wl_flag == 'L', player's set games are the second number in each set token.
    A tiebreak set is strictly 7-6 or 6-7 (parentheses optional).
    """
    if not isinstance(score, str) or not score:
        return 0, 0
    s = _standardize_score_dashes(score)
    is_winner = str(wl_flag).upper() == 'W'
    tb_won = 0
    tb_lost = 0
    for a_str, b_str, _ in SET_TOKEN_RE.findall(s):
        try:
            a = int(a_str)
            b = int(b_str)
        except Exception:
            continue
        p_games = a if is_winner else b
        o_games = b if is_winner else a
        if p_games == 7 and o_games == 6:
            tb_won += 1
        elif p_games == 6 and o_games == 7:
            tb_lost += 1
    return tb_won, tb_lost

class TennisPlayersDirectory:
    """
    Fetch current ATP/WTA rankings from Tennis Abstract reports pages,
    cache them on disk, and serve suggestions for autocomplete.
    """
    
    def __init__(self, session: requests.Session | None = None, cache_expire_hours: int | None = None):
        self.session = session or requests.Session()
        self.cache_expire_hours = cache_expire_hours or CACHE_EXPIRE_HOURS
        
        # Cache file placed next to the database path if possible
        try:
            cache_dir = os.path.dirname(DATABASE_PATH) if DATABASE_PATH else os.path.dirname(__file__)
            if not cache_dir:
                cache_dir = "."
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_path = os.path.join(cache_dir, "players_cache.pkl")
            # Profile HTML cache
            self.profile_cache_dir = os.path.join(cache_dir, "cache", "profiles")
            os.makedirs(self.profile_cache_dir, exist_ok=True)
            self.profile_ttl_seconds = int(self.cache_expire_hours * 3600)
        except Exception:
            self.cache_path = os.path.join(os.path.dirname(__file__), "players_cache.pkl")
            self.profile_cache_dir = os.path.join(os.path.dirname(__file__), "cache", "profiles")
            try:
                os.makedirs(self.profile_cache_dir, exist_ok=True)
            except Exception:
                pass
            self.profile_ttl_seconds = int(CACHE_EXPIRE_HOURS * 3600)

    def _now(self) -> float:
        return time.time()
    @staticmethod
    def _normalize(s: str) -> str:
        if not isinstance(s, str):
            return ""
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        s = s.replace("\u00A0", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s
    @staticmethod
    def _canon_header(s: str) -> str:
        s = (s or "").replace("\u00A0", " ")
        s = re.sub(r"\s+", " ", s).strip().lower()
        return re.sub(r"[^a-z0-9]+", "", s)

    def _parse_date(self, text: str) -> datetime | None:
        if not text:
            return None
        s = self._normalize(text)
        # Prefer ISO if present
        for key in ("data-sort", "data-order"):
            if isinstance(text, dict) and text.get(key):
                s = self._normalize(text.get(key))
                break
        # Try common formats
        for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%B-%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y.%m.%d", "%d %b %Y", "%d %B %Y"):
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                pass
        # Fallback: find yyyy-mm-dd within
        m = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)
        if m:
            try:
                return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except Exception:
                return None
        return None

    @staticmethod
    def _compute_age(born: datetime | None) -> float | None:
        """Compute age with decimal precision (e.g., 23.4 years)"""
        if not born:
            return None
        today = dt_date.today()
        b = born.date()
        
        # Calculate years as a decimal
        years = today.year - b.year
        
        # Calculate the fraction of the year
        # Get the birthday this year
        try:
            birthday_this_year = dt_date(today.year, b.month, b.day)
        except ValueError:
            # Handle leap year edge case (Feb 29)
            birthday_this_year = dt_date(today.year, b.month, 28)
        
        if today >= birthday_this_year:
            # Birthday has passed this year
            days_since_birthday = (today - birthday_this_year).days
            days_in_year = 366 if today.year % 4 == 0 and (today.year % 100 != 0 or today.year % 400 == 0) else 365
            fraction = days_since_birthday / days_in_year
        else:
            # Birthday hasn't passed yet
            years -= 1
            try:
                birthday_last_year = dt_date(today.year - 1, b.month, b.day)
            except ValueError:
                # Handle leap year edge case
                birthday_last_year = dt_date(today.year - 1, b.month, 28)
            days_since_birthday = (today - birthday_last_year).days
            days_in_last_year = 366 if (today.year - 1) % 4 == 0 and ((today.year - 1) % 100 != 0 or (today.year - 1) % 400 == 0) else 365
            fraction = days_since_birthday / days_in_last_year
        
        age = years + fraction
        return round(age, 1)  # Round to 1 decimal place

    def _is_cache_valid(self, cache_obj: dict) -> bool:
        try:
            expires_at = cache_obj.get("expires_at")
            return bool(expires_at and self._now() < float(expires_at))
        except Exception:
            return False

    def _load_cache(self) -> dict | None:
        if not os.path.exists(self.cache_path):
            return None
        try:
            with open(self.cache_path, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict):
                return obj
        except Exception as e:
            logging.warning(f"Failed to load players cache: {e}")
        return None

    def _save_cache(self, data: dict) -> None:
        try:
            with open(self.cache_path, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            logging.warning(f"Failed to save players cache: {e}")

    def _fetch_rankings_html(self, tour: str) -> str:
        tour = (tour or "").upper()
        if tour not in ("ATP", "WTA"):
            raise ValueError("tour must be 'ATP' or 'WTA'")
        url = f"{TA_REPORTS_BASE}/{'atp' if tour=='ATP' else 'wta'}Rankings.html"
        try:
            resp = scraper._make_request(url)
        except Exception:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
        return resp.text

    @staticmethod
    def _extract_country_code(td) -> str | None:
        # try flag alt/title first
        img = td.find("img") if hasattr(td, "find") else None
        if img and (img.get("alt") or img.get("title")):
            alt = (img.get("alt") or img.get("title") or "").strip().upper()
            m = re.search(r"\b([A-Z]{3})\b", alt)
            if m:
                return m.group(1)
        # fallback to text
        text = ""
        try:
            text = td.get_text(" ", strip=True)
        except Exception:
            text = str(td) if td else ""
        text = (text or "").replace("\u00A0", " ")
        text = re.sub(r"\s+", " ", text).strip().upper()
        m = re.search(r"\b([A-Z]{3})\b", text)
        if m:
            return m.group(1)
        m2 = re.search(r"\b([A-Z]{2})\b", text)
        if m2:
            return m2.group(1)
        return None

    def _parse_players(self, html: str, tour: str) -> list[dict]:
        """
        Parse rankings table; extract rank, player name, country (3-letter code), birthdate if present, and age with decimal precision.
        """
        try:
            soup = BeautifulSoup(html, "html.parser")
            tables = soup.find_all("table")
            target = None
            headers = []
            for tbl in tables:
                ths = tbl.find_all("th")
                if not ths:
                    continue
                labels = [th.get_text(" ", strip=True) for th in ths]
                canon = [self._canon_header(x) for x in labels]
                if any("rank" in h or h == "rk" for h in canon) and any("player" in h or "name" in h for h in canon):
                    target = tbl
                    headers = labels
                    break
            if target is None and tables:
                target = tables[0]
                headers = [th.get_text(" ", strip=True) for th in target.find_all("th")]

            if target is None:
                logging.warning("No table found on rankings page.")
                return []

            canon_headers = [self._canon_header(h) for h in headers]
            # locate columns
            def find_idx(*keys):
                for k in keys:
                    for i, h in enumerate(canon_headers):
                        if k in h:
                            return i
                return None

            idx_rank = find_idx("rank", "rk")
            idx_name = find_idx("player", "name")
            idx_nat = find_idx("nat", "country", "nation", "ctry", "cntry")
            idx_birth = find_idx("birth", "dob", "born","birthdate")
            idx_age = find_idx("age")

            out: list[dict] = []
            for tr in target.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) < 2:
                    continue
                # rank
                rank_text = tds[idx_rank].get_text(strip=True) if idx_rank is not None and idx_rank < len(tds) else ""
                if not rank_text or not rank_text[0].isdigit():
                    continue
                try:
                    rank = int(re.sub(r"[^\d]", "", rank_text))
                except Exception:
                    continue
                # name
                name_cell = tds[idx_name] if idx_name is not None and idx_name < len(tds) else None
                if not name_cell:
                    continue
                a = name_cell.find("a") if hasattr(name_cell, "find") else None
                name_text = (a.get_text(strip=True) if a else name_cell.get_text(strip=True))
                name_text = self._normalize(name_text)  # convert NBSP, collapse spaces, trim

                # country
                country = None
                if idx_nat is not None and idx_nat < len(tds):
                    country = self._extract_country_code(tds[idx_nat])

                # birthdate / age
                birthdate_iso = None
                age_years = None
                if idx_birth is not None and idx_birth < len(tds):
                    cell = tds[idx_birth]
                    raw = cell.get("data-sort") or cell.get("data-order") or cell.get_text(" ", strip=True)
                    dt = self._parse_date(raw)
                    if not dt:
                        # try the visible text if data-sort didn't parse
                        dt = self._parse_date(cell.get_text(" ", strip=True))
                    if dt:
                        birthdate_iso = dt.strftime("%Y-%m-%d")
                        age_years = self._compute_age(dt)
                if age_years is None and idx_age is not None and idx_age < len(tds):
                    try:
                        age_txt = tds[idx_age].get_text(strip=True).replace("\u00A0", " ")
                        # Try to extract decimal age from text
                        m = re.search(r"(\d{1,2}(?:\.\d{1,2})?)", age_txt)
                        if m:
                            age_years = float(m.group(1))
                    except Exception:
                        pass

                out.append({
                    "rank": rank,
                    "name": name_text,
                    "tour": tour,
                    "country": country,
                    "birthdate": birthdate_iso,
                    "age": age_years,
                })

            out.sort(key=lambda r: (r["rank"], r["name"]))
            return out
        except Exception as e:
            logging.exception(f"Failed to parse players for tour={tour}: {e}")
            return []

    def get_players(self, tour: str | None = None, force_refresh: bool = False) -> dict | list[dict]:
        cache_obj = self._load_cache()
        if cache_obj and not force_refresh and self._is_cache_valid(cache_obj):
            if tour:
                return cache_obj.get(tour.upper(), [])
            return {"ATP": cache_obj.get("ATP", []), "WTA": cache_obj.get("WTA", [])}

        result = {"ATP": None, "WTA": None}
        for t in ("ATP", "WTA"):
            try:
                html = self._fetch_rankings_html(t)
                result[t] = self._parse_players(html, tour=t)
            except Exception as e:
                logging.warning(f"Error fetching {t} rankings: {e}")
                if cache_obj and cache_obj.get(t):
                    logging.warning(f"Using stale cached players for {t}")
                    result[t] = cache_obj.get(t, [])
                else:
                    result[t] = []

        if any(result[t] for t in ("ATP", "WTA")):
            expires_at = self._now() + float(self.cache_expire_hours) * 3600.0
            cache_to_save = {"ATP": result["ATP"], "WTA": result["WTA"], "expires_at": expires_at, "updated_at": self._now()}
            self._save_cache(cache_to_save)

        if tour:
            return result[tour.upper()]
        return result

    # --------- Player profile enrichment from player-classic page ---------
    def _fetch_player_profile_html(self, name: str) -> str | None:
        try:
            safe = re.sub(r'[^a-zA-Z0-9]+', '', name)
            path = os.path.join(self.profile_cache_dir, f"{safe}.html")
            # serve from cache if fresh
            if os.path.exists(path):
                try:
                    import time as _time
                    if _time.time() - os.path.getmtime(path) < self.profile_ttl_seconds:
                        with open(path, 'r', encoding='utf-8') as f:
                            return f.read()
                except Exception:
                    pass

            url_name = name.replace(' ', '')
            url = f"{BASE_URL}/cgi-bin/player-classic.cgi?p={url_name}"
            resp = scraper._make_request(url)
            html = resp.text
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(html)
            except Exception:
                pass
            return html
        except Exception as e:
            logging.warning(f"Failed to fetch profile page for {name}: {e}")
            return None

    @staticmethod
    def _parse_profile_fields(html: str) -> dict:
        """Parse current rank, peak rank, handedness/backhand, age, birthdate from the profile table snippet."""
        if not html:
            return {}
        soup = BeautifulSoup(html, 'html.parser')
        # First, try to parse from inline JS variables, which are reliable on TA pages
        js_text = html
        try:
            # currentrank, peakrank
            m_cur = re.search(r"var\s+currentrank\s*=\s*([0-9]+|""|'')\s*;", js_text)
            m_peak = re.search(r"var\s+peakrank\s*=\s*([0-9]+|""|'')\s*;", js_text)
            # dob as yyyymmdd
            m_dob = re.search(r"var\s+dob\s*=\s*([0-9]{8})\s*;", js_text)
            # hand 'L'/'R'
            m_hand = re.search(r"var\s+hand\s*=\s*'([LR])'\s*;", js_text)
            # backhand '1'/'2'
            m_bh = re.search(r"var\s+backhand\s*=\s*'([12])'\s*;", js_text)
            # wiki id like 'Yoshihito_Nishioka'
            m_wiki = re.search(r"var\s+wiki_id\s*=\s*'([^']+)'\s*;", js_text)
        except Exception:
            m_cur = m_peak = m_dob = m_hand = m_bh = m_wiki = None

        current_rank = None
        peak_rank = None
        handedness = None
        backhand_style = None
        age_years = None
        birthdate_iso = None

        if m_cur:
            try:
                val = m_cur.group(1)
                if val and val.strip('"\''):
                    current_rank = int(val)
            except Exception:
                pass
        if m_peak:
            try:
                val = m_peak.group(1)
                if val and val.strip('"\''):
                    peak_rank = int(val)
            except Exception:
                pass
        if m_dob:
            try:
                s = m_dob.group(1)
                birthdate_iso = f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
            except Exception:
                birthdate_iso = None
        if m_hand:
            handedness = 'Left' if m_hand.group(1) == 'L' else 'Right'
        if m_bh:
            backhand_style = 'Two-handed backhand' if m_bh.group(1) == '2' else 'One-handed backhand'
        wiki_id = m_wiki.group(1) if m_wiki else None

        # If some fields missing, fall back to textual parsing below
        raw_text = soup.get_text("\n", strip=True)
        # Normalize unicode spaces and dashes to improve regex robustness
        text = (raw_text
                .replace("\u00A0", " ")
                .replace("\u2011", "-")  # non-breaking hyphen
                .replace("\u2012", "-")
                .replace("\u2013", "-")
                .replace("\u2014", "-")
                .replace("\u2212", "-")
        )
        # Current rank (bold number inside may be adjacent)
        if current_rank is None:
            m = re.search(r"Current\s*rank\s*:\s*(?:<[^>]+>\s*)?\b(\d+)\b", text, re.IGNORECASE)
        else:
            m = None
        if m:
            try:
                current_rank = int(m.group(1))
            except Exception:
                pass
        # Peak rank / Career high rank (e.g., "Peak rank: 1 (10-Jun-2024)")
        if peak_rank is None:
            m = re.search(r"(Peak|Career\s*high)\s*rank\s*:\s*\b(\d+)\b", text, re.IGNORECASE)
        else:
            m = None
        if m:
            try:
                peak_rank = int(m.group(2))
            except Exception:
                pass
        # Plays: Right (two-handed backhand) — allow various hyphen characters
        if handedness is None or backhand_style is None:
            m = re.search(r"Plays\s*:\s*([A-Za-z]+)\s*\(([^)]+backhand)\)", text, re.IGNORECASE)
        else:
            m = None
        if m:
            handedness = m.group(1).capitalize()
            backhand_style = m.group(2)
        else:
            m2 = re.search(r"Plays\s*:\s*([A-Za-z]+)\b", text, re.IGNORECASE) if handedness is None else None
            if m2:
                handedness = m2.group(1).capitalize()
        # Normalize backhand style to expected labels
        if backhand_style:
            bl = backhand_style.lower()
            if re.search(r"two\s*[- ]?handed\s*backhand", bl):
                backhand_style = "Two-handed backhand"
            elif re.search(r"one\s*[- ]?handed\s*backhand", bl):
                backhand_style = "One-handed backhand"

        # Age: 24 (16-Aug-2001)
        m = None if birthdate_iso is not None else re.search(r"Age\s*:\s*([0-9]{1,2}(?:\.[0-9])?)\s*\(([^)]+)\)", text, re.IGNORECASE)
        if m:
            try:
                age_years = float(m.group(1))
            except Exception:
                pass
            # parse date
            raw = m.group(2)
            for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
                try:
                    birthdate_iso = datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
                    break
                except Exception:
                    continue
        else:
            # Sometimes shows only date of birth
            m3 = None if birthdate_iso is not None else re.search(r"\((\d{1,2}-[A-Za-z]{3,}-\d{4})\)", text)
            if m3:
                raw = m3.group(1)
                for fmt in ("%d-%b-%Y", "%d-%B-%Y"):
                    try:
                        birthdate_iso = datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
                        break
                    except Exception:
                        continue

        return {
            'current_rank': current_rank,
            'peak_rank': peak_rank,
            'handedness': handedness,
            'backhand_style': backhand_style,
            'age_years': age_years,
            'birthdate': birthdate_iso,
            'wiki_id': wiki_id
        }

    def enrich_and_save_players(self, tour: str | None = None, force_refresh: bool = False) -> dict | list[dict]:
        """Fetch rankings, enrich each player with profile fields, and upsert into SQLite."""
        data = self.get_players(tour=tour, force_refresh=force_refresh)
        if tour:
            players_lists = {tour.upper(): data}
        else:
            players_lists = data

        for t, players in players_lists.items():
            for p in players:
                name = p.get('name')
                try:
                    html = self._fetch_player_profile_html(name)
                    prof = self._parse_profile_fields(html) if html else {}
                    # Fallback to parsed rank/age/birthdate from rankings page if profile missing
                    if prof.get('current_rank') is None:
                        prof['current_rank'] = p.get('rank')
                    if prof.get('birthdate') is None and p.get('birthdate'):
                        prof['birthdate'] = p.get('birthdate')
                    if prof.get('age_years') is None and p.get('age') is not None:
                        try:
                            prof['age_years'] = float(p.get('age'))
                        except Exception:
                            pass
                    db.upsert_player_profile(name, prof)
                    # also attach to object returned
                    p.update({
                        'current_rank': prof.get('current_rank'),
                        'peak_rank': prof.get('peak_rank'),
                        'handedness': prof.get('handedness'),
                        'backhand_style': prof.get('backhand_style'),
                        'age': prof.get('age_years', p.get('age')),
                        'birthdate': prof.get('birthdate', p.get('birthdate')),
                    })
                except Exception as e:
                    logging.warning(f"Failed to enrich player {name}: {e}")
                    # still upsert minimal fields
                    db.upsert_player_profile(name, {
                        'current_rank': p.get('rank'),
                        'peak_rank': None,
                        'handedness': None,
                        'backhand_style': None,
                        'age_years': p.get('age'),
                        'birthdate': p.get('birthdate')
                    })

        return players_lists if not tour else players_lists.get(tour.upper(), [])

    # ------------------- Suggestions (unchanged) -------------------
    def suggest_players(self, query: str, limit: int = 10, tour: str | None = None) -> list[dict]:
        if not query or len(query.strip()) < 2:
            return []
        qn = re.sub(r"[^a-zA-Z0-9]+", "", unicodedata.normalize("NFKD", query.strip()).lower())
        data = self.get_players(tour=None) if tour is None else {tour.upper(): self.get_players(tour=tour)}
        candidates: list[dict] = []
        for t, arr in data.items():
            for item in arr:
                nm = item.get("name", "")
                nn = re.sub(r"[^a-zA-Z0-9]+", "", unicodedata.normalize("NFKD", nm).lower())
                idx = nn.find(qn)
                if idx != -1:
                    score = (idx, len(nn), item.get("rank") or 99999)
                    candidates.append({"name": nm, "tour": item.get("tour", t), "rank": item.get("rank"), "_score": score})
        candidates.sort(key=lambda x: x["_score"])
        out = [{"label": c["name"], "value": c["name"], "tour": c["tour"]} for c in candidates[: max(1, int(limit))]]
        return out

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
        """Make HTTP request with retry logic, jitter, and 429-aware backoff."""
        import random
        backoff = max(0.5, float(delay))
        for attempt in range(retries):
            try:
                time.sleep(backoff + random.uniform(0.1, 0.7))
                response = self.session.get(url, timeout=20)
                if response.status_code == 429:
                    retry_after = response.headers.get('Retry-After')
                    try:
                        wait = int(retry_after)
                    except Exception:
                        wait = max(60, int(backoff * 3))
                    logging.error(f"Attempt {attempt + 1} got 429 for {url}; sleeping {wait}s")
                    time.sleep(wait)
                    backoff = min(backoff * 2, 60)
                    continue
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                logging.error(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                if attempt == retries - 1:
                    raise
                sleep_s = min(max(5, int(backoff * 2 + random.uniform(0.5, 1.5))), 90)
                time.sleep(sleep_s)
                backoff = min(backoff * 2, 60)

    def get_player_matches(self, player_name, force_refresh=False):
        """Get all matches for a player with database caching and tb backfill"""
        if not force_refresh:
            cached_data = db.get_player_matches(player_name)
            if cached_data is not None:
                logging.info(f"Using database cached data for {player_name}")
                if 'tb_won' not in cached_data.columns or 'tb_lost' not in cached_data.columns:
                    try:
                        cached_data = self._add_tb_columns(cached_data)
                        try:
                            db.cache_player_matches(player_name, cached_data)
                        except Exception:
                            pass
                    except Exception as e:
                        logging.warning(f"Failed to backfill tb columns for cached data: {e}")
                return cached_data

        data = self._fetch_player_matches(player_name)
        if data is not None:
            db.cache_player_matches(player_name, data)
        return data

    def _add_tb_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add tb_won/tb_lost to an existing DataFrame if missing, using WL-oriented parsing."""
        if df is None or df.empty:
            return df
        if 'tb_won' in df.columns and 'tb_lost' in df.columns:
            return df

        def _safe_count(row):
            sc = row.get('score', '')
            wl = row.get('wl', '')
            # Ignore W/O and empties
            if sc in ['W/O', '', None] or pd.isna(sc):
                return (0, 0)
            return _count_tb_oriented(str(sc), str(wl))

        counts = df.apply(_safe_count, axis=1)
        df = df.copy()
        df[['tb_won', 'tb_lost']] = pd.DataFrame(counts.tolist(), index=df.index).astype(int)
        return df

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
        """Create a DataFrame from matches data and compute tb_won/tb_lost at ingestion (WL-oriented)."""
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
        df = df[df['score'].notna()].copy()

        # Compute oriented tiebreak counts per match
        tb_counts = df.apply(lambda r: _count_tb_oriented(str(r['score']), r['wl']), axis=1)
        df[['tb_won', 'tb_lost']] = pd.DataFrame(tb_counts.tolist(), index=df.index).astype(int)

        df = df.sort_values('date')
        return df


class TennisStatsCalculator:
    @staticmethod
    def _pct(numer: int, denom: int) -> float:
        try:
            return round(100.0 * numer / denom, 1) if denom and denom > 0 else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _sum_or_zero(df, cols):
        present = [c for c in cols if c in df.columns]
        if not present:
            return pd.DataFrame(index=df.index)
        return df[present]

    @staticmethod
    def calculate_yearly_stats(df, surface=None):
        """Calculate tennis statistics by year and optionally by surface, including tiebreak stats."""
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

        # Ensure tb columns exist (fallback if needed)
        if 'tb_won' not in valid_df.columns or 'tb_lost' not in valid_df.columns:
            tb_counts = valid_df.apply(lambda r: _count_tb_oriented(str(r.get('score', '')), r.get('wl', '')), axis=1)
            valid_df[['tb_won', 'tb_lost']] = pd.DataFrame(tb_counts.tolist(), index=valid_df.index).astype(int)

        # Win-loss record for all valid matches
        wl_record = valid_df.groupby('year')['wl'].agg(
            wins=lambda x: (x == 'W').sum(),
            losses=lambda x: (x == 'L').sum()
        )
        wl_record['total'] = wl_record['wins'] + wl_record['losses']
        wl_record['win%'] = (wl_record['wins'] / wl_record['total'] * 100).round(1)
        wl_record['W-L'] = wl_record.apply(lambda x: f"{int(x['wins'])}-{int(x['losses'])}", axis=1)

        # Filter for matches with detailed point stats
        with_pts = valid_df[valid_df['pts'].notna()]

        # Calculate yearly sums (for detailed stats subset)
        yearly_sums = with_pts.groupby('year')[numeric_columns].agg({
            'aces': 'sum', 'dfs': 'sum', 'pts': 'sum', 'firsts': 'sum',
            'fwon': 'sum', 'swon': 'sum', 'saved': 'sum', 'chances': 'sum',
            'games': 'sum', 'ogames': 'sum', 'osaved': 'sum', 'ochances': 'sum',
            'orank': 'mean'
        }).fillna(0)

        # Build result frame indexed by all years where matches occurred
        years_index = wl_record.index.union(yearly_sums.index).sort_values()
        yearly_stats = pd.DataFrame(index=years_index)

        # Basic records
        yearly_stats['W-L'] = wl_record['W-L'].reindex(yearly_stats.index)
        yearly_stats['win%'] = wl_record['win%'].reindex(yearly_stats.index)

        # Reindex sums to align
        ys = yearly_sums.reindex(yearly_stats.index).fillna(0)

        # Percentages from detailed stats
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

        yearly_stats['avg_opp_rank'] = ys['orank'].round(1).fillna(0)

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

        # Tiebreak records (use precomputed columns)
        tb_year = valid_df.groupby('year')[['tb_won', 'tb_lost']].sum().reindex(yearly_stats.index).fillna(0)
        tb_totals = tb_year['tb_won'] + tb_year['tb_lost']
        with np.errstate(divide='ignore', invalid='ignore'):
            tb_winp = (tb_year['tb_won'] / tb_totals * 100).round(1)

        # Include numeric tb columns and readable strings
        yearly_stats['tb_won'] = tb_year['tb_won'].astype(int)
        yearly_stats['tb_lost'] = tb_year['tb_lost'].astype(int)
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
        """Calculate career statistics, including tiebreak stats and Top-N opponent records"""
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
        stats_df = stats_df[stats_df['score'].notna()].copy()

        # Ensure tb columns exist (fallback if needed)
        if 'tb_won' not in stats_df.columns or 'tb_lost' not in stats_df.columns:
            tb_counts = stats_df.apply(lambda r: _count_tb_oriented(str(r.get('score', '')), r.get('wl', '')), axis=1)
            stats_df[['tb_won', 'tb_lost']] = pd.DataFrame(tb_counts.tolist(), index=stats_df.index).astype(int)

        total_matches = len(stats_df)
        if total_matches == 0:
            return pd.DataFrame()

        wins = int((stats_df['wl'] == 'W').sum())
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

        # Tiebreak totals across all valid matches (not only those with detailed stats)
        tb_won_total = int(stats_df['tb_won'].sum())
        tb_lost_total = int(stats_df['tb_lost'].sum())
        tb_total = tb_won_total + tb_lost_total
        tb_winp = round(tb_won_total / tb_total * 100, 1) if tb_total > 0 else 0.0

        career_stats = {
            'W-L': wl_record,
            'win%': win_pct,
            'ace%': (career_sums['aces'] / career_sums['pts'] * 100).round(2) if career_sums['pts'] > 0 else 0,
            'df%': (career_sums['dfs'] / career_sums['pts'] * 100).round(2) if career_sums['pts'] > 0 else 0,
            '1st_in%': (career_sums['firsts'] / career_sums['pts'] * 100).round(1) if career_sums['pts'] > 0 else 0,
            '1st_win%': (career_sums['fwon'] / career_sums['firsts'] * 100).round(1) if career_sums['firsts'] > 0 else 0,
            '2nd_win%': (career_sums['swon'] / (career_sums['pts'] - career_sums['firsts']) * 100).round(1) if (career_sums['pts'] - career_sums['firsts']) > 0 else 0,
            'bp_saved%': (career_sums['saved'] / career_sums['chances'] * 100).round(1) if career_sums['chances'] > 0 else 0,
            'hold%': (100 - ((career_sums['chances'] - career_sums['saved']) / career_sums['games'] * 100)).round(1) if career_sums['games'] > 0 else 0,
            'break%': ((career_sums['ochances'] - career_sums['osaved']) / career_sums['ogames'] * 100).round(1) if career_sums['ogames'] > 0 else 0,
            'avg_opp_rank': round(career_sums['orank'], 1) if not np.isnan(career_sums['orank']) else 0,
            'matches_with_stats%': stats_pct,
            # Tiebreak outputs
            'tb_won': tb_won_total,
            'tb_lost': tb_lost_total,
            'tb_W-L': f"{tb_won_total}-{tb_lost_total}",
            'tb_win%': tb_winp
        }

        # Top-N opponent records (ignore matches with missing orank)
        orank_num = pd.to_numeric(stats_df['orank'], errors='coerce')
        for top_n in (5, 10, 20, 50, 100):
            sub = stats_df[orank_num <= top_n]
            w = int((sub['wl'] == 'W').sum())
            l = int((sub['wl'] == 'L').sum())
            t = w + l
            wl = f"{w}-{l}"
            wp = round((w / t * 100), 1) if t > 0 else 0.0
            career_stats[f'top{top_n}_W-L'] = wl
            career_stats[f'top{top_n}_win%'] = wp

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
        Return per-year tiebreak W-L and win% based on set scores (7-6/6-7),
        using tb_won/tb_lost columns if present.
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

        if 'tb_won' not in data.columns or 'tb_lost' not in data.columns:
            tb_counts = data.apply(lambda r: _count_tb_oriented(str(r.get('score', '')), r.get('wl', '')), axis=1)
            data[['tb_won', 'tb_lost']] = pd.DataFrame(tb_counts.tolist(), index=data.index).astype(int)

        out = data.groupby('year')[['tb_won', 'tb_lost']].sum().sort_index()
        totals = out['tb_won'] + out['tb_lost']
        with np.errstate(divide='ignore', invalid='ignore'):
            winp = (out['tb_won'] / totals * 100).round(1)

        result = pd.DataFrame(index=out.index)
        result['tb_won'] = out['tb_won'].astype(int)
        result['tb_lost'] = out['tb_lost'].astype(int)
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
players_dir = TennisPlayersDirectory(session=scraper.session)

# Convenience functions for backward compatibility
def get_player_matches(player_name, force_refresh=False):
    return scraper.get_player_matches(player_name, force_refresh=force_refresh)

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
    
# Initialize or update global instance (reuse scraper session if it exists)
try:
    players_dir
except NameError:
    players_dir = TennisPlayersDirectory(session=scraper.session if 'scraper' in globals() else None)

# Convenience functions
def get_players(tour: str | None = None, force_refresh: bool = False):
    return players_dir.get_players(tour=tour, force_refresh=force_refresh)

def suggest_players(query: str, limit: int = 10, tour: str | None = None):
    return players_dir.suggest_players(query=query, limit=limit, tour=tour)

# Example usage (optional)
if __name__ == '__main__':
    try:
        player = "Iga Swiatek"
        matches_df = get_player_matches(player)
        print(f"Fetched matches for {player}. Columns: {list(matches_df.columns) if matches_df is not None else 'No data'}")

        ys = calculate_yearly_stats(matches_df)
        print("\nYearly stats (tail):")
        print(ys.tail(3))

        tb_yearly = calculate_tiebreak_records(matches_df)
        print("\nTiebreak records per year (tail):")
        print(tb_yearly.tail(3))

        car = career(player)
        if car is not None and 'career' in car.index:
            print("\nCareer stats (selected columns):")
            print(car.loc['career', ['W-L', 'win%', 'tb_won', 'tb_lost', 'tb_W-L', 'tb_win%']])

    except Exception as e:
        logging.error(f"Example usage error: {e}")
