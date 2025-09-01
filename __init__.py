"""Tennis Abstract package initializer.

Exposes common scraper APIs for external scripts.
"""

from .database import TennisDatabase
from .mod import db, get_player_matches, get_players, players_dir  # noqa: F401
