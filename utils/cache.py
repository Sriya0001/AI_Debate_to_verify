import os
import sqlite3
import hashlib
import time
import logging
import threading
from typing import Optional, Dict, Any

from config import config

logger = logging.getLogger(__name__)


class LLMResponseCache:
    """
    Persistent SQLite-backed cache for LLM responses.
    Keyed on (model_name, prompt_text, seed, temperature).
    Guarantees deterministic, instant replays and resumable execution.
    """
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or getattr(config, "llm_cache_path", ".cache/llm_cache.sqlite")
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._init_db()

    def _init_db(self):
        dir_name = os.path.dirname(os.path.abspath(self.db_path))
        os.makedirs(dir_name, exist_ok=True)
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS response_cache (
                        cache_key TEXT PRIMARY KEY,
                        model_name TEXT NOT NULL,
                        prompt_text TEXT NOT NULL,
                        seed INTEGER,
                        temperature REAL NOT NULL,
                        response_text TEXT NOT NULL,
                        prompt_tokens INTEGER DEFAULT 0,
                        completion_tokens INTEGER DEFAULT 0,
                        latency REAL DEFAULT 0.0,
                        created_at REAL NOT NULL
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_model ON response_cache(model_name)")
                conn.commit()

    @staticmethod
    def compute_key(model_name: str, prompt_text: str, seed: Optional[int], temperature: float) -> str:
        s_val = str(seed) if seed is not None else "none"
        raw = f"{model_name.strip()}|{prompt_text.strip()}|{s_val}|{float(temperature):.4f}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(
        self,
        model_name: str,
        prompt_text: str,
        seed: Optional[int] = None,
        temperature: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """Retrieves cached response if present."""
        if not getattr(config, "llm_cache_enabled", True):
            return None

        cache_key = self.compute_key(model_name, prompt_text, seed, temperature)
        with self._lock:
            try:
                with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT response_text, prompt_tokens, completion_tokens, latency FROM response_cache WHERE cache_key = ?",
                        (cache_key,)
                    )
                    row = cursor.fetchone()
                    if row:
                        self._hits += 1
                        return {
                            "response_text": row[0],
                            "prompt_tokens": row[1],
                            "completion_tokens": row[2],
                            "latency": row[3],
                            "cached": True,
                        }
                    else:
                        self._misses += 1
                        return None
            except Exception as e:
                logger.warning("Cache lookup error for key %s: %s", cache_key[:8], e)
                return None

    def set(
        self,
        model_name: str,
        prompt_text: str,
        response_text: str,
        seed: Optional[int] = None,
        temperature: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency: float = 0.0,
    ):
        """Stores response and token telemetry in cache."""
        if not getattr(config, "llm_cache_enabled", True):
            return

        cache_key = self.compute_key(model_name, prompt_text, seed, temperature)
        with self._lock:
            try:
                with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO response_cache
                        (cache_key, model_name, prompt_text, seed, temperature, response_text, prompt_tokens, completion_tokens, latency, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        cache_key,
                        model_name.strip(),
                        prompt_text.strip(),
                        seed,
                        float(temperature),
                        response_text,
                        int(prompt_tokens),
                        int(completion_tokens),
                        float(latency),
                        time.time(),
                    ))
                    conn.commit()
            except Exception as e:
                logger.warning("Cache write error for key %s: %s", cache_key[:8], e)

    def clear(self):
        """Clears all entries in the cache."""
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                conn.execute("DELETE FROM response_cache")
                conn.commit()
        self._hits = 0
        self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """Returns statistics on cache usage."""
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT count(*) FROM response_cache")
                count = cursor.fetchone()[0]
        return {
            "total_entries": count,
            "hits": self._hits,
            "misses": self._misses,
            "hit_ratio": round(self._hits / (self._hits + self._misses), 3) if (self._hits + self._misses) > 0 else 0.0,
        }


# Global singleton cache instance
llm_cache = LLMResponseCache()
