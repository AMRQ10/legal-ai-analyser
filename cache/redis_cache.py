import redis
import json
import hashlib
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class RedisCache:
    def __init__(self):
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.client = redis.from_url(redis_url, decode_responses=True)
        self.default_ttl = 3600

    def _make_key(self, prefix: str, content: str) -> str:
            """
            Creates a cache key by hashing the content.
            Same content always produces same key. 
            Hash keeps keys short regardless of content length.
            """
            content_hash = hashlib.md5(content.encode()).hexdigest()
            return f"{prefix}:{content_hash}"

    def get(self, prefix: str, content: str) -> Optional[dict]:
            """Returns cached result if it exists, None if not."""
            key = self._make_key(prefix, content)
            try:
                cached = self.client.get(key)
                if cached:
                    return json.loads(cached)
                return None
            except redis.RedisError:
                return None

    def set(self, prefix: str, content: str, result: dict, ttl: Optional[int] = None) -> None:
            """Stors a result in cache with expiry time."""
            key = self._make_key(prefix, content)
            try:
                self.client.setex(
                    key,
                    ttl or self.default_ttl,
                    json.dumps(result)
                )
            except redis.RedisError:
                pass

    def invalidate(self, prefix: str, content: str) -> None:
            """Removes a specific cached result."""
            key = self._make_key(prefix, content)
            try:
                self.client.delete(key)
            except redis.RedisError:
                pass

    def is_available(self) -> bool:
            """Checks if Redis is reachable."""
            try:
                self.client.ping()
                return True
            except redis.RedisError:
                return False
            