import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    env: str = os.getenv("QT_ENV", "dev")
    # Where the real engine would live (if you do HTTP-based separation)
    engine_base_url: str = os.getenv("ENGINE_BASE_URL", "http://engine:9000")
    # If you later use Redis pub/sub
    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379/0")

settings = Settings()