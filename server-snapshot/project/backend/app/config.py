from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

PROJECT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT / '.env')


@dataclass
class Settings:
    storage: Path = PROJECT / 'storage'
    remote_url: str = os.getenv('CAD_REMOTE_URL', '').strip().rstrip('/')
    remote_token: str = os.getenv('CAD_REMOTE_TOKEN', '')
    remote_timeout: int = int(os.getenv('CAD_REMOTE_TIMEOUT', '600'))
    max_upload: int = 20 * 1024 * 1024

    def __post_init__(self):
        if self.remote_url:
            from urllib.parse import urlsplit
            u = urlsplit(self.remote_url)
            if u.scheme not in ('http', 'https') or not u.hostname or u.username or u.password or u.query or u.fragment:
                raise ValueError('CAD_REMOTE_URL must be an http(s) base URL without credentials or query')
        self.storage.mkdir(parents=True, exist_ok=True)
