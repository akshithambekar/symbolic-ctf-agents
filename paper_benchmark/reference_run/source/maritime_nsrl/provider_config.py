"""Small, non-executing local configuration reader. Never logs credentials."""
import os
from pathlib import Path
from urllib.parse import urlparse


def load_env(path=Path('.env')):
    allowed = {'OPENROUTER_API_KEY', 'OPENROUTER_MODEL', 'LTL_BASE_URL', 'LTL_API_KEY'}
    if not Path(path).exists():
        return
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        key, sep, value = line.strip().partition('=')
        if sep and key in allowed:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in '\"\'':
                value = value[1:-1]
            if value:
                os.environ.setdefault(key, value)


def api_key(endpoint):
    if urlparse(endpoint).hostname == 'openrouter.ai':
        return os.environ.get('OPENROUTER_API_KEY')
    return os.environ.get('LTL_API_KEY') or os.environ.get('OPENAI_API_KEY')
