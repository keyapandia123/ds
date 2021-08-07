"""Module to retrieve Reddit API credentials."""
import json
import os


CREDENTIALS_DEFAULT_PATH = '~/.redbot/client_secret.json'


def load_credentials(file_path=None):
    """Return saved credentials.

    Args:
        file_path: str. Path where credentials are stored. If None, use CREDENTIALS_DEFAULT_PATH.

    Returns:
        creds: dict. The credentials.
    """
    if not file_path:
        file_path = CREDENTIALS_DEFAULT_PATH
    file_path = os.path.expanduser(file_path)
    with open(file_path) as fid:
        return json.load(fid)
