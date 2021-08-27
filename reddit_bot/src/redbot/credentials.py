"""Module to retrieve Reddit API credentials."""
import json
import os

from google.oauth2 import service_account


REDDIT_CREDENTIALS_DEFAULT_PATH = '~/.redbot/client_secret.json'
GCP_CREDENTIALS_DEFAULT_PATH = '~/.redbot/redbot-gcp.json'


def load_reddit_credentials(file_path=None):
    """Return saved Reddit credentials.

    Args:
        file_path: str. Path where Reddit credentials are stored. If None, use REDDIT_CREDENTIALS_DEFAULT_PATH.

    Returns:
        creds: dict. The credentials.
    """
    if not file_path:
        file_path = REDDIT_CREDENTIALS_DEFAULT_PATH
    file_path = os.path.expanduser(file_path)
    with open(file_path) as fid:
        return json.load(fid)


def load_gcp_credentials(file_path=None):
    """Return saved GCP credentials.

    Args:
        file_path: str. Path where GCP credentials are stored. If None, use GCP_CREDENTIALS_DEFAULT_PATH.

    Returns:
        creds: dict. The credentials.
    """
    if not file_path:
        file_path = GCP_CREDENTIALS_DEFAULT_PATH
    file_path = os.path.expanduser(GCP_CREDENTIALS_DEFAULT_PATH)
    credentials = service_account.Credentials.from_service_account_file(file_path, scopes=[
            'https://www.googleapis.com/auth/cloud-platform'])
    return credentials
