"""Basic unit tests."""
from redbot import credentials


def test_credentials():
    creds = credentials.load_credentials()
    assert 'client_id' in creds
    assert 'client_secret' in creds
    assert 'user_agent' in creds