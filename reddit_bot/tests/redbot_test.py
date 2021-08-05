"""Basic unit tests."""
import collections
import os
import tempfile

import pandas as pd  # TODO: Remove this dependency
import pytest

from redbot import credentials
from redbot import db


# TODO: Move to db module.
Post = collections.namedtuple(
    'Post', ['id', 'url', 'title', 'score']
)


def test_credentials():
    creds = credentials.load_credentials()
    assert 'client_id' in creds
    assert 'client_secret' in creds
    assert 'user_agent' in creds


def test_db_creation():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, 'temp_db.sqlite')
        con = db.connect_to_db(temp_path, True)
        cursor = con.cursor()
        sql = """
        SELECT name FROM PRAGMA_TABLE_INFO('posts')
        """
        ll = cursor.execute(sql)
        assert len(list(ll)) > 0


def test_db_creation_exception():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, 'temp_db.sqlite')
        with pytest.raises(db.DatabaseNotFoundError):
            db.connect_to_db(temp_path, False)


def test_db_insertion_and_update():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, 'temp_db.sqlite')
        con = db.connect_to_db(temp_path, create_if_empty=True)
        new_post = Post('uid', 'url', 'title', 5)
        high_rank = 4
        db.insert_new_post(con, new_post, high_rank)
        saved_post = pd.read_sql_query('''SELECT * FROM posts''', con)
        assert saved_post.loc[0]['uid'] == 'uid'
        assert saved_post.loc[0]['url'] == 'url'
        assert saved_post.loc[0]['title'] == 'title'
        assert saved_post.loc[0]['score'] == 5
        assert saved_post.loc[0]['highrank24'] == 4

        list_uids = db.get_uids(con)
        assert len(list_uids) == 1
        assert list_uids[0] == 'uid'

        current_high_rank = db.get_highrank(con, 'uid')
        assert current_high_rank[0] == 4

        new_high_rank = 0
        db.update_highrank(con, new_high_rank, 'uid')
        saved_high_rank = db.get_highrank(con, 'uid')
        assert saved_high_rank[0] == new_high_rank

        new_score = 25
        db.update_score(con, new_score, 'uid')
        cursor = con.cursor()
        sql = """
        SELECT score FROM posts WHERE uid=?
        """
        saved_score_gen = cursor.execute(sql, ('uid',))
        assert list(saved_score_gen)[0][0] == new_score
