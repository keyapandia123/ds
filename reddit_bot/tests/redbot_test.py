"""Basic unit tests."""
import collections
from datetime import datetime
import os
import tempfile
import time

import pandas as pd  # TODO: Remove this dependency
import pytest

from redbot import credentials
from redbot import db


# TODO: Move to db module.
Post = collections.namedtuple(
    'Post', ['id', 'url', 'title', 'score', 'upvote_ratio', 'created_utc']
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
        ll = list(cursor.execute(sql))
        true_cols = ['id', 'uid', 'url', 'title', 'score', 'upvote_ratio',
                     'highrank24', 'created_utc', 'time_highrank', 'subreddit',
                     'prediction']
        assert len(ll) == len(true_cols)
        for c1, c2 in zip(ll, true_cols):
            assert c1[0] == c2


def test_db_creation_exception():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, 'temp_db.sqlite')
        with pytest.raises(db.DatabaseNotFoundError):
            db.connect_to_db(temp_path, False)


def test_db_insertion_and_update():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, 'temp_db.sqlite')
        con = db.connect_to_db(temp_path, create_if_empty=True)
        current_time = time.time()
        current_time_utc = datetime.utcfromtimestamp(current_time)
        new_post = Post('uid', 'url', 'title', 5, 0.9, current_time)
        high_rank = 4
        time_highrank = current_time_utc
        subreddit = 'politics'
        prediction = None
        db.insert_new_post(con, new_post, high_rank, time_highrank, subreddit, prediction)
        saved_post = con.cursor().execute("select * from posts").fetchall()
        assert len(saved_post) == 1
        saved_post = saved_post[0]
        assert saved_post[1] == 'uid'
        assert saved_post[2] == 'url'
        assert saved_post[3] == 'title'
        assert saved_post[4] == 5
        assert saved_post[5] == 0.9
        assert saved_post[6] == 4
        assert saved_post[7] == current_time_utc
        assert saved_post[8] == current_time_utc
        assert saved_post[9] == 'politics'
        assert saved_post[10] is None

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
