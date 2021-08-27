"""Basic unit tests."""
import collections
from datetime import datetime, timedelta
import os
import tempfile
import time

import numpy as np
import pandas as pd
import pytest

from redbot import analysis
from redbot import credentials
from redbot import db
from redbot import main
from redbot import train


# TODO: Move to db module.
Post = collections.namedtuple(
    'Post', ['id', 'url', 'title', 'score', 'upvote_ratio', 'created_utc']
)


def test_credentials():
    creds = credentials.load_reddit_credentials()
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
        true_cols = ['uuid', 'uid', 'url', 'title', 'score', 'upvote_ratio',
                     'highrank24', 'created_utc', 'time_highrank', 'subreddit',
                     'prediction', 'db_version']
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
        new_post = Post('uid', 'url', 'title', np.int64(5), np.float(0.9), current_time)
        high_rank = 4
        time_highrank = current_time_utc
        subreddit = 'politics'
        prediction = None
        con.commit()
        con = db.connect_to_db(temp_path, create_if_empty=False)
        db.insert_new_post(con, new_post, np.int64(high_rank), time_highrank, subreddit, prediction, 'uuid1')
        con = db.connect_to_db(temp_path, create_if_empty=False)
        saved_post = con.cursor().execute("select * from posts").fetchall()
        assert len(saved_post) == 1
        saved_post = saved_post[0]
        assert saved_post[0] == 'uuid1'
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
        assert saved_post[11] == db.DB_VERSION

        list_uids = db.get_uids(con)
        assert len(list_uids) == 1
        assert list_uids[0] == 'uid'

        con.commit()
        con = db.connect_to_db(temp_path, create_if_empty=False)
        current_high_rank = db.get_highrank(con, 'uid')
        assert current_high_rank == 4

        new_high_rank = 0
        new_time_high_rank = time.time()
        new_time_high_rank_utc = datetime.utcfromtimestamp(new_time_high_rank)
        con.commit()
        con = db.connect_to_db(temp_path, create_if_empty=False)
        db.update_highrank(con, np.int64(new_high_rank), new_time_high_rank_utc, 'uid')
        con = db.connect_to_db(temp_path, create_if_empty=False)
        saved_high_rank = db.get_highrank(con, 'uid')
        assert saved_high_rank == new_high_rank
        cursor = con.cursor()
        sql = """
        SELECT time_highrank FROM posts WHERE uid=?
        """
        saved_time_gen = cursor.execute(sql, ('uid',))
        assert list(saved_time_gen)[0][0] == new_time_high_rank_utc

        new_score = 25
        con.commit()
        con = db.connect_to_db(temp_path, create_if_empty=False)
        db.update_score(con, np.int64(new_score), 'uid')
        con = db.connect_to_db(temp_path, create_if_empty=False)
        cursor = con.cursor()
        sql = """
        SELECT score FROM posts WHERE uid=?
        """
        saved_score_gen = cursor.execute(sql, ('uid',))
        assert list(saved_score_gen)[0][0] == new_score

        new_pred = 1
        con.commit()
        con = db.connect_to_db(temp_path, create_if_empty=False)
        db.update_prediction(con, np.int64(new_pred), 'uuid1')
        con = db.connect_to_db(temp_path, create_if_empty=False)
        cursor = con.cursor()
        sql = """
        SELECT prediction FROM posts WHERE uuid='uuid1'
        """
        saved_pred_gen = cursor.execute(sql)
        assert list(saved_pred_gen)[0][0] == new_pred

        new_post = Post('uid', 'url', 'title', np.int64(5), np.float(0.9), current_time)
        high_rank = None
        time_highrank = None
        subreddit = 'politics'
        prediction = None
        con.commit()
        con = db.connect_to_db(temp_path, create_if_empty=False)
        db.insert_new_post(con, new_post, high_rank, time_highrank, subreddit, prediction, 'uuid2')
        con = db.connect_to_db(temp_path, create_if_empty=False)
        df = pd.read_sql("select * from posts", con)
        assert df['time_highrank'].dtype.name == 'datetime64[ns]'
        assert df['highrank24'].dtype.name == 'float64'


def test_analyze():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, 'temp_db.sqlite')
        con = db.connect_to_db(temp_path, create_if_empty=True)
        current_time = time.time()
        current_time_utc = datetime.utcfromtimestamp(current_time)

        valid_hot_post_1 = Post('uid1', 'https://www.politico.com', 'Hot Post 1', 500, 0.9, current_time - 30 * 3600)
        high_rank = 4
        time_highrank = current_time_utc - timedelta(hours=20, minutes=00)
        subreddit = 'politics'
        prediction = None
        db.insert_new_post(con, valid_hot_post_1, high_rank, time_highrank, subreddit, prediction, 'uuid1')

        valid_hot_post_2 = Post('uid2', 'https://politico.com', 'Hot Post 2', 450, 0.9, current_time - 32 * 3600)
        high_rank = 3
        time_highrank = current_time_utc - timedelta(hours=30, minutes=00)
        subreddit = 'politics'
        prediction = None
        db.insert_new_post(con, valid_hot_post_2, high_rank, time_highrank, subreddit, prediction, 'uuid2')

        valid_hot_post_3 = Post('uid3', 'https://washingtonpost.com', 'Hot Post 3', 550, 0.9, current_time - 28 * 3600)
        high_rank = 7
        time_highrank = current_time_utc - timedelta(hours=26, minutes=00)
        subreddit = 'politics'
        prediction = None
        db.insert_new_post(con, valid_hot_post_3, high_rank, time_highrank, subreddit, prediction, 'uuid3')

        invalid_hot_post_4 = Post('uid4', 'https://cnn.com', 'Invalid Hot Post 4', 700, 0.9, current_time - 12 * 3600)
        high_rank = 3
        time_highrank = current_time_utc - timedelta(hours=10, minutes=00)
        subreddit = 'politics'
        prediction = None
        db.insert_new_post(con, invalid_hot_post_4, high_rank, time_highrank, subreddit, prediction, 'uuid4')

        valid_non_hot_post_1 = Post('uid5', 'https://businessinsider.com', 'Non-hot Post 1', 10, 0.9, current_time - 30 * 3600)
        high_rank = None
        time_highrank = None
        subreddit = 'politics'
        prediction = None
        db.insert_new_post(con, valid_non_hot_post_1, high_rank, time_highrank, subreddit, prediction, 'uuid5')

        valid_non_hot_post_2 = Post('uid6', 'http://www.businessinsider.com', 'Non-hot Post 2', 15, 0.9, current_time - 28 * 3600)
        high_rank = None
        time_highrank = None
        subreddit = 'politics'
        prediction = None
        db.insert_new_post(con, valid_non_hot_post_2, high_rank, time_highrank, subreddit, prediction, 'uuid6')

        invalid_non_hot_post_3 = Post('uid7', 'http://www.independent.co.uk', 'Non-hot Post 3', 20, 0.9, current_time)
        high_rank = None
        time_highrank = None
        subreddit = 'politics'
        prediction = None
        db.insert_new_post(con, invalid_non_hot_post_3, high_rank, time_highrank, subreddit, prediction, 'uuid7')

        total_post_cnt, total_hot_post_cnt, total_valid_post_cnt, total_hot_valid_post_cnt = analysis.return_post_counts(con)

        [mean_hot, min_hot, max_hot], [mean_non_hot, min_non_hot, max_non_hot] = analysis.return_scores(con)

        analysis.title_keywords(con)

        analysis.domains(con)


def test_run_analysis():
    con = db.connect_to_db(main.DATABASE_DEFAULT_PATH)

    total_post_cnt, total_hot_post_cnt, total_valid_post_cnt, total_hot_valid_post_cnt = analysis.return_post_counts(con)

    [mean_hot, min_hot, max_hot], [mean_non_hot, min_non_hot, max_non_hot] = analysis.return_scores(con)

    analysis.title_keywords(con)

    analysis.domains(con)


def test_run_train():
    con = db.connect_to_db(main.DATABASE_DEFAULT_PATH)
    train.run_and_evaluate_training(con)


def test_analyze_inference():
    con = db.connect_to_db(main.DATABASE_DEFAULT_PATH)
    analysis.calculate_recall(con, win_hr=5.0)