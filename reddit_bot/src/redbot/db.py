"""Module to handle database operations."""
from datetime import datetime
import os
import sqlite3

from google.cloud import bigquery
from google.cloud.bigquery import dbapi


DB_VERSION = 'v2'


class DatabaseNotFoundError(Exception):
    """Raised when database is not found."""


def create_schema(con):
    """Define Schema.

    Args:
        con: database connection.
    """
    cursor = con.cursor()
    cursor.execute("""
        CREATE TABLE posts(
            uuid TEXT, 
            uid TEXT, 
            url TEXT, 
            title TEXT, 
            score INTEGER,
            upvote_ratio FLOAT,
            highrank24 INTEGER,
            created_utc TIMESTAMP,
            time_highrank TIMESTAMP, 
            subreddit TEXT, 
            prediction INTEGER,
            db_version TEXT
            )
        """)
    con.commit()


def connect_to_db(db_path, create_if_empty=False):
    """Connect to existing or new database.

    Checks if database exists. If not, creates a new database.

    Args:
        db_path: str. Path where database is stored.
        create_if_empty: bool. Create new database if True.

    Returns:
         con: database connection

    Raises:
        DatabaseNotFoundError
    """
    db_path = os.path.expanduser(db_path)

    if not os.path.exists(db_path) and not create_if_empty:
        msg = f'Database not found at {db_path}'
        raise DatabaseNotFoundError(msg)

    if not os.path.exists(db_path):
        con = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        create_schema(con)
    con = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    return con


def create_posts_table_gbq(client, gbq_credentials, create_if_empty=False):
    """Create new posts table if it does not exist.

    Args:
        client: GBQ client
        gbq_credentials: GBQ Credentials.
        create_if_empty: bool. Create new posts table if True.
    """
    tables = client.list_tables('redbotdb')
    found = False
    for table in tables:
        if table.table_id == 'posts':
            found = True

    if not found and not create_if_empty:
        msg = 'Posts table not found'
        raise DatabaseNotFoundError(msg)

    if not found:
        schema = [
            bigquery.SchemaField('uuid', 'STRING', mode='NULLABLE'),
            bigquery.SchemaField('uid', 'STRING', mode='NULLABLE'),
            bigquery.SchemaField('url', 'STRING', mode='NULLABLE'),
            bigquery.SchemaField('title', 'STRING', mode='NULLABLE'),
            bigquery.SchemaField('score', 'FLOAT', mode='NULLABLE'),
            bigquery.SchemaField('upvote_ratio', 'FLOAT', mode='NULLABLE'),
            bigquery.SchemaField('highrank24', 'FLOAT', mode='NULLABLE'),
            bigquery.SchemaField('created_utc', 'TIMESTAMP', mode='NULLABLE'),
            bigquery.SchemaField('time_highrank', 'TIMESTAMP', mode='NULLABLE'),
            bigquery.SchemaField('subreddit', 'STRING', mode='NULLABLE'),
            bigquery.SchemaField('prediction', 'FLOAT', mode='NULLABLE'),
            bigquery.SchemaField('db_version', 'STRING', mode='NULLABLE')
        ]

        table = bigquery.Table(gbq_credentials.project_id + '.redbotdb.posts', schema=schema)
        client.create_table(table)


def connect_to_gbq(gbq_credentials, create_if_empty):
    """Connect to Google BigQuery.

    Args:
        gbq_credentials: GBQ Credentials.
        create_if_empty: bool. Create new posts table if True.

    Returns:
         con: database connection
    """
    client = bigquery.Client(credentials=gbq_credentials, project=gbq_credentials.project_id)
    create_posts_table_gbq(client, gbq_credentials, create_if_empty)
    con = dbapi.Connection(client)
    return con


def get_uids(con):
    """Retrieve post ids from database.

    Args:
        con: database connection

    Returns:
        uids: list of post ids.
    """
    cursor = con.cursor()
    sql = """
    SELECT uid FROM posts
    """
    uid_gen = cursor.execute(sql)
    return [elem[0] for elem in uid_gen]


def get_highrank(con, uid):
    """Retrieve highest rank of a specified post from database.

    Args:
        con: database connection
        uid: str. post id obtained from PRAW

    Returns:
        high_rank: int. highest rank of the post as stored in the database.
    """
    cursor = con.cursor()
    sql = """
    SELECT highrank24 FROM posts WHERE uid=?
    """
    high_rank_gen = cursor.execute(sql, (uid,))
    return next(high_rank_gen)[0]


def update_highrank(con, new_high_rank, new_time_highrank, uid):
    """Update the highest rank of a specified post in the database and its corresponding time.

    Args:
        con: database connection
        new_high_rank: int. The highest updated high rank for the specified post.
        new_time_highrank: datetime. The time corresponding to the highest updated high rank for the specified post.
        uid: str. The post id obtained from PRAW.
    """
    cursor = con.cursor()
    sql = """
    UPDATE posts SET highrank24=?, time_highrank=? WHERE uid=?
    """
    cursor.execute(sql, (int(new_high_rank), new_time_highrank, uid))
    con.commit()


def update_score(con, new_score, uid):
    """Update the score of a specified post in the database.

    Args:
        con: database connection
        new_score: int. The most recent updated score for the specified post.
        uid: str. The post id obtained from PRAW.
    """
    cursor = con.cursor()
    sql = """
    UPDATE posts SET score=? WHERE uid=?
    """
    cursor.execute(sql, (int(new_score), uid))
    con.commit()


def insert_new_post(con, post, high_rank, time_highrank, subreddit, prediction, uuid):
    """Insert a new post entry into the database.

    Create new entry with uid, url, title, score, up_vote ratio, highrank, time of high rank,
    subreddit, and prediction corresponding to the new post.

    Args:
        con: database connection
        post: post object obtained from PRAW.
        high_rank: int. The highest rank of the post at the time of creation/insertion into the database.
        Initialized to None.
        time_highrank: datatime. The time corresponding to highest rank at the time of post creation.
        Initialized to None.
        subreddit: str. Name of subreddit of the new post.
        prediction: int. Prediction on whether the post is likely to appear among the top 10 hot posts.
        Initialized to None.
        uuid: str. Globally unique id.
    """
    cursor = con.cursor()
    sql = """
    INSERT INTO posts(uuid, uid, url, title, score, upvote_ratio, highrank24, created_utc, time_highrank, subreddit, prediction, db_version) 
    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    cursor.execute(
        sql,
        (uuid,
         post.id,
         post.url,
         post.title,
         int(post.score),
         post.upvote_ratio,
         int(high_rank) if high_rank is not None else high_rank,
         datetime.utcfromtimestamp(post.created_utc),
         time_highrank,
         subreddit,
         prediction,
         DB_VERSION)
    )
    con.commit()


def update_prediction(con, new_pred, uuid):
    """Update the prediction of a specified post in the database.

    Args:
        con: database connection
        new_pred: int. The most recent updated prediction (hot or not) for the specified post.
        uuid: str. The post uid obtained from PRAW.
    """
    cursor = con.cursor()
    sql = """
    UPDATE posts SET prediction=? WHERE uuid=?
    """
    cursor.execute(sql, (int(new_pred), uuid))
    con.commit()
