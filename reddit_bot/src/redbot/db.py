"""Module to handle database operations."""
from datetime import datetime
import os
import sqlite3


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
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            uid TEXT, 
            url TEXT, 
            title TEXT, 
            score INTEGER,
            upvote_ratio FLOAT,
            highrank24 INTEGER,
            created_utc TIMESTAMP,
            time_highrank TIMESTAMP, 
            subreddit TEXT, 
            prediction INTEGER
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


def update_highrank(con, new_high_rank, uid):
    """Update the highest rank of a specified post in the database.

    Args:
        con: database connection
        new_high_rank: int. The highest updated high rank for the specified post.
        uid: str. The post id obtained from PRAW.
    """
    cursor = con.cursor()
    sql = """
    UPDATE posts SET highrank24=? WHERE uid=?
    """
    cursor.execute(sql, (int(new_high_rank), uid))
    con.commit()


def update_time_highrank(con, new_time_highrank, uid):
    """Update time of the highest rank of a specified post in the database.

    Args:
        con: database connection
        new_time_highrank: datetime. The time corresponding to the highest updated high rank for the specified post.
        uid: str. The post id obtained from PRAW.
    """
    cursor = con.cursor()
    sql = """
    UPDATE posts SET time_highrank=? WHERE uid=?
    """
    cursor.execute(sql, (new_time_highrank, uid))
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


def insert_new_post(con, post, high_rank, time_highrank, subreddit, prediction):
    """Insert a new post entry into the database.

    Create new entry with uid, url, title, score, and highrank24 corresponding to the new post.

    Args:
        con: database connection
        post: post object obtained from PRAW.
        high_rank: int. The highest rank of the post at the time of creation/insertion into the database.
    """
    cursor = con.cursor()
    sql = """
    INSERT INTO posts(uid, url, title, score, upvote_ratio, highrank24, created_utc, time_highrank, subreddit, prediction) 
    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    cursor.execute(
        sql,
        (post.id,
         post.url,
         post.title,
         int(post.score),
         post.upvote_ratio,
         int(high_rank),
         datetime.utcfromtimestamp(post.created_utc),
         time_highrank,
         subreddit,
         prediction)
    )
    con.commit()


def update_prediction(con, new_pred, idy):
    """Update the prediction of a specified post in the database.

    Args:
        con: database connection
        new_pred: int. The most recent updated prediction (hot or not) for the specified post.
        idy: int. The post id or primary key.
    """
    cursor = con.cursor()
    sql = """
    UPDATE posts SET prediction=? WHERE id=?
    """
    cursor.execute(sql, (int(new_pred), int(idy)))
    con.commit()
