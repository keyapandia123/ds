"""Module to handle database operations."""
import os
import sqlite3


DATABASE_DEFAULT_PATH = '~/.redbot/redbotdb.sqlite'


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
            highrank24 INTEGER
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
        con = sqlite3.connect(db_path)
        create_schema(con)
    con = sqlite3.connect(db_path)
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
    return list(uid_gen)


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
    return next(high_rank_gen)


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
    cursor.execute(sql, (new_high_rank, uid))


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
    cursor.execute(sql, (new_score, uid))



def insert_new_post(con, post, high_rank):
    """Insert a new post entry into the database.

    Create new entry with uid, url, title, score, and highrank24 corresponding to the new post.

    Args:
        con: database connection
        post: post object obtained from PRAW.
        high_rank: int. The highest rank of the post at the time of creation/insertion into the database.
    """
    cursor = con.cursor()
    sql = """
    INSERT INTO posts(uid, url, title, score, highrank24) 
    VALUES(?, ?, ?, ?, ?)
    """
    cursor.execute(
        sql,
       (post.id,
        post.url,
        post.title,
        post.score,
        high_rank)
    )