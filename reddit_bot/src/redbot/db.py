"""Module to handle database operations."""
from datetime import datetime
import os
import sqlite3

from google.cloud import bigquery
from google.cloud.bigquery import dbapi
import pandas as pd


DB_VERSION = 'v2'
# Use 'posts' when interfacing with SQLite and replace with f'{gbq_credentials.project_id}.redbotdb.posts'
# when interfacing with GBQ
table_name = 'posts'


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
    sql = f"""
    SELECT uid FROM {table_name}
    """
    df = pd.read_sql(sql, con)
    return list(df['uid'])


def get_highrank(con, uid):
    """Retrieve highest rank of a specified post from database.

    Args:
        con: database connection
        uid: str. post id obtained from PRAW

    Returns:
        high_rank: int. highest rank of the post as stored in the database.
    """
    sql = f"""
    SELECT highrank24 FROM {table_name} WHERE uid=%(:string)s
    """
    df = pd.read_sql(sql, con, params=[uid])
    return list(df['highrank24'])[0]


def update_highrank(con, new_high_rank, new_time_highrank, uid):
    """Update the highest rank of a specified post in the database and its corresponding time.

    Args:
        con: database connection
        new_high_rank: int. The highest updated high rank for the specified post.
        new_time_highrank: datetime. The time corresponding to the highest updated high rank for the specified post.
        uid: str. The post id obtained from PRAW.
    """
    cursor = con.cursor()
    sql = f"""
    UPDATE {table_name} SET highrank24=%(:numeric)s, time_highrank=%(:timestamp)s WHERE uid=%(:string)s
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
    sql = f"""
    UPDATE {table_name} SET score=%(:numeric)s WHERE uid=%(:string)s
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
    sql = f"""
    INSERT INTO `{table_name}` (uuid, uid, url, title, score, upvote_ratio, highrank24, created_utc, 
    time_highrank, subreddit, prediction, db_version) 
    VALUES (%(:string)s, %(:string)s, %(:string)s, %(:string)s, 
    %(:numeric)s, %(:numeric)s, %(:numeric)s, %(:timestamp)s, 
    %(:timestamp)s, %(:string)s, %(:numeric)s, %(:string)s)
    """
    cursor.execute(
        sql,
        (uuid,
         post.id,
         post.url,
         post.title,
         int(post.score),
         post.upvote_ratio,
         int(high_rank) if high_rank is not None else None,
         datetime.utcfromtimestamp(post.created_utc),
         time_highrank if time_highrank is not None else None,
         subreddit,
         prediction if prediction is not None else None,
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
    sql = f"""
    UPDATE {table_name} SET prediction=%(:numeric)s WHERE uuid=%(:string)s
    """
    cursor.execute(sql, (int(new_pred), uuid))
    con.commit()


def retrieve_posts_for_inference(con, current_time_utc):
    """Retrieve posts ingested between 1 and 3 hours ago that have no predictions.

    Args:
        con: database connection.
        current_time_utc: datetime. Current UTC time.

    Returns:
        raw_test_df: DataFrame. DataFrame containing raw attributes
        (uuid, url, title, score, upvote_ratio) for posts ingested
        between 1 and 3 hours ago.
    """
    sql = f"""
    SELECT uuid, url, title, score, upvote_ratio FROM {table_name} 
    WHERE (prediction IS NULL) AND  
    ((TIMESTAMP_DIFF(TIMESTAMP('{str(current_time_utc)}'), created_utc, SECOND) / 3600.0) >= 1.0) AND 
    ((TIMESTAMP_DIFF(TIMESTAMP('{str(current_time_utc)}'), created_utc, SECOND) / 3600.0) <= 3.0)
    """
    raw_test_df = pd.read_sql(sql, con)
    return raw_test_df


def retrieve_valid_posts_for_training(con, current_time_utc):
    """Return a raw dataframe of all valid posts (created over 24 hours ago).

    Args:
       con: database connection.
       current_time_utc: datetime. Current UTC time.

    Returns:
         raw_df: DataFrame. DataFrame containing raw attributes
        (url, title, score, upvote_ratio, highrank24) for posts ingested
        24 hours ago.
    """
    sql = f"""
    SELECT url, title, score, upvote_ratio, highrank24 FROM {table_name} 
    WHERE (TIMESTAMP_DIFF(TIMESTAMP('{str(current_time_utc)}'), created_utc, SECOND) / 3600.0) >= 24.0
    """
    raw_df = pd.read_sql(sql, con)
    return raw_df


def return_total_post_count_for_analysis(con):
    sql = f"""
    SELECT COUNT(uuid) AS cnt FROM {table_name}
    """
    df = pd.read_sql(sql, con)
    return list(df['cnt'])[0]


def return_total_hot_post_count_for_analysis(con):
    sql = f"""
    SELECT COUNT(uuid) AS cnt FROM {table_name} WHERE highrank24 IS NOT NULL"""
    df = pd.read_sql(sql, con)
    return list(df['cnt'])[0]


def return_total_valid_post_count_for_analysis(con, current_time_utc):
    sql = f"""
    SELECT COUNT(uuid) AS cnt FROM {table_name} 
    WHERE (TIMESTAMP_DIFF(TIMESTAMP('{str(current_time_utc)}'), created_utc, SECOND) / 3600.0) >= 24.0
    """
    df = pd.read_sql(sql, con)
    return list(df['cnt'])[0]


def return_total_hot_valid_post_count_for_analysis(con, current_time_utc):
    sql = f"""
    SELECT COUNT(uuid) AS cnt FROM {table_name} 
    WHERE (TIMESTAMP_DIFF(TIMESTAMP('{str(current_time_utc)}'), created_utc, SECOND) / 3600.0) >= 24 AND 
    highrank24 IS NOT NULL
    """
    df = pd.read_sql(sql, con)
    return list(df['cnt'])[0]


def return_hot_post_scores_for_analysis(con, current_time_utc):
    sql = f"""
    SELECT AVG(score), MIN(score), MAX(score)
    FROM {table_name} 
    WHERE highrank24 IS NOT NULL AND 
    (TIMESTAMP_DIFF(TIMESTAMP('{str(current_time_utc)}'), created_utc, SECOND) / 3600.0) >= 24
    """
    df = pd.read_sql(sql, con)
    return list(df.values.squeeze())


def return_non_hot_post_scores_for_analysis(con, current_time_utc):
    sql = f"""
    SELECT AVG(score), MIN(score), MAX(score)
    FROM {table_name} 
    WHERE highrank24 IS NULL AND 
    (TIMESTAMP_DIFF(TIMESTAMP('{str(current_time_utc)}'), created_utc, SECOND) / 3600.0) >= 24
    """
    df = pd.read_sql(sql, con)
    return list(df.values.squeeze())


def return_hot_post_titles_for_analysis(con, current_time_utc):
    sql = f"""
    SELECT title
    FROM {table_name} 
    WHERE highrank24 IS NOT NULL AND 
    (TIMESTAMP_DIFF(TIMESTAMP('{str(current_time_utc)}'), created_utc, SECOND) / 3600.0) >= 24
    """
    df = pd.read_sql(sql, con)
    return list(df['title'])


def return_non_hot_post_titles_for_analysis(con, current_time_utc):
    sql = f"""
    SELECT title
    FROM {table_name} 
    WHERE highrank24 IS NULL AND 
    (TIMESTAMP_DIFF(TIMESTAMP('{str(current_time_utc)}'), created_utc, SECOND) / 3600.0) >= 24
    """
    df = pd.read_sql(sql, con)
    return list(df['title'])


def return_trending_post_titles_for_analysis(con, current_time_utc):
    sql = f"""
    SELECT title
    FROM {table_name} 
    WHERE highrank24 IS NOT NULL AND 
    (TIMESTAMP_DIFF(TIMESTAMP('{str(current_time_utc)}'), created_utc, SECOND) / 3600.0) <= 24
    """
    df = pd.read_sql(sql, con)
    return list(df['title'])


def return_hot_post_urls_for_analysis(con, current_time_utc):
    sql = f"""
    SELECT url
    FROM {table_name} 
    WHERE highrank24 IS NOT NULL AND 
    (TIMESTAMP_DIFF(TIMESTAMP('{str(current_time_utc)}'), created_utc, SECOND) / 3600.0) >= 24
    """
    df = pd.read_sql(sql, con)
    return list(df['url'])


def return_non_hot_post_urls_for_analysis(con, current_time_utc):
    sql = f"""
    SELECT url
    FROM {table_name} 
    WHERE highrank24 IS NULL AND
    (TIMESTAMP_DIFF(TIMESTAMP('{str(current_time_utc)}'), created_utc, SECOND) / 3600.0) >= 24
    """
    df = pd.read_sql(sql, con)
    return list(df['url'])


def retrieve_posts_older_than_window_for_analysis_of_inference(con, current_time_utc, win_hr):
    sql = f"""
    SELECT uuid, uid, highrank24, score, prediction FROM {table_name}
    WHERE (prediction IS NOT NULL) AND
    ((TIMESTAMP_DIFF(TIMESTAMP('{str(current_time_utc)}'), created_utc, SECOND) / 3600.0) >= %(:numeric)s) AND
    ((TIMESTAMP_DIFF(TIMESTAMP('{str(current_time_utc)}'), created_utc, SECOND) / 3600.0) <= %(:numeric)s)
    """
    pred_df = pd.read_sql(sql, con, params=[win_hr, win_hr + 24.0])
    return pred_df
