from datetime import datetime
import logging
from logging import handlers
import os
import time
import uuid

import praw

from redbot import credentials
from redbot import db
from redbot import inference


LOG_FORMATTER = logging.Formatter('%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s')
LOGFILE = os.path.expanduser('~/.redbot/log')
REDDIT = praw.Reddit(**credentials.load_reddit_credentials())
DATABASE_DEFAULT_PATH = '~/.redbot/redbotdb.sqlite'
SUB_NAME = 'politics'
NEW_POST_LIMIT = 20
HOT_POST_LIMIT = 10
SLEEP_S = 60


def setup_logging(logFile, log_formatter):
    my_handler = handlers.RotatingFileHandler(logFile, maxBytes=10 * 1024 * 1024, backupCount=2)
    my_handler.setFormatter(log_formatter)
    logging.getLogger().setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)

    _log = logging.getLogger(__name__)
    _log.addHandler(my_handler)
    _log.addHandler(stream_handler)

    return _log


_log = setup_logging(LOGFILE, LOG_FORMATTER)


def ingest_new_posts(limit, sub_name):
    """Add new posts to the database as they are created in Reddit.

    Ingest predefined number of new posts from the specified subreddit.
    As a new post gets created (the new post is not already present in
    the database), insert its uid, url, title, score, upvote_ratio,
    time of creation, and subreddit into the posts table of the database.
    If a post exists in the table, update its score (upvotes minus downvotes)
    within 1 hour of its creation.

    Args:
        limit: int. predefined number of new posts to ingest everytime this function is called
        sub_name: str. name of subreddit from which to ingest new posts
    """
    subreddit = REDDIT.subreddit(sub_name)
    posts = subreddit.new(limit=limit)
    con = db.connect_to_db(DATABASE_DEFAULT_PATH, create_if_empty=True)
    uids = db.get_uids(con)
    for num, p in enumerate(posts):
        now = time.time()
        duration_s = now - p.created_utc
        duration_hr = duration_s / 3600.
        if not p.stickied:
            if p.id in uids:
                if duration_hr <= 1:
                    # Update score of an existing post within 1 hr of its creation.
                    # The score (upvote - downvote) within the first hour of
                    # creation can be used as a predictor for the model.
                    idy = p.id
                    new_score = p.score
                    db.update_score(con, new_score, idy)
                    _log.info(f"Score update for {p.id} to {new_score}")
            else:
                # Create new post entry
                db.insert_new_post(con, p, None, None, sub_name, None, str(uuid.uuid4()))
                _log.info(f"Post insertion for {p.id} with title {p.title}")

    con.commit()


def check_hot_posts(limit, sub_name):
    """Compare ingested posts against top hot posts in Reddit.

    Check if posts ingested into the table have appeared among a predefined number of
    the top hot posts in Reddit.
    Fetch a predefined number of top hot posts from Reddit and check
    which of those hot posts were present in the posts table within 24 hrs
    of the posts' creation.
    If a post present in the posts table appeared in the top hot posts
    within 24 hrs of the post's creation, update its high rank (its rank among the top hot posts)
    and time corresponding to the high rank for that post.

    Args:
        limit: int. predefined number of top hot posts to fetch from Reddit.
        sub_name: str. Subbreddit from which to fetch hot posts.
    """
    subreddit = REDDIT.subreddit(sub_name)
    posts = subreddit.hot(limit=limit)
    con = db.connect_to_db(DATABASE_DEFAULT_PATH, create_if_empty=True)
    uids = db.get_uids(con)
    for num, p in enumerate(posts):
        if not p.stickied:
            now = time.time()
            duration_s = now - p.created_utc
            duration_hr = duration_s / 3600.
            if p.id in uids and duration_hr <= 24:
                # Update high rank and time corresponding to high rank within
                # 24 hrs of creation of the post.
                idy = p.id
                high_rank = db.get_highrank(con, idy)
                if high_rank is None or num < high_rank:
                    db.update_highrank(con, num, datetime.utcfromtimestamp(now), idy)
                    _log.info(f"High rank update for {p.id} from {high_rank} to {num} at {datetime.utcfromtimestamp(now)}")
    con.commit()


def main():
    """Main function loop.

    Continually ingest new posts from a specified subreddit
    and check if any of the new ingested posts are present in the top
    hot posts for that subreddit within 24 hours of the post's creation.
    """
    while True:
        ingest_new_posts(NEW_POST_LIMIT, SUB_NAME)
        time.sleep(1)
        check_hot_posts(HOT_POST_LIMIT, SUB_NAME)
        con = db.connect_to_db(DATABASE_DEFAULT_PATH, create_if_empty=False)
        inference.run_inference(con, new_model=False, to_save=False)
        time.sleep(SLEEP_S)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass