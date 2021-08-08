from datetime import datetime
import time

import praw

import credentials
import db


REDDIT = praw.Reddit(**credentials.load_credentials())
DATABASE_DEFAULT_PATH = '~/.redbot/redbotdb.sqlite'
SUB_NAME = 'politics'
NEW_POST_LIMIT = 20
HOT_POST_LIMIT = 10
SLEEP_S = 60


def ingest_new_posts(limit, sub_name):
    subreddit = REDDIT.subreddit(sub_name)
    posts = subreddit.new(limit=limit)
    con = db.connect_to_db(DATABASE_DEFAULT_PATH, create_if_empty=True)
    uids = db.get_uids(con)
    for num, p in enumerate(posts):
        now = time.time()
        duration_s = now - p.created_utc
        duration_hr = duration_s / 3600.
        if p.id in uids:
            if duration_hr <= 1:
                # Update score of an existing post within 1 hr of its creation.
                # The score (upvote - downvote) within the first hour of
                # creation can be used as a predictor for the model.
                idy = p.id
                new_score = p.score
                db.update_score(con, new_score, idy)
        else:
            # Create new post entry
            db.insert_new_post(con, p, None, now, sub_name, None)

    con.commit()


def check_hot_posts(limit, sub_name):
    subreddit = REDDIT.subreddit(sub_name)
    posts = subreddit.hot(limit=limit)
    con = db.connect_to_db(DATABASE_DEFAULT_PATH, create_if_empty=True)
    uids = db.get_uids(con)
    for num, p in enumerate(posts):
        now = time.time()
        duration_s = now - p.created_utc
        duration_hr = duration_s / 3600.
        if p.id in uids and duration_hr <= 24:
            # Update high rank and time corresponding to high rank within
            # 24 hrs of creation of the post.
            idy = p.id
            high_rank = db.get_highrank(con, idy)
            if high_rank is None or num < high_rank:
                db.update_highrank(con, num, idy)
                db.update_time_highrank(con, datetime.utcfromtimestamp(now), idy)

    con.commit()


def main():
    while True:
        ingest_new_posts(NEW_POST_LIMIT, SUB_NAME)
        time.sleep(1)
        check_hot_posts(HOT_POST_LIMIT, SUB_NAME)
        time.sleep(SLEEP_S)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass