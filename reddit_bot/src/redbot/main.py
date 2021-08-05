import pandas as pd
import praw

from . import credentials
from . import db


REDDIT = praw.Reddit(**credentials.load_credentials())
DATABASE_DEFAULT_PATH = '~/.redbot/redbotdb.sqlite'

def generate_subreddit_posts_database(limit, sub_name):
    subreddit = REDDIT.subreddit(sub_name)
    posts = subreddit.hot(limit=limit)
    con = db.connect_to_database(DATABASE_DEFAULT_PATH, create_if_empty=True)
    uids = db.get_uids(con)
    for num, p in enumerate(posts):
        if p.id in uids:
            # update post entry in table
            idy = p.id
            high_rank = db.get_highrank(con, idy)
            if num < high_rank:
                db.update_highrank(con, num, idy)
            new_score = p.score
            db.update_score(con, new_score, idy)
        else:
            # create new post entry
            db.insert_new_post(con, p, num)

