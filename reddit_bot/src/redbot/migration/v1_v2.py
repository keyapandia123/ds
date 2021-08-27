"""Migrate database from v1 to v2."""
import sqlite3
import os
import uuid

import pandas as pd

from redbot import db


DATABASE_DEFAULT_PATH = '~/.redbot/redbotdb.sqlite'


def recreate_uuids(path_old):
    """Replace numeric ids with string uuids, add database version.

    Args:
        path_old: str. Path where database v1 is located.
    """
    path_old = os.path.expanduser(path_old)
    con_old = db.connect_to_db(path_old)
    df = pd.read_sql("select id, uid, url, title, score, upvote_ratio, highrank24, created_utc, time_highrank, "
                     "subreddit, prediction from posts", con_old)
    df.drop('id', axis=1, inplace=True)

    def foo(x):
        return str(uuid.uuid4())

    df['uuid'] = df.apply(foo, axis=1)
    df['db_version'] = db.DB_VERSION
    path_new = f'{path_old}.new'
    con_new = sqlite3.connect(path_new)
    db.create_schema(con_new)
    df.to_sql('posts', con_new, if_exists='append', index=False)

    df_new = pd.read_sql("select * from posts", db.connect_to_db(path_new, create_if_empty=False))

    df.sort_values('uuid', inplace=True)
    df_new.sort_values('uuid', inplace=True)
    df_new = df_new[df.columns]

    assert df.shape == df_new.shape
    assert df.equals(df_new)

    os.rename(path_old, f'{path_old}.save')
    os.rename(path_new, path_old)


if __name__ == '__main__':
    recreate_uuids(DATABASE_DEFAULT_PATH)