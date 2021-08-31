"""Migrate database to GBQ"""
import os

import pandas as pd

from redbot import credentials
from redbot import db

DATABASE_DEFAULT_PATH = '~/.redbot/redbotdb.sqlite'


def transfer_to_gbq(path_old):
    path_old = os.path.expanduser(path_old)
    con_old = db.connect_to_db(path_old)
    df = pd.read_sql("select * from posts", con_old)

    df['score'] = df['score'].astype(float)
    df['highrank24'] = df['highrank24'].astype(float)
    df['prediction'] = df['prediction'].astype(float)

    gbq_credentials = credentials.load_gcp_credentials()
    con_gbq = db.connect_to_gbq(gbq_credentials, create_if_empty=True)
    df.to_gbq('redbotdb.posts', credentials=gbq_credentials, if_exists='append')

    df_new = pd.read_sql(f"select * from {gbq_credentials.project_id}.redbotdb.posts", con_gbq)

    df.sort_values('uuid', inplace=True)
    df_new.sort_values('uuid', inplace=True)
    df_new = df_new[df.columns]

    # BigQuery timestamps are timezone aware, while sqlite aren't.
    df_new['created_utc'] = df_new['created_utc'].apply(lambda x: x.astimezone(None) if not pd.isnull(x) else x)
    df_new['time_highrank'] = df_new['time_highrank'].apply(lambda x: x.astimezone(None) if not pd.isnull(x) else x)

    assert df.shape == df_new.shape
    assert df.reset_index(drop=True).equals(df_new.reset_index(drop=True))


if __name__ == '__main__':
    transfer_to_gbq(DATABASE_DEFAULT_PATH)
