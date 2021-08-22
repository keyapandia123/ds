"""Module to handle database migrations."""
from redbot import db

DATABASE_DEFAULT_PATH = '~/.redbot/redbotdb.sqlite'


def correct_time_high_rank(con):
    cursor = con.cursor()
    sql = """
    UPDATE posts SET time_highrank = NULL WHERE highrank24 IS NULL
    """
    cursor.execute(sql)
    con.commit()


if __name__ == '__main__':
    con = db.connect_to_db(DATABASE_DEFAULT_PATH, create_if_empty=False)
    correct_time_high_rank(con)