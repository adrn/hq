"""
Reset the statuses of all of the stars in the database.
"""

# Standard library
import sys
from os import path
from tqdm import tqdm

# Project
from hq.config import HQ_CACHE_PATH
from hq.db import db_connect, StarResult

def main(database_path):
    Session, engine = db_connect(database_path)
    session = Session()

    for r in tqdm(session.query(StarResult).all()):
        r.status_id = 0
    session.commit()


if __name__ == "__main__":
    main(database_path=path.join(HQ_CACHE_PATH, 'apogee.sqlite')) # HARD CODED
    sys.exit(0)
