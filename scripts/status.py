from os import path
from twoface.db import (db_connect, AllStar, AllVisit, AllVisitToAllStar,
                        StarResult, Status, JokerRun)
from twoface.config import TWOFACE_CACHE_PATH

def main(run_name):
    Session, _ = db_connect(path.join(TWOFACE_CACHE_PATH, 'apogee.sqlite'))
    session = Session()

    done_subq = session.query(AllStar.apogee_id)\
                       .join(StarResult, JokerRun, Status)\
                       .filter(Status.id > 0).distinct()

    run = session.query(JokerRun).filter(JokerRun.name == run_name).one()

    n_total = session.query(AllStar).join(StarResult, JokerRun, Status)\
                                    .filter(JokerRun.name == run.name)\
                                    .count()
    print("{0} total".format(n_total))

    n_left = session.query(AllStar).join(StarResult, JokerRun, Status)\
                                   .filter(JokerRun.name == run.name)\
                                   .filter(~AllStar.apogee_id.in_(done_subq))\
                                   .count()
    print("{0} left to process".format(n_left))

    print("\nDone:")
    for status in session.query(Status).order_by(Status.id).all():
        star_query = session.query(AllStar).join(StarResult, JokerRun, Status)\
                                           .filter(JokerRun.name == run.name)\
                                           .filter(Status.id == status.id)
        print("{0} ({1}): {2}".format(status.message, status.id,
                                      star_query.count()))

if __name__ == "__main__":
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")

    # HACK: TODO: add option to specify name
    main(run_name='apogee-jitter')
