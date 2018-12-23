from os import path
import sys
from hq.db import (db_connect, AllStar, AllVisitToStar, AllVisit,
                   StarResult, Status, JokerRun)
from hq.config import HQ_CACHE_PATH

def main(run_name):
    Session, _ = db_connect(path.join(HQ_CACHE_PATH, 'apogee.sqlite'))
    session = Session()

    if run_name is None:
        n_stars = session.query(AllStar.apogee_id).distinct().count()
        n_stars_w_visits = session.query(AllStar.apogee_id)\
                                  .join(AllVisitToStar, AllVisit).count()
        print("{0} stars loaded".format(n_stars))
        print("{0} stars with visits".format(n_stars_w_visits))

    else:
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
    if len(sys.argv) > 1:
        run = sys.argv[1]
    else:
        run = None

    main(run_name=run)
