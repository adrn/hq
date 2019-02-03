# Standard library
from os.path import abspath, expanduser, basename, splitext, join

# Third-party
from astropy.io import fits
from astropy.table import Table
import numpy as np

# Project
from ..log import log as logger
from .connect import db_connect, Base
from .model import AllStar, AllVisit, Status
from .helpers import paged_query
from ..config import HQ_CACHE_PATH

__all__ = ['initialize_db']


def tblrow_to_dbrow(tblrow, colnames, varchar_cols=[]):
    row_data = dict()
    for k in colnames:
        if k in varchar_cols:
            row_data[k.lower()] = tblrow[k].strip()

        # special case the array columns
        elif k.lower().startswith('fparam') and 'cov' not in k.lower():
            i = int(k[-1])
            row_data[k.lower()] = tblrow[k[:-1]][i]

        elif k.lower().startswith('fparam') and 'cov' in k.lower():
            i = int(k[-2])
            j = int(k[-1])
            row_data[k.lower()] = tblrow[k[:-2]][i,j]

        else:
            row_data[k.lower()] = tblrow[k]

    return row_data


def initialize_db(allVisit_file, allStar_file, 
                  drop_all=False, batch_size=4096, min_nvisits=3,
                  progress=True):
    """Initialize the database given FITS filenames for the APOGEE data.

    TODO: allow customizing bitmasks used.

    Parameters
    ----------
    allVisit_file : str
        Full path to APOGEE allVisit file.
    allStar_file : str
        Full path to APOGEE allStar file.
    drop_all : bool (optional)
        Drop all existing tables and re-create the database.
    batch_size : int (optional)
        How many rows to create before committing.
    """

    if progress:
        from tqdm import tqdm
        iterate = tqdm
        load_print = tqdm.write
    else:
        iterate = lambda x: x
        load_print = print
    
    run_name = splitext(basename(allVisit_file))[0][9:]
    database_path = join(HQ_CACHE_PATH, 
                         'apogee-{0}.sqlite'.format(run_name))

    norm = lambda x: abspath(expanduser(x))
    allvisit_tbl = fits.getdata(norm(allVisit_file))
    allstar_tbl = fits.getdata(norm(allStar_file))

    # Remove bad velocities and flagged bad visits:
    # LOW_SNR, PERSIST_HIGH, PERSIST_JUMP_POS, PERSIST_JUMP_NEG
    skip_mask = np.sum(2 ** np.array([4, 9, 12, 13]))
    allvisit_tbl = allvisit_tbl[np.isfinite(allvisit_tbl['VHELIO']) &
                                np.isfinite(allvisit_tbl['VRELERR'])]
    allvisit_tbl = allvisit_tbl[(allvisit_tbl['VRELERR'] < 100.) & # MAGIC NUMBER
                                ((allvisit_tbl['STARFLAG'] & skip_mask) == 0)]

    v_apogee_ids, counts = np.unique(allvisit_tbl['APOGEE_ID'],
                                     return_counts=True)
    star_mask = np.isin(allstar_tbl['APOGEE_ID'],
                        v_apogee_ids[counts >= min_nvisits])

    # Remove stars flagged with:
    # TEFF_WARN, ROTATION_WARN, CHI2_WARN, STAR_BAD
    skip_mask = np.sum(2 ** np.array([0, 8, 10, 23]))
    star_mask &= ((allstar_tbl['ASPCAPFLAG'] & skip_mask) == 0)

    # Remove stars flagged with:
    # VERY_BRIGHT_NEIGHBOR, SUSPECT_RV_COMBINATION, SUSPECT_BROAD_LINES
    skip_mask = np.sum(2 ** np.array([3, 16, 17]))
    star_mask &= ((allstar_tbl['STARFLAG'] & skip_mask) == 0)
    allstar_tbl = allstar_tbl[star_mask]

    # Only load visits for stars that we're loading
    allvisit_tbl = allvisit_tbl[np.isin(allvisit_tbl['APOGEE_ID'],
                                        allstar_tbl['APOGEE_ID'])]
    v_apogee_ids2 = np.unique(allvisit_tbl['APOGEE_ID'])
    star_mask2 = np.isin(allstar_tbl['APOGEE_ID'], v_apogee_ids2)

    allvisit_tbl = Table(allvisit_tbl)
    allstar_tbl = Table(allstar_tbl[star_mask2])

    Session, engine = db_connect(database_path, ensure_db_exists=True)
    logger.debug("Connected to database at '{}'".format(database_path))

    if drop_all:
        # this is the magic that creates the tables based on the definitions in
        # twoface/db/model.py
        Base.metadata.drop_all()
        Base.metadata.create_all()

    session = Session()

    logger.debug("Loading {0} rows in allStar table, {1} rows "
                 "in allVisit table...".format(len(allstar_tbl),
                                               len(allvisit_tbl)))

    # Figure out what data we need to pull out of the FITS files based on what
    # columns exist in the (empty) database
    allstar_skip = ['ID']
    allstar_colnames = []
    allstar_varchar = []
    for x in AllStar.__table__.columns:
        col = str(x).split('.')[1].upper()
        if col in allstar_skip:
            continue

        if str(x.type) == 'VARCHAR':
            allstar_varchar.append(col)

        allstar_colnames.append(col)

    allvisit_skip = ['ID']
    allvisit_colnames = []
    allvisit_varchar = []
    for x in AllVisit.__table__.columns:
        col = str(x).split('.')[1].upper()
        if col in allvisit_skip:
            continue

        if str(x.type) == 'VARCHAR':
            allvisit_varchar.append(col)

        allvisit_colnames.append(col)

    # --------------------------------------------------------------------------
    # First load the status table:
    #
    if session.query(Status).count() == 0:
        logger.debug("Populating Status table...")
        statuses = list()
        statuses.append(Status(id=0, message='untouched'))
        statuses.append(Status(id=1, message='needs more prior samples'))
        statuses.append(Status(id=2, message='needs mcmc'))
        statuses.append(Status(id=3, message='error'))
        statuses.append(Status(id=4, message='completed'))

        session.add_all(statuses)
        session.commit()
        logger.debug("...done")

    # --------------------------------------------------------------------------
    # Load the AllStar table:
    #
    logger.info("Loading AllStar table")

    # What APSTAR_ID's are already loaded?
    all_ap_ids = np.array([x.strip() for x in allstar_tbl['APSTAR_ID']])
    loaded_ap_ids = [x[0] for x in session.query(AllStar.apstar_id).all()]
    mask = np.logical_not(np.isin(all_ap_ids, loaded_ap_ids))
    logger.debug("{0} stars already loaded".format(len(loaded_ap_ids)))
    logger.debug("{0} stars left to load".format(mask.sum()))

    stars = []
    i = 0
    for row in iterate(allstar_tbl[mask]): # Load every star
        row_data = tblrow_to_dbrow(row, allstar_colnames, allstar_varchar)

        # create a new object for this row
        star = AllStar(**row_data)
        stars.append(star)

        if i % batch_size == 0 and i > 0:
            session.add_all(stars)
            session.commit()
            if logger.getEffectiveLevel() <= 10:
                load_print("Loaded batch {0}".format(i))
            stars = []

        i += 1

    if len(stars) > 0:
        session.add_all(stars)
        session.commit()

    # --------------------------------------------------------------------------
    # Load the AllVisit table:
    #
    logger.info("Loading AllVisit table")

    # What VISIT_ID's are already loaded?
    all_vis_ids = np.array([x.strip() for x in allvisit_tbl['VISIT_ID']])
    loaded_vis_ids = [x[0] for x in session.query(AllVisit.visit_id).all()]
    mask = np.logical_not(np.isin(all_vis_ids, loaded_vis_ids))
    logger.debug("{0} visits already loaded".format(len(loaded_vis_ids)))
    logger.debug("{0} visits left to load".format(mask.sum()))

    visits = []
    i = 0
    for row in iterate(allvisit_tbl[mask]): # Load every visit
        row_data = tblrow_to_dbrow(row, allvisit_colnames, allvisit_varchar)

        # create a new object for this row
        visit = AllVisit(**row_data)
        visits.append(visit)

        if i % batch_size == 0 and i > 0:
            session.add_all(visits)
            session.commit()
            if logger.getEffectiveLevel() <= 10:
                load_print("Loaded batch {0}".format(i))
            visits = []

        i += 1

    if len(visits) > 0:
        session.add_all(visits)
        session.commit()

    # --------------------------------------------------------------------------
    # Now associate rows in AllStar with rows in AllVisit
    #
    # Note: here we do something a little crazy. Because APOGEE_ID isn't really
    # a unique identifier (the same APOGEE_ID might be observed on different
    # plates), the same visits might belong to two different APSTAR_ID's if we
    # associate visits to APOGEE_ID's. But that is what we do: we give all
    # visits to any APOGEE_ID, so any processing we do over all sources should
    # UNIQUE/DISTINCT on APOGEE_ID.
    logger.info("Linking AllVisit and AllStar tables")

    q = session.query(AllStar).order_by(AllStar.id)

    for i, sub_q in iterate(enumerate(paged_query(q, page_size=batch_size))):
        for star in sub_q:
            if len(star.visits) > 0:
                continue

            visits = session.query(AllVisit).filter(
                AllVisit.apogee_id == star.apogee_id).all()

            if len(visits) == 0:
                logger.warn("Visits not found for star {0}".format(star))
                continue

            star.visits = visits

        session.commit()
        if logger.getEffectiveLevel() <= 10:
            load_print("Loaded batch {0}".format(i))

    session.commit()
    session.close()
