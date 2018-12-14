# Standard library
from os import path, unlink

# Third-party
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
import yaml

# Project
from ...config import TWOFACE_CACHE_PATH
from ..connect import db_connect
from ..model import AllStar, AllVisit, StarResult, Status, JokerRun
from ..init import initialize_db


class TestDB(object):

    def setup(self):
        # connect to database
        with open(get_pkg_data_filename('travis_db.yml')) as f:
            config = yaml.load(f)

        self.db_path = path.join(TWOFACE_CACHE_PATH, config['database_file'])

        # initialize the database
        Session, self.engine = db_connect(self.db_path)

        initialize_db(allVisit_file=get_pkg_data_filename('test-allVisit.fits'),
                      allStar_file=get_pkg_data_filename('test-allStar.fits'),
                      database_file=self.db_path,
                      drop_all=True)

        self.session = Session()

    def test_one(self):
        s = self.session

        # a target in both test FITS files included in the repo
        star = s.query(AllStar).filter(AllStar.apogee_id == "2M00000068+5710233").one()
        test_target_ID = star.target_id

        # get star entry and check total num of visits
        star = s.query(AllStar).filter(AllStar.target_id == test_target_ID).one()
        assert len(star.visits) == 3

        # get a visit and check that it has one star
        visit = s.query(AllVisit).filter(AllVisit.target_id == test_target_ID).limit(1).one()
        assert len(visit.stars) == 1

    def test_jokerrun_cascade(self):
        """
        Make sure the Results and Statuses are deleted when a JokerRun is
        deleted
        """
        s = self.session

        NAME = 'test-cascade'

        # first set up:
        stars = s.query(AllStar).all()

        run = JokerRun()
        run.config_file = ''
        run.name = NAME
        run.P_min = 1.*u.day
        run.P_max = 100.*u.day
        run.requested_samples_per_star = 128
        run.max_prior_samples = 1024
        run.prior_samples_file = ''
        run.stars = stars
        s.add(run)
        s.commit()

        assert s.query(JokerRun).count() == 1
        assert s.query(AllStar).count() == s.query(StarResult).count()

        for run in s.query(JokerRun).filter(JokerRun.name == NAME).all():
            s.delete(run)
        s.commit()

        assert s.query(JokerRun).count() == 0
        assert s.query(StarResult).count() == 0
        assert s.query(Status).count() > 0 # just to be safe

    def teardown(self):
        self.session.close()

        # delete the test database
        unlink(self.db_path)
