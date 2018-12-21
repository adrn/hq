# Standard library
import os
import warnings

# Third-party
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.engine import Engine
from sqlalchemy import event
from sqlalchemy.exc import SAWarning
import yaml

Base = declarative_base()

@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()
    warnings.filterwarnings('ignore', ".*Decimal objects natively.*", SAWarning)


def db_connect(database_path, ensure_db_exists=False):
    """Connect to the specified SQLite database file.

    Parameters
    ----------
    database_path : str
        Path to the SQLite database file.
    ensure_db_exists : bool (optional)
        Ensure that the database file exists at the specified path.

    Returns
    -------
    engine :
        The ``sqlalchemy`` database engine.
    """

    if not os.path.exists(database_path) and not ensure_db_exists:
        raise IOError("Database file '{0}' does not exist! To create it "
                      "automatically, pass `ensure_db_exists=True`.")

    engine = create_engine("sqlite:///{}"
                           .format(os.path.abspath(database_path)))
    Session = scoped_session(sessionmaker(bind=engine, autoflush=True,
                                          autocommit=False))
    Base.metadata.bind = engine

    if ensure_db_exists:
        Base.metadata.create_all(engine)

    return Session, engine


def session_from_config(config_file, **db_connect_kw):
    """Create a database session given an ``hq`` config file.

    Parameters
    ----------
    config_file : str
        Path to the YAML config file for the ``hq`` run.
    **db_connect_kw
        All other keyword arguments are passed to ``db_connect()``.

    Returns
    -------
    session :
        The ``sqlalchemy`` session object with an open cursor to the database.

    """
    from ..config import HQ_CACHE_PATH
    with open(config_file, 'r') as f:
        config = yaml.load(f.read())

    db_path = os.path.join(HQ_CACHE_PATH, config.get('database_file', None))
    return db_connect(db_path, **db_connect_kw)
