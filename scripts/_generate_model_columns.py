from __future__ import division, print_function

__author__ = "adrn <adrn@astro.princeton.edu>"

# Standard library
from collections import OrderedDict
import os

# Third-party
from astropy.table import Table
import numpy as np

numpy_type_map = dict()
numpy_type_map[np.float32] = 'types.REAL'
numpy_type_map[np.float64] = 'types.Numeric'
numpy_type_map[np.int16] = 'types.SmallInteger'
numpy_type_map[np.int32] = 'types.Integer'
numpy_type_map[np.int64] = 'types.BigInteger'
numpy_type_map[np.str_] = 'types.String'

def table_to_column_code(table, skip=None):
    """
    Convert an `~astropy.table.Table` to a dictionary of
    `sqlalchemy.Column` objects.
    """

    if skip is None:
        skip = []

    col_map = OrderedDict()
    for name in table.columns:
        if name in skip:
            continue

        dtype,_ = table.dtype.fields[name]
        sql_type = numpy_type_map[table[name].dtype.type]

        if len(dtype.shape) > 0:
            sql_type = 'postgresql.ARRAY({})'.format(sql_type)

        col_map[name.lower()] = ('{name} = Column("{name}", {type})'
                                 .format(name=name.lower(), type=sql_type))

    return col_map

def main(allVisit_file, allStar_file, rc_file, **kwargs):
    norm = lambda x: os.path.abspath(os.path.expanduser(x))
    allvisit_tbl = Table.read(norm(allVisit_file), format='fits', hdu=1)
    allstar_tbl = Table.read(norm(allStar_file), format='fits', hdu=1)
    rc_tbl = Table.read(norm(rc_file), format='fits', hdu=1)

    # Columns to skip
    allstar_skip = ['VISITS', 'ALL_VISITS', 'ALL_VISIT_PK', 'VISIT_PK']
    allvisit_skip = []
    rc_skip = ['VISITS', 'ALL_VISITS', 'ALL_VISIT_PK', 'VISIT_PK']

    # populate columns of the tables
    print("-------- AllStar --------")
    for name,col in table_to_column_code(allstar_tbl,
                                         skip=allstar_skip).items():
        print(col)

    print("\n"*4)
    print("-------------------------------------------------------------------")
    print("\n"*4)

    print("-------- AllVisit --------")
    for name,col in table_to_column_code(allvisit_tbl,
                                         skip=allvisit_skip).items():
        print(col)

    print("\n"*4)
    print("-------------------------------------------------------------------")
    print("\n"*4)

    print("-------- Red Clump --------")
    for name,col in table_to_column_code(rc_tbl, skip=rc_skip).items():
        print(col)

if __name__ == "__main__":
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="Initialize the TwoFace database.")

    parser.add_argument("--allstar", dest="allStar_file", required=True,
                        type=str, help="Path to APOGEE allStar FITS file.")
    parser.add_argument("--allvisit", dest="allVisit_file", required=True,
                        type=str, help="Path to APOGEE allVisit FITS file.")
    parser.add_argument("--redclump", dest="rc_file", required=True,
                        type=str, help="Path to APOGEE Red Clump catalog FITS "
                                       "file.")

    args = parser.parse_args()

    main(**vars(args))
