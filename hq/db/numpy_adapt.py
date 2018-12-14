import sqlite3
import numpy as np

def adapt_np_integer(np_integer): return int(np_integer)
def adapt_np_int8(np_int8): return int(np_int8)
def adapt_np_int16(np_int16): return int(np_int16)
def adapt_np_int32(np_int32): return int(np_int32)
def adapt_np_int64(np_int64): return int(np_int64)
def adapt_np_uint8(np_uint8): return int(np_uint8)
def adapt_np_uint16(np_uint16): return int(np_uint16)
def adapt_np_uint32(np_uint32): return int(np_uint32)
def adapt_np_uint64(np_uint64): return int(np_uint64)

def adapt_np_float(np_float): return float(np.float)
def adapt_np_float16(np_float16): return float(np_float16)
def adapt_np_float32(np_float32): return float(np_float32)
def adapt_np_float64(np_float64): return float(np_float64)

def adapt_np_bool(np_bool): return int(np_bool)

sqlite3.register_adapter(np.integer, adapt_np_integer)
sqlite3.register_adapter(np.int8, adapt_np_int8)
sqlite3.register_adapter(np.int16, adapt_np_int16)
sqlite3.register_adapter(np.int32, adapt_np_int32)
sqlite3.register_adapter(np.int64, adapt_np_int64)
sqlite3.register_adapter(np.uint8, adapt_np_uint8)
sqlite3.register_adapter(np.uint16, adapt_np_uint16)
sqlite3.register_adapter(np.uint32, adapt_np_uint32)
sqlite3.register_adapter(np.uint64, adapt_np_uint64)
sqlite3.register_adapter(np.float, adapt_np_float)
sqlite3.register_adapter(np.float16, adapt_np_float16)
sqlite3.register_adapter(np.float32, adapt_np_float32)
sqlite3.register_adapter(np.float64, adapt_np_float64)
sqlite3.register_adapter(np.bool, adapt_np_bool)
