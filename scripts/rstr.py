# rstr.py: write Twill save files ("RSTR" format) directly from Python.
#
# This is data-prep tooling, not the runtime. The Twill runtime reads these
# files with its native `load`, which is fast; the format is small and fully
# specified in the Twill interpreter (internal/interp/serialize.go):
#
#   header : magic b"RSTR", version byte 0x01
#   value  : tag byte + payload, little-endian throughout
#     tensor 'T' : int32 rank, int32[rank] dims, int64 n, float64[n] data
#     list   'L' : int32 count, then each value
#     record 'R' : int32 nkeys, then (int32 keylen, key bytes, value) per key
#     str    'S' : int32 len, bytes
#     int    'I' : int64
#     bool   'B' : 1 byte
#     unit   'U'
#
# A Python dict is written as a record (insertion order preserved), a list as a
# list, a numpy float array (any dtype) as a float64 tensor, a str as a Str, an
# int as an I64, a float as a rank-0 tensor.

import struct
import numpy as np


def _write_value(w, v):
    if isinstance(v, dict):
        w.write(b"R")
        w.write(struct.pack("<i", len(v)))
        for k, val in v.items():
            kb = k.encode("utf-8")
            w.write(struct.pack("<i", len(kb)))
            w.write(kb)
            _write_value(w, val)
    elif isinstance(v, (list, tuple)):
        w.write(b"L")
        w.write(struct.pack("<i", len(v)))
        for it in v:
            _write_value(w, it)
    elif isinstance(v, np.ndarray):
        a = np.ascontiguousarray(v, dtype="<f8")
        w.write(b"T")
        w.write(struct.pack("<i", a.ndim))
        for d in a.shape:
            w.write(struct.pack("<i", int(d)))
        w.write(struct.pack("<q", int(a.size)))
        w.write(a.tobytes())
    elif isinstance(v, str):
        b = v.encode("utf-8")
        w.write(b"S")
        w.write(struct.pack("<i", len(b)))
        w.write(b)
    elif isinstance(v, bytes):
        w.write(b"S")
        w.write(struct.pack("<i", len(v)))
        w.write(v)
    elif isinstance(v, bool):
        w.write(b"B")
        w.write(b"\x01" if v else b"\x00")
    elif isinstance(v, int):
        w.write(b"I")
        w.write(struct.pack("<q", v))
    elif isinstance(v, float):
        # a rank-0 tensor, which is what a Num becomes
        w.write(b"T")
        w.write(struct.pack("<i", 0))
        w.write(struct.pack("<q", 1))
        w.write(struct.pack("<d", v))
    else:
        raise TypeError("cannot serialize %r" % type(v))


def save(value, path):
    with open(path, "wb") as w:
        w.write(b"RSTR")
        w.write(b"\x01")
        _write_value(w, value)
