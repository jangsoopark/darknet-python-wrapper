import ctypes


class BOX(ctypes.Structure):
    _fields_ = [
        ('x', ctypes.c_float),
        ('y', ctypes.c_float),
        ('w', ctypes.c_float),
        ('h', ctypes.c_float),
    ]


class DETECTION(ctypes.Structure):
    _fields_ = [
        ("bbox", BOX),
        ("classes", ctypes.c_int),
        ("prob", ctypes.POINTER(ctypes.c_float)),
        ("mask", ctypes.POINTER(ctypes.c_float)),
        ("objectness", ctypes.c_float),
        ("sort_class", ctypes.c_int),
        ("uc", ctypes.POINTER(ctypes.c_float)),
        ("points", ctypes.c_int),
        ("embeddings", ctypes.POINTER(ctypes.c_float)),
        ("embedding_size", ctypes.c_int),
        ("sim", ctypes.c_float),
        ("track_id", ctypes.c_int)
    ]


class DETNUMPAIR(ctypes.Structure):
    _fields_ = [
        ("num", ctypes.c_int),
        ("dets", ctypes.POINTER(DETECTION))
    ]


class IMAGE(ctypes.Structure):
    _fields_ = [
        ("w", ctypes.c_int),
        ("h", ctypes.c_int),
        ("c", ctypes.c_int),
        ("data", ctypes.POINTER(ctypes.c_float))
    ]


class METADATA(ctypes.Structure):
    _fields_ = [
        ("classes", ctypes.c_int),
        ("names", ctypes.POINTER(ctypes.c_char_p))
    ]
