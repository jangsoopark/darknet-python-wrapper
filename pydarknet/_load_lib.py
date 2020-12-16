import ctypes
import os


# This library doesn't assume to be used without GPU
def load():
    lib_dir = os.path.join(os.path.dirname(__file__), 'lib')
    os.environ['PATH'] = lib_dir + ';' + os.environ['PATH']

    if os.name == 'nt':
        lib_path = os.path.join(lib_dir, 'darknet.dll')
    else:
        lib_path = os.path.join(lib_dir, 'libdarnknet.so')

    lib = ctypes.CDLL(lib_path, ctypes.RTLD_GLOBAL)

    return lib
