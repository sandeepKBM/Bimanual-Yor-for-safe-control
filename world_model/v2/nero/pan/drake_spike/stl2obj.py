"""Minimal STL (binary/ascii) -> OBJ converter (Drake MJCF parser accepts only .obj). Vertices welded by exact value."""
import re, struct, sys, os
import numpy as np

def read_stl(path):
    b = open(path, 'rb').read()
    if len(b) >= 84 and 84 + 50 * struct.unpack('<I', b[80:84])[0] == len(b):
        n = struct.unpack('<I', b[80:84])[0]
        a = np.frombuffer(b[84:], dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]), count=n)
        tri = a['v'].astype(np.float64)
    else:
        tri = np.array(re.findall(rb'vertex\s+(\S+)\s+(\S+)\s+(\S+)', b), dtype=np.float64).reshape(-1, 3, 3)
    return tri

def convert(src, dst):
    tri = read_stl(src)
    v, inv = np.unique(tri.reshape(-1, 3), axis=0, return_inverse=True)
    f = inv.reshape(-1, 3)
    ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    f = f[ok]
    with open(dst, 'w') as fh:
        for p in v: fh.write('v %.9g %.9g %.9g\n' % tuple(p))
        for t in f + 1: fh.write('f %d %d %d\n' % tuple(t))
    return len(v), len(f)

if __name__ == '__main__':
    print(convert(sys.argv[1], sys.argv[2]))
