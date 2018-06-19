import numpy as np
cimport cython
cimport numpy as np
np.import_array()

def depthFilter(np.ndarray[np.float32_t, ndim=2] c_maxDepth,
                np.ndarray[np.float32_t, ndim=2] c_depth,
                np.ndarray[np.uint8_t, ndim=3] c_foreground):
    cdef int x, y
    for x in range(512):
        for y in range(424):
            if c_maxDepth[y,x] != 0 and not c_depth[y,x] < (c_maxDepth[y,x] - 100):
                c_foreground[y,x,0] = 0
                c_foreground[y,x,1] = 0
                c_foreground[y,x,2] = 0
                c_foreground[y,x,3] = 0


