#!/usr/bin/env python3

import numpy as np
import pyopencl as cl

class CL(object):
    def __init__(self):
        a_np = np.array([0], dtype=np.float32)

        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

        self.mf = cl.mem_flags
        self.a_g = cl.Buffer(
            self.ctx,
            self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
            hostbuf=a_np)

        f = open("ex.cl", "r")
        fstr = "".join(f.readlines())
        f.close()
        self.prg = cl.Program(self.ctx, fstr).build()

        self.res_g = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, a_np.nbytes)
        self.prg.proc(self.queue, a_np.shape, None, self.a_g, self.res_g)

        res_np = np.empty_like(a_np)
        cl.enqueue_copy(self.queue, res_np, self.res_g)

        # Check on CPU with Numpy:
        print(res_np)

if __name__ == "__main__":
    cl = CL()
