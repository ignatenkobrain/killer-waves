#!/usr/bin/env python3

import numpy as np
import pyopencl as cl

class CL(object):
    def __init__(self):
        t_np = np.arange(20000, 50000, dtype=np.float32)

        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

        self.mf = cl.mem_flags
        self.t_g = cl.Buffer(
            self.ctx,
            self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
            hostbuf=t_np)

        f = open("ex.cl", "r")
        fstr = "".join(f.readlines())
        f.close()
        self.prg = cl.Program(self.ctx, fstr).build()

        self.res_g = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, t_np.nbytes)
        self.prg.proc(self.queue, t_np.shape, None, self.t_g, self.res_g)

        res_np = np.empty_like(t_np)
        cl.enqueue_copy(self.queue, res_np, self.res_g)

        # Check on CPU with Numpy:
        print(res_np)

if __name__ == "__main__":
    cl = CL()
