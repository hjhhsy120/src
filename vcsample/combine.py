from __future__ import print_function
from . import app
from . import deepwalk
import numpy as np

class combine(object):

    # jump_factor: prob to stop
    def __init__(self, g, args):
        self.ratio = args.combine
        self.batch_size = args.batch_size
        self.model1 = deepwalk.deepwalk(graph=g, fac=int(args.epoch_fac * self.ratio),
                    window=args.window_size, degree_bound=args.degree_bound, degree_power=args.degree_power)
        self.model2 = app.APP(graph=g, jump_factor=args.app_jump_factor,
                        sample=int(args.epoch_fac * (1 - self.ratio)), step=args.app_step)

    def sample_v(self, batch_size):
        v1s = self.model1.sample_v(int(self.batch_size * self.ratio))
        v2s = self.model2.sample_v(int(self.batch_size * (1 - self.ratio)))
        for v1 in v1s:
            self.l1 = len(v1)
            v2 = next(v2s)
            yield np.append(v1, v2)

    def sample_c(self, h):
        c1 = self.model1.sample_c(h[:self.l1])
        c2 = self.model2.sample_c(h[self.l1:])
        return c1 + c2


