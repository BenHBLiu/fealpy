from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger
from .optimizer_base import Optimizer

"""
Marine Predators Algorithm

Reference:
~~~~~~~~~~
Hang Su, Dong Zhao, Ali Asghar Heidari, Lei Liu, Xiaoqin Zhang, Majdi Mafarja, Huiling Chen. 
RIME: A physics-based optimization. 
Neurocomputing, 2023, 532: 183-214.

"""
class CrowDrinkingWaterAlg(Optimizer):
    def __init__(self, option) -> None:
        super().__init__(option)


    def run(self):
        options = self.options
        x = options["x0"]
        N = options["NP"]
        fit = self.fun(x)[:, None]
        MaxIT = options["MaxIters"]
        dim = options["ndim"]
        lb, ub = options["domain"]
        gbest_index = bm.argmin(fit)
        gbest = x[gbest_index]
        gbest_f = fit[gbest_index]

        P = 0.9
        # curve = bm.zeros((1, MaxIT))
        for it in range(0, MaxIT):
            r = bm.random.rand(N, 1)
            x = ((r < P) * (x + bm.random.rand(N, 1) * (ub -x) + bm.random.rand(N, 1) * lb) + 
                (r >= P) * ((2 * bm.random.rand(N, 1) - 1) * (ub - lb) + lb))
            x = x + (lb - x) * (x < lb) + (ub - x) * (x > ub)
            fit = self.fun(x)[:, None]
            gbest_idx = bm.argmin(fit)
            (gbest, gbest_f) = (x[gbest_idx], fit[gbest_idx]) if fit[gbest_idx] < gbest_f else (gbest, gbest_f)
            # curve[0, it] = gbest_f
        return gbest, gbest_f