import typing as tp
from dora import Explorer
import treetable as tt
from treetable.table import _Node
from typing import List

class MyExplorer(Explorer):
    def get_grid_metrics(self) -> List[_Node]:
        return [tt.leaf('wer', '.4f')
                ]

@MyExplorer
def explorer(launcher):
    for kernel_size in [3, 5, 9]:
        for sigma in [0.1, 1.0, 2.0]:
            with launcher.job_array():
                for encoding in ['prior_sampling', 'forward_diffusion', 'ode']:
                    for decoding in ['euler', 'ode']:
                        sub = launcher.bind(
                            {
                                'uda.kernel_size':kernel_size,
                                'uda.sigma':sigma,
                                'uda.encoding':encoding,
                                'uda.decoding':decoding
                            }
                        )

                        sub()