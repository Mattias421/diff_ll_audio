import typing as tp
from dora import Explorer
import treetable as tt
from treetable.table import _Node
from typing import List
from itertools import product

class MyExplorer(Explorer):
    def get_grid_metrics(self) -> List[_Node]:
        return [tt.leaf('wer', '.4f'),
                tt.leaf('wer_per_talk')]

@MyExplorer
def explorer(launcher):
    uda_sub = launcher.bind({'uda.kernel_size':5,
                             'uda.sigma':1.0,
                             'uda.encoding':'ode',
                             'uda.decoding':'ode'})
    
    base_sub = launcher.bind({'asr.baseline':True})

    for rvad, source_sep, use_lm in product([False, True], [False, True], [False, True]):
        preproc = {'preprocessing.rvad':rvad,
                   'preprocessing.source_sep':source_sep,
                   'asr.use_lm':use_lm}
        
        uda_sub(preproc)
        base_sub(preproc)