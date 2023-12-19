import typing as tp
from dora import Explorer
import treetable as tt
from treetable.table import _Node
from typing import List
from itertools import product

from jiwer import wer as jwer
import numpy as np
import pandas as pd

def bits_per_sample(log_likelihood, audio_len):
    bpd = log_likelihood / np.log(2)
    bps = bpd / audio_len
    return bps

class MyExplorer(Explorer):
    def get_grid_metrics(self) -> List[_Node]:
        return [           
            tt.leaf('wer', '.4f'),

            tt.group('LL (BPS)',[
                tt.leaf('tgt', '.4f'),
                tt.leaf('src', '.4f')
            ]),

            tt.leaf('completion', '.2f')
                ]

    def process_history(self, history: List[dict]) -> dict:

        N = len(history)

        if N == 0:
            return {}
        
        completion = N / history[0]['len_data']       

        sum_tgt = 0
        sum_src = 0

        for metrics in history:
            audio_len = metrics['audio_len']
            sum_tgt += bits_per_sample(metrics['ll_tgt'], audio_len)
            sum_src += bits_per_sample(metrics['ll_src'], audio_len)

        ll = {'tgt':sum_tgt / N,
              'src':sum_src / N}

        references = [m['reference'] for m in history]
        hypothesis = [m['hypothesis'] for m in history]
        wer = jwer(references, hypothesis)
        
        return {'wer':wer, 'LL (BPS)':ll, 'completion':completion}

@MyExplorer
def explorer(launcher):
    sub = launcher.bind({'uda.report_loglikelihood':True,
                         'uda.kernel_size':5,
                         'uda.sigma':1.0,
                         'preprocessing.rvad':False,
                         'preprocessing.source_sep':False})
    
    with launcher.job_array():
        for enc, dec in product(['prior_sampling', 'forward_diffusion', 'ode'], ['euler', 'ode']):
            sub({'uda.encoding':enc,
                 'uda.decoding':dec})
            
            