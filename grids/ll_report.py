import typing as tp
from dora import Explorer
import treetable as tt
from treetable.table import _Node
from typing import List
from itertools import product

from jiwer import wer as jwer
import numpy as np
import pandas as pd
from collections import defaultdict

def bits_per_sample(log_likelihood, audio_len):
    # bpd = log_likelihood / np.log(2)
    # bps = bpd / audio_len
    return log_likelihood / np.log(2) / audio_len

class MyExplorer(Explorer):
    def get_grid_metrics(self) -> List[_Node]:
        return [
            tt.group('talks',[
                    
                    tt.group(talk_id, [
                        tt.leaf('wer', '.2f'),
                        tt.leaf('tgt', '.2f'),
                        tt.leaf('src', '.2f'),
                    ], align='<')

                    for talk_id in self.talks
            ])
                ]

    def process_history(self, history: List[dict]) -> dict:

        N = len(history)

        if N == 0:
            return {}
        
        completion = N / history[0]['len_data']       

        df = pd.DataFrame(history)

        df['src_bps'] = df.apply(lambda x : bits_per_sample(x['ll_src'], x['audio_len']), axis=1)
        df['tgt_bps'] = df.apply(lambda x : bits_per_sample(x['ll_tgt'], x['audio_len']), axis=1)

        print()

        wer = jwer(df['reference'].to_list(), df['hypothesis'].to_list())

        results = {}

        for talk in df['talk_id'].unique():
            df_t = df[df['talk_id'] == talk]

            refs = df_t['reference'].to_list()
            hyps = df_t['hypothesis'].to_list()

            results[talk] = {'wer':jwer(refs, hyps),
                             'src':df_t['src_bps'].mean(),
                             'tgt':df_t['tgt_bps'].mean(),
            }

        results['total'] = {'wer':wer,
                            'tgt':df['tgt_bps'].mean(),
                            'src':df['src_bps'].mean(),
                            }
        print(df['src_bps'].mean())

        self.talks = list(results.keys())

        print(results)
        
        return {'talks':results, 'completion':completion}

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
            
            