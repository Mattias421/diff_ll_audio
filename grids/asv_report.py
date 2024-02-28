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

class MyExplorer(Explorer):

    talks = 'total'

    def get_grid_metrics(self) -> List[_Node]:
        return [
            tt.group('talks',[
                    
                    tt.group(talk_id, [
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

        results = {}

        for talk in df['talk_id'].unique():
            df_t = df[df['talk_id'] == talk]

            results[talk] = {'src':df_t['asv_src'].mean(),
                             'tgt':df_t['asv_tgt'].mean(),
            }

        results['total'] = {'src':df['asv_src'].mean(),
                             'tgt':df['asv_tgt'].mean(),
            }

        self.talks = list(results.keys())

        return {'talks':results, 'completion':completion}

@MyExplorer
def explorer(launcher):
    sub = launcher.bind({'uda.report_asv':True,
                         'uda.kernel_size':5,
                         'uda.sigma':1.0,
                         'preprocessing.rvad':False,
                         'preprocessing.source_sep':False})
    
    with launcher.job_array():
        sub()
        for enc, dec in product(['prior_sampling', 'forward_diffusion', 'ode'], ['euler', 'ode']):
            sub({'uda.encoding':enc,
                 'uda.decoding':dec})
            
            