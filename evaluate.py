import pandas as pd
from jiwer import wer
from scipy.stats import spearmanr
import numpy as np
import torchaudio

results = pd.read_csv('baseline_no_lm.csv')

print(wer(results['reference'].to_list(), results['hypothesis'].to_list()))