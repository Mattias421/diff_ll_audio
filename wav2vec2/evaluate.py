import pandas as pd
from jiwer import wer
from scipy.stats import spearmanr
import numpy as np
import torchaudio

print(f'Evaluating wav2vec librispeech ft 10min')
results = pd.read_csv(f'/fastdata/acq22mc/exp/diff_ll_audio/gtts_ll_w2v.csv')
references = []
whisp_bests = []
gtts_bests = []
oracle_best = []
oracle_worst = []
rand = []
wer_whisp_covariances = []
wer_gtts_covariances = []
whisp_gtts_covariances = []

for audio_path in set(results['audio_path']):
    nbl = results[results['audio_path'] == audio_path]

    references.append(nbl.iloc[0]['reference'])
    whisp_bests.append(nbl.loc[nbl['w2v_score'].idxmax(), 'w2v_hyp']) # max nll
    gtts_bests.append(nbl.loc[nbl['gtts_ll'].idxmax(), 'w2v_hyp']) # min ll
    oracle_best.append(nbl.loc[nbl['wer'].idxmin(), 'w2v_hyp'])
    oracle_worst.append(nbl.loc[nbl['wer'].idxmax(), 'w2v_hyp'])
    rand.append(nbl.sample()['w2v_hyp'].iloc[0])

    wer_whisp_cov = nbl['wer'].cov(nbl['w2v_score'])
    wer_gtts_cov = nbl['wer'].cov(nbl['gtts_ll'])
    whisp_gtts_cov = nbl['w2v_score'].cov(nbl['gtts_ll'])

print('wer')
print(f'w2v: {wer(references, whisp_bests)}')
print(f'GTTS: {wer(references, gtts_bests)}')
print(f'Oracle best: {wer(references, oracle_best)}')
print(f'Oracle worst: {wer(references, oracle_worst)}')
print(f'Random: {wer(references, rand)}')

print(list(zip(gtts_bests, whisp_bests, references)))

# concat all references into one long string, do same for transcript
references = ' '.join(references).split()
transcripts = ' '.join(whisp_bests).split()

wer = torchaudio.functional.edit_distance(references, transcripts) / len(references)

print(wer)

# his defense being that he had intended to commit sissie but that on the appearance of this fraser who had wronged him
# his defense being that he had intended to commit suicide but that on the appearance of this officer who had wronged him

# ('these principles of holland are essential to a correct interpretation of the facts of more foley', 
# 'these principles of harmony are essential to a correct interpretation of the facts of more fully', 
# 'these principles of homology are essential to a correct interpretation of the facts of morphology'