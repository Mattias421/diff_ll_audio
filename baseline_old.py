import torchaudio
from torchaudio.models.decoder import ctc_decoder, download_pretrained_files
import csv

from whisper.whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer

import os
import torch
import numpy as np
import random
from torchaudio.datasets import TEDLIUM


seed = 1

os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# load model and processor
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_10M
acoustic_model = bundle.get_model()

files = download_pretrained_files("librispeech-4-gram")

# default weights
LM_WEIGHT = 3.23 
WORD_SCORE = -0.26

N = 100

use_lm = False

if use_lm:
    beam_search_decoder = ctc_decoder(
        lexicon=files.lexicon,
        tokens=files.tokens,
        nbest=N,
        beam_size=500,
        lm=files.lm,
        lm_weight=LM_WEIGHT,
        word_score=WORD_SCORE
    )
else:
    beam_search_decoder = ctc_decoder(
        lexicon=files.lexicon,
        tokens=files.tokens,
        nbest=N,
        beam_size=500,
    )

normalise = EnglishTextNormalizer()

dataset = TEDLIUM(root='/fastdata/acq22mc/data/tedlium3/',
                  release='release3',
                  subset='dev')

print(f'Testing {len(dataset)} samples of TEDLIUM3 devset on baseline model')

output_csv = '/fastdata/acq22mc/exp/diff_ll_audio/baseline_no_lm.csv'

with open(output_csv, 'w', newline='') as output:
    fieldnames = ['id', 'talk_id', 'speaker_id', 'reference', 'hypothesis', 'wer']
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for speech, _, reference, talk_id, speaker_id, id in dataset:

        reference = reference[:-1]

        if reference != 'ignore_time_segment_in_scoring':
            am_results, _ = acoustic_model(speech)

            beam_search_result = beam_search_decoder(am_results)

            hypothesis = ' '.join(beam_search_result[0][0].words).strip()


            reference = normalise(reference)
            hypothesis = normalise(hypothesis)

            wer = torchaudio.functional.edit_distance(reference.split(), hypothesis.split()) / len(reference)

            out_dict = {'id':id, 'talk_id':talk_id, 'speaker_id':speaker_id,
                        'reference':reference, 'hypothesis':hypothesis, 'wer':wer}
            
            print(out_dict.values())

            writer.writerow(out_dict)
