import torchaudio
from torchaudio.models.decoder import ctc_decoder, download_pretrained_files
import csv

from whisper.whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer

import os
import torch
import numpy as np
import random


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

beam_search_decoder = ctc_decoder(
    lexicon=files.lexicon,
    tokens=files.tokens,
    nbest=N,
    beam_size=500,
)

normalise = EnglishTextNormalizer()

# define function to read in sound file
def get_audio(path):
    speech, _ = torchaudio.load(path)
    return speech
    
file_list = '/fastdata/acq22mc/exp/diff_ll_audio/resources/filelists/ljspeech/valid.txt'
output_csv = '/fastdata/acq22mc/exp/diff_ll_audio/wav2vec2/nbest_w2v_no_lm.csv'

with open(output_csv, 'w', newline='') as output:
    fieldnames = ['audio_path', 'reference', 'wer', 'w2v_hyp', 'w2v_score']
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    with open(file_list) as f:
        for i, line in enumerate(f):

            wav_path, reference = line.split('|')

            reference = normalise(reference[:-1])

            speech = get_audio(wav_path)

            am_results, _ = acoustic_model(speech)

            beam_search_result = beam_search_decoder(am_results)

            hypotheses = []
            scores = []

            print(len(beam_search_result[0]))

            for n, result in enumerate(beam_search_result[0]):
                transcript = " ".join(result.words).strip()
                transcript = normalise(transcript)
                score = result.score

                wer = torchaudio.functional.edit_distance(reference.split(), transcript.split())

                out_dict = {'audio_path':wav_path,
                                'reference':reference,
                                'wer':wer,
                                'w2v_hyp':transcript,
                                'w2v_score':score}
                
                print(out_dict.values())

                writer.writerow(out_dict)
