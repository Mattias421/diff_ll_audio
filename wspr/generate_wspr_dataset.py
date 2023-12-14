from whisper import whisper
from whisper.whisper.model import Whisper
from whisper.whisper.tokenizer import Tokenizer
from whisper.whisper.utils import compression_ratio
from whisper.whisper.decoding import DecodingOptions, DecodingResult, DecodingTask
from whisper.whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from jiwer import wer
import yaml
from tqdm import tqdm
import csv

def get_audio_embedding(model: Whisper, audio_path: str) -> torch.Tensor:
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    encoding: torch.Tensor = model.embed_audio(mel[None, :, :]) # batch size of 1
    return encoding.detach()


def decode_audio_embedding(model: Whisper, embedding: torch.Tensor) -> DecodingResult:
    decoding_options = DecodingOptions(
        language='en',
        beam_size=10,
        temperature=0,
        fp16=torch.cuda.is_available() 
    )
    # result is squeezed since we're dealing with one < 30s segment
    result = whisper.decode(model, embedding.squeeze(0), decoding_options)
    return result

@torch.no_grad()
def get_n_best_hypothesis(decoding_task: DecodingTask, mel: Tensor) -> List[DecodingResult]:
    decoding_task.decoder.reset()
    tokenizer: Tokenizer = decoding_task.tokenizer
    n_audio: int = mel.shape[0]

    audio_features: Tensor = decoding_task._get_audio_features(mel)  # encoder forward pass
    tokens: Tensor = torch.tensor([decoding_task.initial_tokens]).repeat(n_audio, 1)

    # detect language if requested, overwriting the language token
    languages, language_probs = decoding_task._detect_language(audio_features, tokens)
    if decoding_task.options.task == "lang_id":
        return [
            DecodingResult(
                audio_features=features, language=language, language_probs=probs
            )
            for features, language, probs in zip(
                audio_features, languages, language_probs
            )
        ]

    # repeat text tensors by the group size, for beam search or best-of-n sampling
    tokens = tokens.repeat_interleave(decoding_task.n_group, dim=0).to(audio_features.device)

    # call the main sampling loop
    tokens, sum_logprobs, no_speech_probs = decoding_task._main_loop(audio_features, tokens)

    # reshape the tensors to have (n_audio, n_group) as the first two dimensions
    audio_features = audio_features[:: decoding_task.n_group]
    no_speech_probs = no_speech_probs[:: decoding_task.n_group]
    assert audio_features.shape[0] == len(no_speech_probs) == n_audio

    tokens = tokens.reshape(n_audio, decoding_task.n_group, -1)
    sum_logprobs = sum_logprobs.reshape(n_audio, decoding_task.n_group)

    n_best_hypothesis = [tokenizer.decode(t.tolist()[2:]) for t in tokens[0]]

    return [{'whisper_hypothesis':hyp[1:], 'whisper_sum_logprob':logprob.item()} for hyp, logprob in zip(n_best_hypothesis, sum_logprobs[0])]

model_size = 'large'
model = whisper.load_model(f'{model_size}.en')

decoding_options = DecodingOptions(
    language='en',
    beam_size=100,
    temperature=0,
    fp16=torch.cuda.is_available() 
)

decoding_task = DecodingTask(model, decoding_options)
normalise = EnglishTextNormalizer()


def n_best_pipeline(audio_path: str) -> List[Dict]:
    embedding = get_audio_embedding(model, audio_path)

    result = get_n_best_hypothesis(decoding_task, embedding)

    return result


file_list = '/fastdata/acq22mc/exp/diff_ll_audio/resources/filelists/ljspeech/valid.txt'

n_best_list_dataset = []

output_filepath = f'{model_size}.csv' 

with open(output_filepath, 'w', newline='') as file:
    fieldnames = ['audio_path', 'reference', 'wer', 'wspr_hyp', 'wspr_prob']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

with open(file_list) as f:
    for i, line in enumerate(tqdm(f)):
        wav_path, reference = line.split('|')

        reference = normalise(reference[:-1])

        n_best_list = n_best_pipeline(wav_path)

        hyp_set = set()

        for hyp in n_best_list:
            hypothesis = normalise(hyp['whisper_hypothesis'])

            if hypothesis not in hyp_set:
                error = wer(reference, hypothesis)

                with open(output_filepath, 'a', newline='') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writerow({
                        'audio_path':wav_path,
                        'reference': reference,
                        'wer':error,
                        'wspr_hyp':hypothesis,
                        'wspr_prob':hyp['whisper_sum_logprob']
                    })

                hyp_set.add(hypothesis)


