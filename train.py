# from speechdiff.model.tts import GradTTS
# from speechdiff.data import mel_spectrogram

from speechbrain.pretrained import HIFIGAN

from omegaconf import DictConfig, OmegaConf

from dora import get_xp, hydra_main

from .whisper.whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer

import os
import torch
import numpy as np
import random
from typing import Callable
from torchaudio.datasets import TEDLIUM
from torchaudio.functional import resample
import torchaudio
from torchaudio.models.decoder import ctc_decoder, download_pretrained_files
import pandas as pd

from jiwer import wer as jwer

from torchvision.transforms import GaussianBlur

from .preprocessing.vad import rVADfast

import sys
sys.path.append('/fastdata/acq22mc/exp/diff_ll_audio/speechdiff')
from model import GradTTS
from utils import save_plot
from data import mel_spectrogram
from likelihood.loglikelihood import ode_sample, log_likelihood

import logging
logger = logging.getLogger()


def fix_len_compatibility(length, num_downsamplings_in_unet=2):
    while True:
        if length % (2**num_downsamplings_in_unet) == 0:
            return length
        length += 1

def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(int(max_length), dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)



class ScoreModel(torch.nn.Module):
    def __init__(self, estimator, mu, mask):
        super().__init__()
        self.y_mask = mask
        self.mu_y =  mu
        self.estimator = estimator

    def forward(self, x, t):
        x_t = self.estimator(x=x, mask=self.y_mask, mu=self.mu_y, t=t, spk=None)
        return x_t

def pad_audio(audio):
    audio_len = audio.shape[-1]

    max_len = fix_len_compatibility(audio.shape[-1])
    mask = sequence_mask(torch.LongTensor([[audio.size(-1)]]), 
                         max_len).to(audio)

    pad = torch.zeros((audio.size(0), audio.size(1), max_len)).to(audio)
    pad[:, :, :audio.size(-1)] = audio

    return pad, mask, audio_len

def adapt_speech(audio: torch.Tensor, decoder: torch.nn.Module, blur: Callable, 
                 encoding: str = 'forward_diffusion', decoding: str = 'euler', t_max: int = 1, report_ll: bool = False):

    audio, mask, audio_len = pad_audio(audio)

    mu = blur(audio)
    mu *= mask

    if report_ll:
        model = ScoreModel(decoder.estimator, mu, mask)
        ll_tgt, _, _, _ = log_likelihood(model,
                                      audio,
                                      0,
                                      1.0,
                                      0.05,
                                      20,
                                      atol=1e-5, rtol=1e-5)

    t_vec = t_max * torch.ones(audio.size(0), device=audio.device) # TODO: is t_max 1 or 999 here?

    if encoding == 'prior_sampling':
        x_T = mu + torch.randn_like(mu, device=mu.device)
    elif encoding == 'forward_diffusion':
        x_T, _ = decoder.forward_diffusion(audio, mask, mu, t_vec) 
    elif encoding == 'ode':
        model = ScoreModel(decoder.estimator, mu, mask)
        x_T = ode_sample(model,
                        audio,
                        0,
                        1.0,
                        0.05,
                        20,
                        atol=1e-5, rtol=1e-5) # static betamax/min for now
    x_T *= mask

    if decoding == 'euler':
        x_0 = decoder(x_T, mask, mu, n_timesteps=100) 

    elif decoding == 'ode':
        model = ScoreModel(decoder.estimator, mu, mask)
        x_0 = ode_sample(model,
                        x_T,
                        1.0,
                        0,
                        0.05,
                        20,
                        atol=1e-5, rtol=1e-5) # static betamax/min for now
    
    audio = x_0[:, :, :audio_len]

    if report_ll:
        model = ScoreModel(decoder.estimator, mu, mask)
        ll_src, _, _, _ = log_likelihood(model,
                                      x_0,
                                      0,
                                      1.0,
                                      0.05,
                                      20,
                                      atol=1e-5, rtol=1e-5)

        return audio, ll_tgt[0], ll_src[0], audio_len
    
    else:
        return audio



seed = 1

os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)


def uda(cfg: DictConfig):

    device = 'cuda:0'
    xp = get_xp()

    # load model and processor
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_10M
    acoustic_model = bundle.get_model()
    acoustic_model.eval()

    files = download_pretrained_files("librispeech-4-gram")

    # default weights
    LM_WEIGHT = 3.23 
    WORD_SCORE = -0.26

    N = 100

    if cfg.asr.use_lm:
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

    # source seperation
    if cfg.preprocessing.source_sep == True:
        source_separator = torchaudio.pipelines.CONVTASNET_BASE_LIBRI2MIX.get_model()
        source_separator.eval().cuda()

    normalise = EnglishTextNormalizer()
    blur = GaussianBlur(kernel_size=cfg.uda.kernel_size, sigma=cfg.uda.sigma)
    resample = torchaudio.transforms.Resample(cfg.data.target_sample_rate,
                                              cfg.data.sample_rate)


    dataset = TEDLIUM(root='/fastdata/acq22mc/data/tedlium3/',
                    release='release3',
                    subset='dev',
                    audio_ext='.wav')
    
    logger.info(f'Loaded {len(dataset)} samples')
    
    
    gtts = GradTTS(cfg)
    state_dict = torch.load(cfg.eval.checkpoint, map_location=lambda loc, storage: loc)

    gtts.load_state_dict(state_dict)
    gtts.eval()
    gtts.to(device)

    vocoder = HIFIGAN.from_hparams(source='speechbrain/tts-hifigan-ljspeech')



    # main training loop

    with torch.no_grad():
        for i, (speech, sr, reference, talk_id, speaker_id, id) in enumerate(dataset):

            reference = normalise(reference)


            if reference == 'ignore time segment in scoring':
                continue

            if cfg.preprocessing.rvad == True:
                speech = rVADfast(speech[0], sr).unsqueeze(0)
            
            if cfg.preprocessing.source_sep == True:
                sep_speech = source_separator(speech.unsqueeze(0).cuda())[0, 1].unsqueeze(0).cpu() # speech should be [1 x 1 x time]
                gain_diff = (speech.pow(2).mean().sqrt()) / (sep_speech.pow(2).mean().sqrt())
                speech = torchaudio.transforms.Vol(gain_diff)(sep_speech)

            speech = resample(speech) # speech should be [1 x time]

            if cfg.asr.baseline == False:

                speech_spec = mel_spectrogram(speech,
                                            n_fft=cfg.data.n_fft,
                                            num_mels=cfg.data.n_feats,
                                            sampling_rate=cfg.data.sample_rate,
                                            hop_size=cfg.data.hop_length,
                                            win_size=cfg.data.win_length,
                                            fmin=cfg.data.f_min,
                                            fmax=cfg.data.f_max)
                
                speech_spec = speech_spec.cuda()

                if cfg.uda.report_loglikelihood:
                    adapted_speech_spec, ll_tgt, ll_src, audio_len = adapt_speech(speech_spec, 
                                                    gtts.decoder,
                                                    blur, 
                                                    encoding=cfg.uda.encoding, 
                                                    decoding=cfg.uda.decoding,
                                                    report_ll=True)
                else:
                    adapted_speech_spec = adapt_speech(speech_spec, 
                                                    gtts.decoder,
                                                    blur, 
                                                    encoding=cfg.uda.encoding, 
                                                    decoding=cfg.uda.decoding,)
                
                adapted_speech = vocoder.decode_batch(adapted_speech_spec.squeeze())

                torchaudio.save(f'source_{speaker_id}.wav', adapted_speech, cfg.data.sample_rate)

                speech = adapted_speech

            am_results, _ = acoustic_model(speech)

            beam_search_result = beam_search_decoder(am_results)

            hypothesis = ' '.join(beam_search_result[0][0].words).strip()
            hypothesis = normalise(hypothesis)

            wer = torchaudio.functional.edit_distance(reference.split(), hypothesis.split()) / len(reference)

            result_data = {
                'id': id,
                'talk_id': talk_id,
                'speaker_id': speaker_id,
                'reference': reference,
                'hypothesis': hypothesis,
                'sentence_wer': wer,
                'len_data':len(dataset)
            }


            logger.info(result_data.values())

            if cfg.uda.report_loglikelihood:
                result_data.update({
                    'll_tgt': ll_tgt,
                    'll_src': ll_src,
                    'audio_len': audio_len
                })

            xp.link.push_metrics(result_data)



@hydra_main(version_base=None, config_path='./config', config_name='uda')
def main(cfg):
    xp = get_xp()
    logger.info(xp.sig)

    uda(xp.cfg)

    logger.info('Finished :3')

if __name__ == '__main__':
    # print(main.__module__)
    main()