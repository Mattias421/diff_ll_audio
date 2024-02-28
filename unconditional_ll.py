import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

from omegaconf import DictConfig, OmegaConf
import hydra

import sys
sys.path.append('speech-diff')
from model import GradTTS
from utils import save_plot
from text.symbols import symbols
from likelihood import log_likelihood, ode_sample

from speechbrain.pretrained import HIFIGAN
from scipy.io.wavfile import write

from utils import intersperse, save_plot
from text import text_to_sequence, cmudict

from data import TextMelDataset, TextMelBatchCollate

import os
import yaml
from tqdm import tqdm

import logging
logger = logging.getLogger()

import random

import pandas as pd
import csv

class NBLDataset(Dataset):
    def __init__(self, n_best_list_csv, text_mel_dataset):
        # Load CSV data using pandas
        df = pd.read_csv(n_best_list_csv)

        # Assuming your CSV has the following columns: audio_path, reference, wer, whisper_hypothesis, whisper_sum_logprob
        self.n_best_list = df.to_dict(orient='records')

        self.num_entries = len(self.n_best_list)
        self.text_mel_dataset = text_mel_dataset

        logger.info(self.num_entries)

    def __len__(self):
        return self.num_entries
    
    def __getitem__(self, idx):
        nbl_entry = self.n_best_list[idx]
        text = nbl_entry['w2v_hyp']

        text_sequence = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=self.text_mel_dataset.cmudict), len(symbols)))

        file_path = nbl_entry['audio_path']
        mel = self.text_mel_dataset.get_mel(file_path)

        logger.info(idx)
        logger.info(text)
        logger.info(file_path)

        return {'x': text_sequence, 'y': mel, 'row':nbl_entry}

seed = 1

os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


@hydra.main(version_base=None, config_path='./config', config_name='config')
def main(cfg: DictConfig):
    device = torch.device(f'cuda:{cfg.training.gpu}')
    os.makedirs(cfg.eval.out_dir, exist_ok=True)

    nbl_file = f'/fastdata/acq22mc/exp/diff_ll_audio/nbest_w2v_no_lm.csv'
    df = pd.read_csv(nbl_file)

    # Assuming your CSV has the following columns: audio_path, reference, wer, whisper_hypothesis, whisper_sum_logprob
    n_best_list = df.to_dict(orient='records')

    text_mel_dataset = TextMelDataset(cfg.eval.split, cfg)
    dataset = NBLDataset(nbl_file, text_mel_dataset)
    batch_collate = TextMelBatchCollate()
    loader = DataLoader(dataset=dataset, batch_size=cfg.training.batch_size,
                        collate_fn=batch_collate, drop_last=False,
                        num_workers=0, shuffle=False)

    print('Initializing model...')
    model = GradTTS(cfg)
    model.load_state_dict(torch.load(cfg.eval.checkpoint, map_location=lambda loc, storage: loc))
    model.to(device).eval()
    print('Number of encoder parameters = %.2fm' % (model.encoder.nparams/1e6))
    print('Number of decoder parameters = %.2fm' % (model.decoder.nparams/1e6))

    output_csv_path = f'/fastdata/acq22mc/exp/diff_ll_audio/gtts_ll_w2v_no_lm.csv'

    with open(output_csv_path, 'w', newline='') as output_csv:
        fieldnames = ['audio_path', 'reference', 'wer', 'w2v_hyp', 'w2v_score', 'gtts_ll', 'gtts_prior', 'gtts_delta']
        writer = csv.DictWriter(output_csv, fieldnames=fieldnames)
        writer.writeheader()

        with torch.no_grad():
            for i, batch in enumerate(loader):
                x, x_lengths = batch['x'].to(device), batch['x_lengths'].to(device)
                y, y_lengths = batch['y'].to(device), batch['y_lengths'].to(device)

                score_model = model.get_score_model(x, x_lengths, y, y_lengths, spk=None)

                ll, prior, delta, latent = log_likelihood(score_model,
                                                        y,
                                                        1e-8, 1,
                                                        cfg.model.decoder.beta_min,
                                                        cfg.model.decoder.beta_max,
                                                        atol=1e-5,
                                                        rtol=1e-5)

                logger.info(f'Log-likelihood {ll}')
                entry = n_best_list[i]
                entry['gtts_ll'] = ll.item()
                entry['gtts_prior'] = prior.item()
                entry['gtts_delta'] = delta.item()

                writer.writerow(entry) 

if __name__ == '__main__':
    main()
