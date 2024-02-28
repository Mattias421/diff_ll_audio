import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from omegaconf import DictConfig, OmegaConf
import hydra

import sys
sys.path.append('../speechdiff')
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
import json
import logging
logger = logging.getLogger()

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


@hydra.main(version_base=None, config_path='../config', config_name='config')
def main(cfg: DictConfig):
    device = torch.device(f'cuda:{cfg.training.gpu}')
    os.makedirs(cfg.eval.out_dir, exist_ok=True)

    dataset = TextMelDataset(cfg.eval.split, cfg)
    batch_collate = TextMelBatchCollate()
    loader = DataLoader(dataset=dataset, batch_size=cfg.training.batch_size,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=cfg.training.num_workers, shuffle=False)

    print('Initializing model...')
    model = GradTTS(cfg)
    model.load_state_dict(torch.load(cfg.eval.checkpoint, map_location=lambda loc, storage: loc))
    model.to(device).eval()
    print('Number of encoder parameters = %.2fm' % (model.encoder.nparams/1e6))
    print('Number of decoder parameters = %.2fm' % (model.decoder.nparams/1e6))

    vocoder = HIFIGAN.from_hparams(source='speechbrain/tts-hifigan-ljspeech')

    history = []

    with torch.no_grad():
        for batch in loader:
            x, x_lengths = batch['x'].to(device), batch['x_lengths'].to(device)
            y, y_lengths = batch['y'].to(device), batch['y_lengths'].to(device)

            score_model = model.get_score_model(x, x_lengths, y, y_lengths, spk=None)

            initial_noise = torch.distributions.Normal(score_model.mu_y, 1)
            initial_noise = initial_noise.sample().to(device)
            initial_noise = score_model.mu_y + torch.randn_like(score_model.mu_y)
            initial_noise = initial_noise.to(device)

            sample_half = ode_sample(score_model,
                                initial_noise,
                                (1 - 1e-8), 0.5,
                                cfg.model.decoder.beta_min,
                                cfg.model.decoder.beta_max,
                                atol=1e-9,
                                rtol=1e-9
                                )

            sample = ode_sample(score_model,
                                initial_noise,
                                (1 - 1e-8), 1e-8,
                                cfg.model.decoder.beta_min,
                                cfg.model.decoder.beta_max,
                                atol=1e-9,
                                rtol=1e-9
                                )
            

            ll_init, prior, delta, latent = log_likelihood(score_model,
                                                    initial_noise,
                                                    1e-8, 1,
                                                    cfg.model.decoder.beta_min,
                                                    cfg.model.decoder.beta_max,
                                                    atol=1e-5,
                                                    rtol=1e-5)

            ll_half, prior, delta, latent = log_likelihood(score_model,
                                                    sample_half,
                                                    1e-8, 1,
                                                    cfg.model.decoder.beta_min,
                                                    cfg.model.decoder.beta_max,
                                                    atol=1e-5,
                                                    rtol=1e-5)

            ll_full, prior, delta, latent = log_likelihood(score_model,
                                                    sample_half,
                                                    1e-8, 1,
                                                    cfg.model.decoder.beta_min,
                                                    cfg.model.decoder.beta_max,
                                                    atol=1e-5,
                                                    rtol=1e-5)
            result = {'init':ll_init.item(),
                'half':ll_half.item(),
                'full':ll_full.item(),
                'len':y.shape(1)}
            logger.info(result)

            history.append(result)

            avg_full, avg_half, avg_init = 0, 0, 0
            for h in history:
                avg_full += h['full']
                avg_half += h['half']
                avg_init += h['init']

            logger.info(avg_full/len(history))
            logger.info(avg_half/len(history))
            logger.info(avg_init/len(history))
            with open('history.json', 'w') as f:
                json.dump(history, f)



if __name__ == '__main__':
    main()
