# create a tsv for each stem (vocals, drums, bass, other, mixture)
"""
<root-dir>
<audio-path-1>
<audio-path-2>
...
"""

import argparse
import os
import soundfile

SPLIT = 'test'

ROOT = f'/mnt/parscratch/users/cadenza/data/cad_icassp_2024/audio/at_mic_music/{SPLIT}'
ROOT_MANIFEST = f'/mnt/parscratch/users/cadenza/cadenza_friendz/clarity/recipes/cad_icassp_2024/baseline/k_means/manifest/{SPLIT}'

os.makedirs(os.path.dirname(ROOT_MANIFEST), exist_ok=True)
vocals_f = open(os.path.join(ROOT_MANIFEST, f'vocals.tsv'), 'w')
drums_f = open(os.path.join(ROOT_MANIFEST, f'drums.tsv'), 'w')
bass_f = open(os.path.join(ROOT_MANIFEST, f'bass.tsv'), 'w')
other_f = open(os.path.join(ROOT_MANIFEST, f'other.tsv'), 'w')
mixture_f = open(os.path.join(ROOT_MANIFEST, f'mixture.tsv'), 'w')

if SPLIT == 'test':
    stem_dict = {'mixture': mixture_f}
else:
    stem_dict = {'vocals': vocals_f, 'drums': drums_f, 'bass': bass_f, 'other': other_f, 'mixture': mixture_f}

for f in stem_dict.values():
    print(ROOT, file=f)

for root, dirs, files in os.walk(ROOT):
    for dir in dirs:
        for stem in stem_dict.keys():
            fname = f'/{dir}/{stem}.wav'
            frames = soundfile.info(os.path.join(root, dir, f'{stem}.wav')).frames
            print(f'/{dir}/{stem}.wav\t{frames}', file=stem_dict[stem])