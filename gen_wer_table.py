import csv
import json
import os
import pandas as pd

from jiwer import wer

with open('results.csv', 'w', newline='') as results_csv:
    fieldnames = ['sig', 'args', 'wer', 'wer_per_talk']
    writer = csv.DictWriter(results_csv, fieldnames=fieldnames)
    writer.writeheader()

    for root, dirs, files in os.walk('/fastdata/acq22mc/exp/outputs/xps/', topdown=False):
        for dir in dirs:

            try:
                with open(f'{root}/{dir}/.argv.json') as f:
                    args = json.load(f)
                results = {}
                results['sig'] = dir
                results['args'] = args

                output = pd.read_csv(f'{root}/{dir}/output.csv')

                refs = output['reference'].to_list()
                hyps = output['hypothesis'].to_list()

                results['wer'] = wer(refs, hyps)

                wer_per_talk = {}

                for talk in output['talk_id'].unique():
                    df = output[output['talk_id'] == talk]
                    refs = df['reference'].to_list()
                    hyps = df['hypothesis'].to_list()
                    wer_per_talk[talk] = wer(refs, hyps)

                results['wer_per_talk'] = wer_per_talk

                print(results)

                writer.writerow(results)
            except:
                print(f'skipping {dir}')
