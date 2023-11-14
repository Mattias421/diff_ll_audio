import yaml
import csv

def filter_dataset(dataset):
    seen_values = set()
    filtered_dataset = []

    for entry in dataset:
        reference = entry['reference']
        n_best_list = entry['n_best_list']

        # Filter n_best_list based on 'whisper_hypothesis'
        filtered_n_best_list = []
        for hyp in n_best_list:
            current_value = hyp['whisper_hypothesis']
            if current_value not in seen_values:
                seen_values.add(current_value)
                filtered_n_best_list.append(hyp)

        filtered_entry = {'audio_path': entry['audio_path'],
                          'reference': reference,
                          'n_best_list': filtered_n_best_list}

        filtered_dataset.append(filtered_entry)

    return filtered_dataset


n_best_list_file = '/fastdata/acq22mc/exp/diff_ll_audio/nbl_data_wspr_filtered.yaml'
with open(n_best_list_file, 'r') as file:
    n_best_list = yaml.safe_load(file)

output_csv_path = 'wspr_ll_flat.csv'

with open(output_csv_path, 'w', newline='') as csv_file:
    fieldnames = ['audio_path', 'reference', 'wer', 'whisper_hypothesis', 'whisper_sum_logprob']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for entry in n_best_list:
        audio_path = entry['audio_path']
        reference = entry['reference']

        for n_best_entry in entry['n_best_list']:
            wer_value = n_best_entry['wer']
            whisper_hypothesis = n_best_entry['whisper_hypothesis']
            whisper_sum_logprob = n_best_entry['whisper_sum_logprob']

            writer.writerow({
                'audio_path': audio_path,
                'reference': reference,
                'wer': wer_value,
                'whisper_hypothesis': whisper_hypothesis,
                'whisper_sum_logprob': whisper_sum_logprob
            })