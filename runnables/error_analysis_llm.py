import re
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd
import ast
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def get_guideline_idx(item, guidelines):
    return guidelines[guidelines.Label == float(item)].index.values[0]


def get_true_labels(df, category, guidelines):
    y_true_classes = df[category].values

    targets = df[f'{category}_targets'].values
    targets = [[int(item) for item in t.replace('[', '').replace(']', '').split(',')] if t is not np.nan else [] for
               t in targets]
    y_true_guidelines = []
    for target_set in targets:
        target_mask = np.zeros((len(guidelines)))
        target_mask[target_set] = 1
        y_true_guidelines.append(target_mask.tolist())

    y_true_guidelines = np.array(y_true_guidelines)

    return y_true_classes, y_true_guidelines


def convert_response_gcam(response, num_labels):
    labels = []
    guidelines = re.findall(r'\bG\d+', response)
    label_array = np.zeros((num_labels,))
    hallucinations = []

    if 'no' in response.casefold():
        return label_array, 'no', hallucinations

    if not len(guidelines):
        return label_array, 'empty', hallucinations

    for item in guidelines:
        try:
            parsed_item = int(item.split('G')[1])
            if parsed_item > 0 and parsed_item - 1 < num_labels:
                labels.append(parsed_item - 1)
        except ValueError:
            hallucinations.append(item)
            continue

    if not len(labels):
        return label_array, 'invalid', hallucinations

    labels = np.array(sorted(labels))
    label_array[labels] = 1

    return label_array, 'yes', hallucinations


def get_sam_info(sam_info, model_name):
    raw_response = sam_info['raw_response'].values
    generation_info = defaultdict(int)
    other_generations = []
    for response in raw_response:
        if 'yes' in response.casefold():
            generation_info['yes'] += 1
        elif 'no' in response.casefold():
            generation_info['no'] += 1
        else:
            generation_info['other'] += 1
            other_generations.append(response)

    print('Model: ', model_name)
    for key, value in generation_info.items():
        print(key, ': ', value)

    print(Counter(other_generations).most_common(n=10))
    print('*' * 50)


def get_gcam_info(gcam_info, num_labels, model_name):
    raw_response = gcam_info['raw_response'].values
    generation_info = defaultdict(int)
    other_generations = []
    other_hallucinations = []
    for response in raw_response:
        converted_response, status, hallucinations = convert_response_gcam(response, num_labels=num_labels)
        generation_info[status] += 1
        if status == 'ok' and len(hallucinations):
            generation_info['ok w/ hallucinations'] += 1
        if status == 'empty':
            other_generations.append(response)
        if len(hallucinations):
            other_hallucinations.extend(hallucinations)

    print('Model: ', model_name)
    for key, value in generation_info.items():
        print(key, ': ', value)

    print(Counter(other_generations).most_common(n=10))
    print(Counter(other_hallucinations).most_common(n=10))
    print(Counter(raw_response))
    print('*' * 50)


def compute_cm(sam_info, gcam_info, category):
    y_true_classes = np.array(sam_info[category].values)
    sam_preds = np.array(sam_info['parsed_response'].values)
    gcam_preds = np.array([ast.literal_eval(item) for item in gcam_info['parsed_response'].values]).max(axis=-1)

    cm_sam = confusion_matrix(y_pred=sam_preds, y_true=y_true_classes)
    cm_gcam = confusion_matrix(y_pred=gcam_preds, y_true=y_true_classes)

    cmd_sam = ConfusionMatrixDisplay(cm_sam, display_labels=[0, 1])
    cmd_gcam = ConfusionMatrixDisplay(cm_gcam, display_labels=[0, 1])

    cmd_sam.plot()
    cmd_gcam.plot()
    plt.show()


if __name__ == '__main__':
    # A, CH, CR, LTD, TER
    category = 'TER'

    base_path = Path(__file__).parent.parent.joinpath('results', 'tos')
    data_dir = Path(__file__).parent.parent.resolve().joinpath('data', 'tos')

    mistral_path = base_path.joinpath(f'mistral_{category.upper()}')
    llama_path = base_path.joinpath(f'llama_{category.upper()}')
    phi_path = base_path.joinpath(f'phi_{category.upper()}')

    # SAM
    mistral_sam_info = pd.read_csv(mistral_path.joinpath('response_classes.csv'))
    llama_sam_info = pd.read_csv(llama_path.joinpath('response_classes.csv'))
    phi_sam_info = pd.read_csv(phi_path.joinpath('response_classes.csv'))

    # GCAM
    mistral_gcam_info = pd.read_csv(mistral_path.joinpath('response_guidelines.csv'))
    llama_gcam_info = pd.read_csv(llama_path.joinpath('response_guidelines.csv'))
    phi_gcam_info = pd.read_csv(phi_path.joinpath('response_guidelines.csv'))

    guidelines_path = data_dir.joinpath(f'{category}_KB.txt')
    guidelines = []
    with open(guidelines_path, 'r') as f:
        for line in f:
            guidelines.append(line)

    y_true_classes, y_true_guidelines = get_true_labels(df=mistral_gcam_info,
                                                        guidelines=guidelines,
                                                        category=category)

    get_sam_info(sam_info=mistral_sam_info, model_name='mistral')
    get_sam_info(sam_info=llama_sam_info, model_name='llama')
    get_sam_info(sam_info=phi_sam_info, model_name='phi')

    get_gcam_info(gcam_info=mistral_gcam_info, num_labels=len(guidelines), model_name='mistral')
    get_gcam_info(gcam_info=llama_gcam_info, num_labels=len(guidelines), model_name='llama')
    get_gcam_info(gcam_info=phi_gcam_info, num_labels=len(guidelines), model_name='phi')

    compute_cm(sam_info=mistral_sam_info, gcam_info=mistral_gcam_info, category=category)
    compute_cm(sam_info=llama_sam_info, gcam_info=llama_gcam_info, category=category)
    compute_cm(sam_info=phi_sam_info, gcam_info=phi_gcam_info, category=category)
