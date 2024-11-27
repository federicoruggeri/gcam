import logging
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification.f_beta import F1Score
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
import ast


class PromptDataset(Dataset):

    def __init__(
            self,
            prompts
    ):
        self.prompts = prompts

    def __getitem__(
            self,
            item
    ):
        return self.prompts[item]

    def __len__(self):
        return len(self.prompts)


def process_response(response):
    return response[response.find('ANSWER:') + len('ANSWER:'):].strip()


def convert_response(response, num_labels):
    labels = []
    guidelines = re.findall(r'\bG\d+', response)
    label_array = np.zeros((num_labels,))

    if not len(guidelines):
        return label_array

    for item in guidelines:
        try:
            parsed_item = int(item.split('G')[1])
            if parsed_item > 0 and parsed_item - 1 < num_labels:
                labels.append(parsed_item - 1)
        except ValueError:
            continue

    if not len(labels):
        return label_array

    labels = np.array(sorted(labels))
    label_array[labels] = 1

    return label_array


def parse_batch(batch, tokenizer):
    return tokenizer.batch_encode_plus(batch, return_tensors='pt', padding=True, truncation=True)


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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument('--category', '-c', default='A', type=str, help='A | CH | CR | LTD | TER')
    parser.add_argument('--model',
                        '-m',
                        default='mistral',
                        type=str,
                        nargs='?',
                        choices=['mistral', 'llama', 'phi'])
    parser.add_argument('--force', '-f', default=False, type=bool)
    args = parser.parse_args()
    category = args.category
    model_name = args.model

    batch_size = 1

    save_path = Path(__file__).parent.parent.resolve().joinpath('results', 'tos', f'{model_name}_{category}')
    if not save_path.exists():
        save_path.mkdir(parents=True)

    data_dir = Path(__file__).parent.parent.resolve().joinpath('data', 'tos')
    data = pd.read_csv(data_dir.joinpath('dataset_100.csv'))
    documents = data['document_ID'].unique()
    test_documents = documents[90:]
    test_data = data[data['document_ID'].isin(test_documents)]

    guidelines_path = data_dir.joinpath(f'{category}_KB.txt')
    guidelines = []
    with open(guidelines_path, 'r') as f:
        for line in f:
            guidelines.append(line)

    y_true_classes, y_true_guidelines = get_true_labels(df=test_data, guidelines=guidelines, category=category)
    num_labels = len(guidelines)

    metrics_path = save_path.joinpath('metrics_guidelines.npy')

    if not metrics_path.exists() or args.force:

        model_map = {
            'mistral': "mistralai/Mistral-7B-Instruct-v0.3",
            'llama': "meta-llama/Meta-Llama-3.1-8B-Instruct",
            'phi': "microsoft/Phi-3-mini-4k-instruct"
        }

        model_id = model_map[model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=th.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            return_dict=True,
            quantization_config=bnb_config,
            device_map='auto'
        )

        generation_config = model.generation_config
        generation_config.max_new_tokens = 100
        generation_config.eos_token_id = tokenizer.eos_token_id
        generation_config.pad_token_id = tokenizer.eos_token_id
        generation_config.temperature = None
        generation_config.num_return_sequences = 1

        guidelines_str = '\n'.join(
            [f'ID: G{idx + 1}  DEF: {definition}' for idx, definition in enumerate(guidelines)])
        guideline_indexes = [f'G{idx + 1}' for idx in range(len(guidelines))]

        prompt = [
            {
                'role': 'system',
                'content': 'You are an annotator for unfair clause detection in the legal domain concerning consumer contracts.'
            },
            {
                'role': 'user',
                'content': """Your are given a GUIDELINE SET about consumer contract unfairness for the user or client.
                Each GUIDELINE SET item in the set is described by an identifier (ID) and a definition (DEF).
                Read carefully the GUIDELINE SET.
                Your task is to assess if each GUIDELINE SET item relates to the input text or not.
                Respond only by choosing from {guideline_indexes} or respond with NO if the GUIDELINE SET does not relate to the input text. 
                
                GUIDELINE: 
                {guideline}
                
                TEXT: 
                {text}
                
                ANSWER:
                """
            }
        ]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

        texts = test_data.text.values.tolist()
        prompts = [prompt.format(text=text, guideline_indexes=guideline_indexes, guideline=guidelines_str) for text in
                   texts]
        prompt_data = PromptDataset(prompts=prompts)
        prompt_loader = DataLoader(prompt_data,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   collate_fn=partial(parse_batch, tokenizer=tokenizer))

        raw_responses = []
        parsed_responses = []
        with th.inference_mode():
            for batch in tqdm(prompt_loader, desc="Generating responses"):
                response = model.generate(
                    input_ids=batch['input_ids'].to(model.device),
                    attention_mask=batch['attention_mask'].to(model.device),
                    generation_config=generation_config,
                    do_sample=False,
                    use_cache=True
                )
                raw_response = tokenizer.batch_decode(response, skip_special_tokens=True)
                raw_response = [process_response(item) for item in raw_response]
                raw_responses.extend(raw_response)
                parsed_response = [convert_response(item, num_labels=num_labels) for item in raw_response]
                parsed_responses.extend(parsed_response)

        parsed_responses = np.array(parsed_responses)
        test_data['raw_response'] = raw_responses
        test_data['parsed_response'] = parsed_responses.tolist()
        test_data.to_csv(save_path.joinpath('response_guidelines.csv'), index=False)
    else:
        test_data = pd.read_csv(save_path.joinpath('response_guidelines.csv'))
        parsed_responses = np.array([ast.literal_eval(item) for item in test_data['parsed_response'].values])

    # guidelines
    metric_guidelines = F1Score(task='multilabel', average='macro', num_labels=num_labels)
    f1_guidelines = metric_guidelines(th.tensor(parsed_responses), th.tensor(y_true_guidelines))
    print("F1 Guidelines: ", f1_guidelines)

    # classes
    y_pred_classes = np.max(parsed_responses, axis=-1)

    metric_classes = F1Score(task='multiclass', average='macro', num_classes=2)
    f1_classes = metric_classes(th.tensor(y_pred_classes), th.tensor(y_true_classes))
    print('F1 Classes: ', f1_classes)

    np.save(metrics_path.as_posix(), {
        'guidelines': f1_guidelines,
        'classes': f1_classes
    })
