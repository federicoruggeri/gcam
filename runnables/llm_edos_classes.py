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


def convert_response(response):
    return 1 if 'yes' in response.casefold() else 0


def parse_batch(batch, tokenizer):
    return tokenizer.batch_encode_plus(batch, return_tensors='pt', padding=True, truncation=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument('--model',
                        '-m',
                        default='mistral',
                        type=str,
                        nargs='?',
                        choices=['mistral', 'llama', 'phi'])
    args = parser.parse_args()
    model_name = args.model

    batch_size = 1

    save_path = Path(__file__).parent.parent.resolve().joinpath('results', 'edos', model_name)
    if not save_path.exists():
        save_path.mkdir(parents=True)

    data_dir = Path(__file__).parent.parent.resolve().joinpath('data')
    test_data = pd.read_csv(data_dir.joinpath('edos', 'test.csv'))
    guidelines = pd.read_csv(data_dir.joinpath('edos', 'guidelines.csv'))
    y_true = [1 if label == 'sexist' else 0 for label in test_data.label_sexist.values]

    model_map = {
        'mistral': "mistralai/Mistral-7B-Instruct-v0.3",
        'llama': "meta-llama/Meta-Llama-3.1-8B-Instruct",
        'phi': "microsoft/Phi-3-mini-4k-instruct"
    }

    model_id = model_map[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

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
        [f'ID: G{idx + 1}  DEF: {definition}' for idx, definition in enumerate(guidelines.Definition.values)])

    prompt = [
        {
            'role': 'system',
            'content': 'You are an annotator for sexism detection.'
        },
        {
            'role': 'user',
            'content': """You are given a GUIDELINE SET about sexism detection.
            Each GUIDELINE SET item in the set is described by an identifier (ID) and a definition (DEF).
            Read carefully the GUIDELINE SET.
            Your task is to classify input TEXT as containing sexism or not.
            Respond only YES or NO.
            
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
    prompts = [prompt.format(text=text, guideline=guidelines_str) for text in texts]
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
            parsed_response = [convert_response(item) for item in raw_response]
            parsed_responses.extend(parsed_response)

    test_data['raw_response'] = raw_responses
    test_data['parsed_response'] = parsed_responses
    test_data.to_csv(save_path.joinpath('response_classes.csv'), index=False)

    metric = F1Score(task='multiclass', average='macro', num_classes=2)
    f1_score = metric(th.tensor(parsed_responses), th.tensor(y_true))
    print('F1: ', f1_score)
    np.save(save_path.joinpath('metrics_classes.npy').as_posix(), f1_score)
