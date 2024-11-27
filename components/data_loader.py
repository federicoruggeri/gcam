from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from datasets import Dataset
from django.utils.functional import cached_property
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer


class EDOSLoader:
    """
    Data loader for EDOS dataset.
    """

    def __init__(
            self,
            pretrained_model_name: str,
            data_dir: Path,
            tokenization_args={},
            batch_size: int = 32
    ):
        self.df_dir = data_dir.joinpath('edos')

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name,
                                                       use_fast=False)
        self.tokenization_args = tokenization_args
        self.batch_size = batch_size

        self.num_classes = None
        self.num_kb_labels = None
        self.kb = None
        self.class_weights = None
        self.kb_class_weights = None

    def load_kb(
            self,
            filepath: Path
    ):
        kb = pd.read_csv(filepath).astype(str)
        return kb

    def get_guideline_idx(
            self,
            sample_label
    ):
        guideline_idx = self.kb[self.kb.Label == sample_label].index.values[0]
        return guideline_idx

    def parse_df(
            self,
            df
    ):
        df['label'] = [1 if label == 'sexist' else 0 for label in df['label_sexist'].values]

        targets = [self.get_guideline_idx(item.split(' ')[0]) if item != 'none' else None for item in
                   df['label_vector']]
        guideline_targets = []
        for target_set in targets:
            target_mask = np.zeros((len(self.kb)))
            if target_set is not None:
                target_mask[target_set] = 1
            guideline_targets.append(target_mask.tolist())
        guideline_targets = np.array(guideline_targets)
        df['kb_label'] = guideline_targets.tolist()

        return df

    def tokenize(
            self,
            inputs
    ):
        tok_info = self.tokenizer(inputs['text'], **self.tokenization_args)
        return tok_info

    @cached_property
    def data(
            self
    ):
        if not self.df_dir.exists():
            self.df_dir.mkdir(parents=True)

        kb_path = self.df_dir.joinpath('guidelines.csv')
        self.kb = self.load_kb(filepath=kb_path)
        self.num_kb_labels = len(self.kb)

        train_df = pd.read_csv(self.df_dir.joinpath('train.csv'))
        train_df = self.parse_df(df=train_df)
        self.num_classes = len(set(train_df.label.values))
        self.class_weights = compute_class_weight(y=train_df.label.values,
                                                  class_weight='balanced',
                                                  classes=[0, 1])
        self.kb_class_weights = compute_class_weight(y=np.array(train_df.kb_label.values.tolist()).ravel(),
                                                     class_weight='balanced',
                                                     classes=[0, 1])
        train_data = Dataset.from_pandas(train_df[['text', 'label', 'kb_label']])
        train_data = train_data.map(self.tokenize, batched=True, batch_size=self.batch_size)
        train_data = train_data.remove_columns(['text'])

        val_df = pd.read_csv(self.df_dir.joinpath('val.csv'))
        val_df = self.parse_df(df=val_df)
        val_data = Dataset.from_pandas(val_df[['text', 'label', 'kb_label']])
        val_data = val_data.map(self.tokenize, batched=True, batch_size=self.batch_size)
        val_data = val_data.remove_columns(['text'])

        test_df = pd.read_csv(self.df_dir.joinpath('test.csv'))
        test_df = self.parse_df(df=test_df)
        test_data = Dataset.from_pandas(test_df[['text', 'label', 'kb_label']])
        test_data = test_data.map(self.tokenize, batched=True, batch_size=self.batch_size)
        test_data = test_data.remove_columns(['text'])

        kb_info = self.tokenizer(self.kb.Definition.values.tolist(),
                                 **self.tokenization_args,
                                 return_tensors='pt',
                                 padding=True)

        self.kb = {
            'kb_input_ids': kb_info['input_ids'],
            'kb_attention_mask': kb_info['attention_mask']
        }

        return train_data, val_data, test_data

    def get_splits(
            self
    ):
        return self.data


class ToSLoader:
    """
    Data loader for ToS dataset.
    """

    def __init__(
            self,
            pretrained_model_name: str,
            data_dir: Path,
            category: str,
            tokenization_args={},
            batch_size: int = 32
    ):
        self.df_dir = data_dir.joinpath('tos')
        self.category = category

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name,
                                                       use_fast=False)
        self.tokenization_args = tokenization_args
        self.batch_size = batch_size

        self.num_classes = None
        self.num_kb_labels = None
        self.kb = None
        self.class_weights = None
        self.kb_class_weights = None

    def load_kb(
            self,
            filepath: Path
    ) -> List[str]:
        sentences = []

        with open(filepath, 'r') as f:
            for line in f:
                sentences.append(line)

        return sentences

    def parse_df(
            self,
            df
    ):
        df['label'] = df[self.category].values

        targets = df[f'{self.category}_targets'].values
        targets = [[int(item) for item in t.replace('[', '').replace(']', '').split(',')] if t is not np.nan else [] for
                   t in targets]
        guideline_targets = []
        for target_set in targets:
            target_mask = np.zeros((len(self.kb)))
            target_mask[target_set] = 1
            guideline_targets.append(target_mask.tolist())

        df['kb_label'] = np.array(guideline_targets).tolist()
        return df

    def tokenize(
            self,
            inputs
    ):
        tok_info = self.tokenizer(inputs['text'], **self.tokenization_args)
        return tok_info

    @cached_property
    def data(
            self
    ):
        if not self.df_dir.exists():
            self.df_dir.mkdir(parents=True)

        df_path = self.df_dir.joinpath('dataset_100.csv')

        kb_name = f'{self.category}_KB.txt'
        kb_path = df_path.with_name(kb_name)
        self.kb = self.load_kb(filepath=kb_path)
        self.num_kb_labels = len(self.kb)

        df = pd.read_csv(df_path)
        df = self.parse_df(df=df)
        self.num_classes = len(set(df.label.values))

        return df

    def get_splits(
            self
    ):
        df = self.data
        df = self.parse_df(df=df)

        documents = df['document_ID'].unique()
        train_documents = documents[:80]
        val_documents = documents[80:90]
        test_documents = documents[90:]

        train_df = df[df['document_ID'].isin(train_documents)]
        self.class_weights = compute_class_weight(y=train_df.label.values,
                                                  class_weight='balanced',
                                                  classes=[0, 1])
        self.kb_class_weights = compute_class_weight(y=np.array(train_df.kb_label.values.tolist()).ravel(),
                                                     class_weight='balanced',
                                                     classes=[0, 1])
        train_data = Dataset.from_dict({
            'text': train_df.text.values,
            'label': train_df.label.values,
            'kb_label': train_df['kb_label'].values
        })
        train_data = train_data.map(self.tokenize, batched=True, batch_size=self.batch_size)
        train_data = train_data.remove_columns(['text', 'token_type_ids'])

        val_df = df[df['document_ID'].isin(val_documents)]
        val_data = Dataset.from_dict({
            'text': val_df.text.values,
            'label': val_df.label.values,
            'kb_label': val_df.kb_label.values
        })
        val_data = val_data.map(self.tokenize, batched=True, batch_size=self.batch_size)
        val_data = val_data.remove_columns(['text', 'token_type_ids'])

        test_df = df[df['document_ID'].isin(test_documents)]
        test_data = Dataset.from_dict({
            'text': test_df.text.values,
            'label': test_df.label.values,
            'kb_label': test_df.kb_label.values
        })
        test_data = test_data.map(self.tokenize, batched=True, batch_size=self.batch_size)
        test_data = test_data.remove_columns(['text', 'token_type_ids'])

        kb_info = self.tokenizer(self.kb,
                                 **self.tokenization_args,
                                 return_tensors='pt',
                                 padding=True)
        self.kb = {
            'kb_input_ids': kb_info['input_ids'],
            'kb_attention_mask': kb_info['attention_mask']
        }

        return train_data, val_data, test_data
