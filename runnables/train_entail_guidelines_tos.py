import logging
from argparse import ArgumentParser
from pathlib import Path

import lightning as L
import numpy as np
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification.f_beta import F1Score

from components.data_loader import ToSLoader
from components.model import GuidelineModel
from components.processing import GuidelineCollator
from configurations.base import ConfigKey
from configurations.model import GuidelineConfig
from modeling.guidelines import EntailGuidelines

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Config
    # -------------
    parser = ArgumentParser()
    parser.add_argument('--category', '-c', default='TER', type=str, help='A | CH | CR | LTD | TER')
    args = parser.parse_args()
    category = args.category

    save_path = Path(__file__).parent.parent.resolve().joinpath('results', 'tos', f'entail-guidelines_{category}')
    if not save_path.exists():
        save_path.mkdir(parents=True)

    ckpt_path = save_path.joinpath('checkpoints')
    if not ckpt_path.exists():
        ckpt_path.mkdir(parents=True)

    data_dir = Path(__file__).parent.parent.resolve().joinpath('data')

    config = GuidelineConfig.from_config(key=ConfigKey(dataset='tos', tags={'entail'}))

    trainer_args = {
        'accelerator': 'auto',
        'devices': 1,
        'accumulate_grad_batches': 4,
        'max_epochs': 3,
    }

    # -------------
    seed_everything(seed=15000)

    loader = ToSLoader(data_dir=data_dir,
                       pretrained_model_name=config.hf_model_name,
                       tokenization_args=config.tokenization_args,
                       category=category)
    train_data, val_data, test_data = loader.get_splits()
    logging.info(f'KB size: {len(loader.kb["kb_input_ids"])}')

    collator = GuidelineCollator(padding_value=loader.tokenizer.pad_token_id,
                                 kb_input_ids=loader.kb['kb_input_ids'],
                                 kb_attention_mask=loader.kb['kb_attention_mask'])

    train_data = DataLoader(train_data,
                            shuffle=True,
                            batch_size=config.batch_size,
                            collate_fn=collator,
                            pin_memory=True)
    val_data = DataLoader(val_data,
                          batch_size=config.batch_size,
                          collate_fn=collator,
                          pin_memory=True)
    test_data = DataLoader(test_data,
                           batch_size=config.batch_size,
                           collate_fn=collator,
                           pin_memory=True)

    metrics = {}
    for seed in config.seeds:
        seed_everything(seed=seed)

        seed_ckpt_path = ckpt_path.joinpath(f'seed={seed}')
        seed_ckpt_path.mkdir(parents=True, exist_ok=True)

        model = EntailGuidelines(hf_model_name=config.hf_model_name,
                                 dropout_rate=config.dropout_rate,
                                 freeze_hf=config.freeze_embeddings)

        model = GuidelineModel(model=model,
                               num_classes=loader.num_kb_labels,
                               optimizer_class=config.optimizer_class,
                               optimizer_kwargs=config.optimizer_kwargs,
                               class_weights=loader.class_weights,
                               val_metrics=MetricCollection({
                                   'f1': F1Score(task='multiclass', average='macro', num_classes=loader.num_classes)}),
                               test_metrics=MetricCollection({
                                   'f1': F1Score(task='multiclass', average='macro', num_classes=loader.num_classes)}),
                               kb_val_metrics=MetricCollection({
                                   'f1': F1Score(task='multilabel', average='macro',
                                                 num_labels=loader.num_kb_labels)}),
                               kb_test_metrics=MetricCollection({
                                   'f1': F1Score(task='multilabel', average='macro', num_labels=loader.num_kb_labels)})
                               )

        should_train = not any(seed_ckpt_path.glob('*.ckpt'))

        trainer = L.Trainer(**trainer_args,
                            callbacks=[ModelCheckpoint(monitor='val_loss', mode='min', dirpath=seed_ckpt_path)],
                            deterministic=True,
                            precision='bf16-mixed'
                            )

        if should_train:
            trainer.fit(model,
                        train_dataloaders=train_data,
                        val_dataloaders=val_data)
        else:
            logging.info('Found existing checkpoint! Skipping model training...')

        ckpt_filepath = seed_ckpt_path.glob('*.ckpt').__next__()

        model = GuidelineModel.load_from_checkpoint(checkpoint_path=ckpt_filepath,
                                                    model=model.model,
                                                    val_metrics=MetricCollection({
                                                        'f1': F1Score(task='multiclass', average='macro',
                                                                      num_classes=loader.num_classes)}),
                                                    test_metrics=MetricCollection({
                                                        'f1': F1Score(task='multiclass', average='macro',
                                                                      num_classes=loader.num_classes)}),
                                                    kb_val_metrics=MetricCollection({
                                                        'f1': F1Score(task='multilabel', average='macro',
                                                                      num_labels=loader.num_kb_labels)}),
                                                    kb_test_metrics=MetricCollection({
                                                        'f1': F1Score(task='multilabel', average='macro',
                                                                      num_labels=loader.num_kb_labels)})
                                                    )
        seed_everything(seed=seed)

        # Metrics
        val_metrics = trainer.validate(model=model, dataloaders=val_data)[0]
        logging.info(f'Validation metrics: {val_metrics}')

        test_metrics = trainer.test(model=model, dataloaders=test_data)[0]
        logging.info(f'Test metrics: {test_metrics}')

        for metric_name, metric_value in val_metrics.items():
            metrics.setdefault('validation', {}).setdefault(metric_name, []).append(metric_value)
        for metric_name, metric_value in test_metrics.items():
            metrics.setdefault('test', {}).setdefault(metric_name, []).append(metric_value)

    # Averaging
    for split_name in ['validation', 'test']:
        metric_names = list(metrics[split_name].keys())
        for metric_name in metric_names:
            metric_values = np.array(metrics[split_name][metric_name]).reshape(len(config.seeds), -1)
            per_seed_avg = metric_values.mean(axis=-1)
            per_seed_std = metric_values.std(axis=-1)
            avg = per_seed_avg.mean(axis=-1)
            std = per_seed_avg.std(axis=-1)
            metrics[split_name][f'per_seed_avg_{metric_name}'] = (per_seed_avg, per_seed_std)
            metrics[split_name][f'avg_{metric_name}'] = (avg, std)

    logging.info(metrics)
    np.save(save_path.joinpath('metrics.npy').as_posix(), metrics)
