# Guideline-Center Annotation Methodology (GCAM)
Official repository of *"Let Guidelines Guide You: A Prescriptive Guideline-Centered Data Annotation Methodology"*.

## Structure

The code is organized as follows:

- ``components``: contains all business logic, including data loaders (`data_loader.py`), lightning models (`model.py`), and text processors (`processing.py`).
- ``configurations``: the ``model.py`` script contains all encoder-based configurations for ToS and EDOS datasets.
- ``modeling``: contains pytorch implementation for SAM and GCAM models, and model-specific layers.
- ``runnables``: contains all executable scripts for reproducing our results.
- ``annotations``: contains annotation data regarding subjectivity detection case study.

## Preliminaries

Before running a script, make sure you address the following preliminaries:

* Install requirements: ``pip install -r requirements.txt``
* [ToS and EDOS] Check `data` folder and unzip compressed folders.
* Check model configurations in ``configurations/model.py``.
* [Docker] Check and build `Dockerfile` to run the code in a docker container

## Reproducing our experiments

All models are implemented and trained using Pytorch Lightning for reproducibility.
The ``runnables`` folder contains scripts for testing each model on each dataset.
In particular, the scripts can be organized as follows:

### ToS scripts

The following scripts reproduce our results on ToS dataset.

Each script accepts a category (`-c` or `--category`) argument among the following {'A', 'CH', 'CR', 'LTD', 'TER'}.

**Encoder-based**
For encoder-based models, the corresponding model configuration defined in ``configurations/model.py`` is loaded (see Configuration section for more details).

* ``train_discrete_tos.py``          &#8594; SAM Binary model
* ``train_multilabel_tos.py``        &#8594; GCAM Fine-grained model
* ``train_entail_guidelines_tos.py`` &#8594; GCAM Entail model 
* ``train_concat_guidelines_tos.py`` &#8594; GCAM Concat model
* ``train_dot_guidelines_tos.py``    &#8594; GCAM Dot model

For instance, to run SAM Binary on ToS A:
```commandline
python runnables/train_discrete_tos.py -c A
```

**LLMs**
Each script accepts a model (`-m` or `--model`) argument among the following {'llama', 'mistral', 'phi'}

* ``llm_tos_classes.py``       &#8594; SAM LLMs
* ``llm_tos_guidelines.py``    &#8594; GCAM LLMs
* ``llm_tos_classes_nog.py``   &#8594; SAM LLMs w/o guidelines (Table 5)

For instance, to run SAM Mistral on ToS CH:
```commandline
python runnables/llm_tos_classes.py -c CH -m mistral
```

### EDOS scripts

The following scripts reproduce our results on EDOS dataset.

**Encoder-based**
For encoder-based models, the corresponding model configuration defined in ``configurations/model.py`` is loaded (see Configuration section for more details).

* ``train_discrete_edos.py``          &#8594; SAM Binary model
* ``train_multilabel_edos.py``        &#8594; GCAM Fine-grained model
* ``train_entail_guidelines_edos.py`` &#8594; GCAM Entail model 
* ``train_concat_guidelines_edos.py`` &#8594; GCAM Concat model
* ``train_dot_guidelines_edos.py``    &#8594; GCAM Dot model

* For instance, to run SAM Binary on EDOS:
```commandline
python runnables/train_discrete_edos.py
```

**LLMs**
Each script accepts a model (`-m` or `--model`) argument among the following {'llama', 'mistral', 'phi'}

* ``llm_edos_classes.py``       &#8594; SAM LLMs
* ``llm_edos_guidelines.py``    &#8594; GCAM LLMs
* ``llm_edos_classes_nog.py``   &#8594; SAM LLMs w/o guidelines (Table 5)

For instance, to run SAM Mistral on EDOS:
```commandline
python runnables/llm_edos_classes.py -m mistral
```

## Results

Results are stored in the `results` folder.
In particular, each model run is structured as follows:

```
results
   |- tos
      |- entail-guidelines-A
         |- checkpoints
         |- metrics.npy
         |- *predictions* 
      |- ... 
   - edos
      |- ...
```

Where ``checkpoints`` contains each seed run model checkpoints for reproducibility, 
``metrics.npy`` stores all validation and test partitions metrics, and `*predictions*` is a placeholder denoting each seed run model predictions.

Example (SAM Binary on EDOS)
```
results
   |- edos
      |- discrete
         |- checkpoints
         |- metrics.npy
         |- predictions_seed=2023.csv
         |- predictions_seed=15451.csv
         |- predictions_seed=1337.csv
         |- predictions_seed=2001.csv
         |- predictions_seed=2080.csv
```

## Configurations

We organize model configurations in Python classes.
In this way, configurations can be easily extended and maintained.

A specific model configuration is implemented as a Python function, referenced via a compound key (`ConfigKey` in `configurations/base.py`).

Example (SAM Binary in EDOS)
```python
    @classmethod
    def get_edos(
            cls
    ):
        return cls(
            hf_model_name='cardiffnlp/twitter-roberta-base-hate',
            dropout_rate=0.20,
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs={
                "lr": 5e-06,
                "weight_decay": 1e-05
            },
            tokenization_args={
                'truncation': True,
                'add_special_tokens': True,
                'max_length': 140
            },
            freeze_embeddings=False,
            batch_size=32,
            seeds=[
                2023,
                15451,
                1337,
                2001,
                2080,
                42,
                666,
                2024,
                33,
                40000
            ]
        )
```

## Data Inspection

We provide scripts for analyzing annotator disagreement, and model error analysis

```commandline
python3 runnables/analyse_ann_disagreement.py      # reports disagreement stas provided in Section 5
python3 runnables/error_analysis_encoder.py        # confusion matrices reported in Figure 3 and info reported in Section 6.1
python3 runnables/error_analysis_llm.py            # confusion matrices like Figure 3 and info reported in Section 6.2
```

## Contact

TBA

## Cite

TBA

## Credits

TBA