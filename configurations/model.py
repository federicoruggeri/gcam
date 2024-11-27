import torch as th

from configurations.base import BaseConfig, ConfigKey


class DiscreteConfig(BaseConfig):
    """
    SAM Binary model configuration (check modeling/discrete.py)
    """

    configs = {
        # EDOS
        ConfigKey(dataset='edos'): 'get_edos',

        # ToS
        ConfigKey(dataset='tos'): 'get_tos'
    }

    def __init__(
            self,
            hf_model_name,
            dropout_rate=0.0,
            tokenization_args={},
            **kwargs
    ):
        super().__init__(**kwargs)

        self.hf_model_name = hf_model_name
        self.dropout_rate = dropout_rate
        self.tokenization_args = tokenization_args

    @classmethod
    def get_edos(
            cls
    ):
        return cls(
            hf_model_name='cardiffnlp/twitter-roberta-base-hate',
            dropout_rate=0.20,
            optimizer_class=th.optim.Adam,
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

    @classmethod
    def get_tos(
            cls
    ):
        return cls(
            hf_model_name='nlpaueb/legal-bert-base-uncased',
            dropout_rate=0.20,
            optimizer_class=th.optim.Adam,
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
            batch_size=8,
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


class MultiLabelConfig(BaseConfig):
    """
    GCAM Fine-grained model configuration (check modeling/discrete.py)
    """

    configs = {
        # EDOS
        ConfigKey(dataset='edos'): 'get_edos',

        # ToS
        ConfigKey(dataset='tos'): 'get_tos'
    }

    def __init__(
            self,
            hf_model_name,
            dropout_rate=0.0,
            tokenization_args={},
            **kwargs
    ):
        super().__init__(**kwargs)

        self.hf_model_name = hf_model_name
        self.dropout_rate = dropout_rate
        self.tokenization_args = tokenization_args

    @classmethod
    def get_edos(
            cls
    ):
        return cls(
            hf_model_name='cardiffnlp/twitter-roberta-base-hate',
            dropout_rate=0.20,
            optimizer_class=th.optim.Adam,
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

    @classmethod
    def get_tos(
            cls
    ):
        return cls(
            hf_model_name='nlpaueb/legal-bert-base-uncased',
            dropout_rate=0.20,
            optimizer_class=th.optim.Adam,
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
            batch_size=8,
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


class GuidelineConfig(BaseConfig):
    """
    GCAM Text-based models configuration (check modeling/guidelines.py)
    """

    configs = {
        # EDOS
        ConfigKey(dataset='edos', tags={'dot'}): 'get_edos',
        ConfigKey(dataset='edos', tags={'concat'}): 'get_edos',
        ConfigKey(dataset='edos', tags={'entail'}): 'get_edos_entail',

        # ToS
        ConfigKey(dataset='tos', tags={'dot'}): 'get_tos',
        ConfigKey(dataset='tos', tags={'concat'}): 'get_tos',
        ConfigKey(dataset='tos', tags={'entail'}): 'get_tos_entail',
    }

    def __init__(
            self,
            hf_model_name,
            dropout_rate=0.0,
            tokenization_args={},
            **kwargs
    ):
        super().__init__(**kwargs)

        self.hf_model_name = hf_model_name
        self.dropout_rate = dropout_rate
        self.tokenization_args = tokenization_args

    @classmethod
    def get_edos(
            cls
    ):
        return cls(
            hf_model_name='cardiffnlp/twitter-roberta-base-hate',
            dropout_rate=0.20,
            optimizer_class=th.optim.Adam,
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
            batch_size=8,
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

    @classmethod
    def get_edos_entail(
            cls
    ):
        return cls(
            hf_model_name='cardiffnlp/twitter-roberta-base-hate',
            dropout_rate=0.20,
            optimizer_class=th.optim.Adam,
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
            batch_size=4,
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

    @classmethod
    def get_tos(
            cls
    ):
        return cls(
            hf_model_name='nlpaueb/legal-bert-base-uncased',
            dropout_rate=0.20,
            optimizer_class=th.optim.Adam,
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
            batch_size=8,
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

    @classmethod
    def get_tos_entail(
            cls
    ):
        return cls(
            hf_model_name='nlpaueb/legal-bert-base-uncased',
            dropout_rate=0.20,
            optimizer_class=th.optim.Adam,
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
            batch_size=1,
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