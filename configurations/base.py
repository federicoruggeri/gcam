from typing import TypeVar, Set, Union, Dict, Callable

import torch as th

Tag = Union[str, Set[str]]

C = TypeVar('C', bound='BaseConfig')


class ConfigKey:
    """
    Compound key used for configurations.
    """

    def __init__(
            self,
            dataset: str,
            tags: Tag = None,
    ):
        self.dataset = dataset
        self.tags = {tags} if type(tags) == str else tags

    def __hash__(
            self
    ) -> int:
        return hash(self.__str__())

    def __str__(
            self
    ) -> str:
        return f'{self.dataset}:{self.tags}'

    def __repr__(
            self
    ) -> str:
        return self.__str__()

    def __eq__(
            self,
            other
    ) -> bool:
        if other is None or not isinstance(other, ConfigKey):
            return False

        return (self.dataset == other.dataset and
                self.tags == other.tags)


class BaseConfig:
    """
    Base configuration for all models.
    """

    configs: Dict[ConfigKey, str] = {}

    def __init__(
            self,
            seeds,
            optimizer_class=th.optim.Adam,
            optimizer_kwargs=None,
            batch_size=8,
            freeze_embeddings: bool = False,
    ):
        self.seeds = seeds
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}
        self.batch_size = batch_size
        self.freeze_embeddings = freeze_embeddings

    @classmethod
    def from_config(
            cls,
            key: ConfigKey
    ) -> C:
        config_method = cls.configs[key]
        return getattr(cls, config_method)()

    @classmethod
    def add_config(
            cls,
            key: ConfigKey,
            constructor: Callable[[], C]
    ):
        setattr(cls, constructor.__name__, classmethod(constructor))
        cls.configs[key] = constructor.__name__
