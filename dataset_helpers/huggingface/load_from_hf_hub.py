from datasets import load_dataset, load_dataset_builder
from datasets import get_dataset_config_names
from datasets import Dataset, DatasetDict, IterableDataset


def inspect():
    ds_builder = load_dataset_builder("rotten_tomatoes")
    print(ds_builder.info.description)
    print(ds_builder.info.features)


def load_hf_hub_dataset_by_split(ds_name: str = "rotten_tomatoes", split: str = "train") -> Dataset:
    """
    Dataset({
        features: ['text', 'label'],
        num_rows: 8530
    })
    :param ds_name: str
    :param split: str
    :return: Dataset object
    """
    dataset = load_dataset(ds_name, split=split)
    return dataset


def load_hf_hub_dataset_all_splits(ds_name: str = "rotten_tomatoes") -> DatasetDict:
    """
    If you don’t specify a split, 🤗 Datasets returns a DatasetDict object instead:

    DatasetDict({
        train: Dataset({
            features: ['text', 'label'],
            num_rows: 8530
        })
        validation: Dataset({
            features: ['text', 'label'],
            num_rows: 1066
        })
        test: Dataset({
            features: ['text', 'label'],
            num_rows: 1066
        })
    })
    :param ds_name: str
    :return: DatasetDict object
    """
    return load_dataset(ds_name)


def load_hf_hub_dataset_configs(ds_name: str = "PolyAI/minds14") -> list[str]:
    """
    Use the get_dataset_config_names() function to retrieve a list of all the possible configurations available to your dataset:

    :param ds_name: str
    :return: configs: list[str]
    """

    configs = get_dataset_config_names(ds_name)
    """ 
    print(configs)
    ['cs-CZ', 'de-DE', 'en-AU', 'en-GB', 'en-US', 'es-ES', 'fr-FR', 'it-IT', 'ko-KR', 'nl-NL', 'pl-PL', 'pt-PT', 'ru-RU', 'zh-CN', 'all']
    
    return load_dataset("PolyAI/minds14", "fr-FR", split="train")
    """
    return configs


def load_iterable_dataset(ds_name: str = "rotten_tomatoes", split: str = "train") -> IterableDataset:
    return load_dataset(ds_name, split=split, streaming=True)