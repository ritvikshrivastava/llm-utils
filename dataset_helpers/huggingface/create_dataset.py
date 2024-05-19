from datasets import load_dataset, load_from_disk
from datasets import Dataset, DatasetDict, IterableDataset


def load_type_dataset(path_to_file: str = 'path/to/local/my_dataset.json',
                      data_file_type: str = "json") -> Dataset or IterableDataset:
    """
    Load a file type dataset from local data files
    :param path_to_file:
    :param data_file_type: str = [json, csv]
    :return: Dataset or IterableDataset


    Eg. train_dataset = load_dataset(
        "json",
        data_files=os.path.join(script_args.dataset_path, "train_dataset.json"),
        split="train",
    )
    """
    return load_dataset(data_file_type, data_files=path_to_file)


def load_dataset_using_script(path_to_script: str = 'path/to/local/loading_script/loading_script.py',
                              split: str = "train") -> Dataset or DatasetDict:
    """
    Dataset scripts are small python scripts that define dataset builders.
    They define the citation, info and format of the dataset, contain the path or URL to the original data files
    and the code to load examples from the original data files.

    :param path_to_script: str
    :param split: str
    :return: Dataset or DatasetDict
    """
    return load_dataset(path_to_script, split=split)


def load_iterable_dataset(ds_name: str = "rotten_tomatoes", split: str = "train") -> IterableDataset:
    return load_dataset(ds_name, split=split, streaming=True)


def load_dataset_from_disk(disk_path: str = 'path/to/dataset/directory') -> Dataset or DatasetDict:
    """
    Loads a dataset that was previously saved using save_to_disk() from a dataset directory,
    or from a filesystem using any implementation of fsspec.spec.AbstractFileSystem.

    :param disk_path:
    :return: Dataset or DatasetDict
    """
    return load_from_disk(disk_path)
