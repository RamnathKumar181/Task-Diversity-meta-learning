import os
import torch
import tensorflow.compat.v1 as tf

from torchmeta.utils.data import CombinationMetaDataset, ClassDataset
from collections import OrderedDict

from src.datasets.meta_dataset.reader import Reader
from src.datasets.meta_dataset.dataset_spec import load_dataset_spec
from src.datasets.meta_dataset.learning_spec import Split
from src.datasets.meta_dataset.decoder import ImageDecoder
from src.datasets.metadataset import SOURCES, MetaDataset


class SingleMetaDataset(MetaDataset):
    def __init__(
        self,
        root,
        source,
        num_ways,
        num_shots,
        num_shots_test,
        meta_train=False,
        meta_val=False,
        meta_test=False,
        meta_split=None
    ):
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_shots_test = num_shots_test

        dataset = SingleMetaDatasetClassDataset(
            root,
            source,
            meta_train=meta_train,
            meta_val=meta_val,
            meta_test=meta_test,
            meta_split=meta_split,
            shuffle_buffer_size=1000,
        )
        CombinationMetaDataset.__init__(
            self,
            dataset,
            num_ways,
            target_transform=None,
            dataset_transform=None
        )


class SingleMetaDatasetClassDataset(ClassDataset):
    def __init__(
        self,
        root,
        source,
        meta_train=False,
        meta_val=False,
        meta_test=False,
        meta_split=None,
        shuffle_buffer_size=None
    ):
        self.root = os.path.expanduser(os.path.join(root, source))
        self.source = source
        super().__init__(
            meta_train=meta_train,
            meta_val=meta_val,
            meta_test=meta_test,
            meta_split=meta_split,
            class_augmentations=None
        )
        if self.meta_train:
            split = Split.TRAIN
        elif self.meta_val:
            split = Split.VALID
        elif self.meta_test:
            split = Split.TEST
        else:
            raise ValueError('Unknown split')
        if source not in SOURCES[self.meta_split]:
            raise ValueError(f'The source `{source}` is not in the list of '
                f'sources for the `{self.meta_split}` split: '
                f'{SOURCES[self.meta_split]}')

        image_decoder = ImageDecoder(image_size=84, data_augmentation=None)
        def image_decode(example_string, source_id):
            image = image_decoder(example_string)
            return tf.transpose(image, (2, 0, 1))

        dataset_spec = load_dataset_spec(self.root)
        reader = Reader(
            dataset_spec,
            split=split,
            shuffle_buffer_size=shuffle_buffer_size,
            read_buffer_size_bytes=None,
            num_prefetch=0,
            num_to_take=-1,
            num_unique_descriptions=0
        )
        class_datasets = reader.construct_class_datasets()
        class_datasets = [dataset.map(image_decode) for dataset in class_datasets]
        self._class_datasets = [dataset.as_numpy_iterator()
                for dataset in class_datasets]

    def __getitem__(self, index):
        return self._class_datasets[index]

    def _get_next(self, index):
        return next(self[index])

    @property
    def num_classes(self):
        return len(self._class_datasets)
