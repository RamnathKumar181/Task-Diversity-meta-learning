import os
import torch
import tensorflow.compat.v1 as tf

from torchmeta.utils.data import CombinationMetaDataset, ClassDataset
from collections import OrderedDict
from itertools import accumulate
from bisect import bisect_right

from src.datasets.meta_dataset.reader import Reader
from src.datasets.meta_dataset.dataset_spec import load_dataset_spec
from src.datasets.meta_dataset.learning_spec import Split
from src.datasets.meta_dataset.decoder import ImageDecoder


SOURCES = {
    'train': [
        'ilsvrc_2012',
        'omniglot',
        'aircraft',
        'cu_birds',
        'dtd',
        'quickdraw',
        'fungi',
        'vgg_flower'
    ],
    'val': [
        'ilsvrc_2012',
        'omniglot',
        'aircraft',
        'cu_birds',
        'dtd',
        'quickdraw',
        'fungi',
        'vgg_flower',
        'mscoco'
    ],
    'test': []  # TODO
}


class MetaDataset(CombinationMetaDataset):
    def __init__(
        self,
        root,
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

        dataset = MetaDatasetClassDataset(
            root,
            meta_train=meta_train,
            meta_val=meta_val,
            meta_test=meta_test,
            meta_split=meta_split,
            shuffle_buffer_size=1000,
        )
        super().__init__(
            dataset,
            num_ways,
            target_transform=None,
            dataset_transform=None
        )

    def __getitem__(self, index):
        support_images, query_images = [], []
        targets = torch.randperm(self.num_ways).unsqueeze(1)
        tasks = torch.tensor(index).unsqueeze(1)

        for class_id in index:
            images = []
            while len(images) < self.num_shots + self.num_shots_test:
                image = self.dataset._get_next(class_id)
                images.append(torch.from_numpy(image))

            support_images.extend(images[:self.num_shots])
            query_images.extend(images[self.num_shots:])

        support_images = torch.stack(support_images, dim=0)
        support_labels = targets.repeat((1, self.num_shots)).view(-1)
        support_tasks = tasks.repeat((1, self.num_shots)).view(-1)

        query_images = torch.stack(query_images, dim=0)
        query_labels = targets.repeat((1, self.num_shots_test)).view(-1)
        query_tasks = tasks.repeat((1, self.num_shots_test)).view(-1)

        return OrderedDict([
            ('train', (support_images, support_labels, support_tasks)),
            ('test', (query_images, query_labels, query_tasks))
        ])


class MetaDatasetClassDataset(ClassDataset):
    def __init__(
        self,
        root,
        meta_train=False,
        meta_val=False,
        meta_test=False,
        meta_split=None,
        shuffle_buffer_size=None
    ):
        self.root = os.path.expanduser(root)
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
        self.sources = SOURCES[self.meta_split]

        image_decoder = ImageDecoder(image_size=84, data_augmentation=None)
        def image_decode(example_string, source_id):
            image = image_decoder(example_string)
            return tf.transpose(image, (2, 0, 1))

        self._class_datasets = []
        for name in self.sources:
            dataset_records_path = os.path.join(self.root, name)
            dataset_spec = load_dataset_spec(dataset_records_path)

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

            self._class_datasets.append([dataset.as_numpy_iterator()
                for dataset in class_datasets])

        self._cum_num_classes = list(accumulate(map(len, self._class_datasets)))
        self._cum_num_classes.insert(0, 0)

    def __getitem__(self, index):
        source = bisect_right(self._cum_num_classes, index) - 1
        index -= self._cum_num_classes[source]
        return self._class_datasets[source][index]

    def _get_next(self, index):
        return next(self[index])

    @property
    def num_classes(self):
        return self._cum_num_classes[-1]
