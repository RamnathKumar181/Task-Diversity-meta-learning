import os
import torch
from torchmeta.utils.data import (CombinationMetaDataset, ClassDataset)
from collections import OrderedDict
from src.datasets.meta_dataset_reader import MetaDatasetReader
import numpy as np

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
    'test': [
        'ilsvrc_2012',
        'omniglot',
        'aircraft',
        'cu_birds',
        'dtd',
        'quickdraw',
        'fungi',
        'vgg_flower',
        'traffic_sign',
        'mscoco'
    ]
}


class SingleMetaDataset(CombinationMetaDataset):
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

        super().__init__(
            dataset,
            num_ways,
            target_transform=None,
            dataset_transform=None
        )

    def __getitem__(self, index):
        return self.dataset[index]


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
        self.root = os.path.expanduser(root)
        self.source = source

        super().__init__(
            meta_train=meta_train,
            meta_val=meta_val,
            meta_test=meta_test,
            meta_split=meta_split,
            class_augmentations=None
        )
        self.meta_dataset = MetaDatasetReader(
            data_path=root,
            train_set=SOURCES["train"],
            validation_set=SOURCES["val"],
            test_set=SOURCES["test"],
            max_way_train=5,
            max_way_test=5,
            max_support_train=1,
            max_support_test=15
        )

    def __getitem__(self, index):
        if self.meta_train:
            task_dict = self.meta_dataset.get_train_task()
        elif self.meta_val:
            task_dict = self.meta_dataset.get_validation_task(self.source)
        else:
            task_dict = self.meta_dataset.get_test_task(self.source)

        support_images_np, support_labels_np, support_tasks_np = task_dict[
            "context_images"], np.int64(task_dict["context_labels"]), task_dict["context_tasks"]
        query_images_np, query_labels_np, query_tasks_np = task_dict[
            "target_images"], np.int64(task_dict["target_labels"]), task_dict["target_tasks"]

        context_images_np = support_images_np.transpose([0, 3, 1, 2])
        context_images_np, context_labels_np, context_tasks_np = self.shuffle(
            context_images_np, support_labels_np, support_tasks_np)
        support_images = torch.from_numpy(context_images_np)
        support_labels = torch.from_numpy(context_labels_np)
        support_tasks = torch.from_numpy(context_tasks_np)

        target_images_np = query_images_np.transpose([0, 3, 1, 2])
        target_images_np, target_labels_np, target_tasks_np = self.shuffle(
            target_images_np, query_labels_np, query_tasks_np)
        query_images = torch.from_numpy(target_images_np)
        query_labels = torch.from_numpy(target_labels_np)
        query_tasks = torch.from_numpy(target_tasks_np)

        return OrderedDict([
            ('train', (support_images, support_labels, support_tasks)),
            ('test', (query_images, query_labels, query_tasks))
        ])

    def shuffle(self, images, labels, tasks):
        """
        Return shuffled data.
        """
        permutation = np.random.permutation(images.shape[0])
        return images[permutation], labels[permutation], tasks[permutation]

    @property
    def num_classes(self):
        return 4096  # We need to calculate this correctly as well. 4096 is a placeholder for now.
