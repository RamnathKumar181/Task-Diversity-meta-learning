import os
import torch

from torchmeta.utils.data import CombinationMetaDataset, ClassDataset
from collections import OrderedDict

from src.datasets.meta_dataset.utils import Split
from src.datasets.meta_dataset.loader import get_dataspecs
from src.datasets.meta_dataset.reader import Reader
from src.datasets.meta_dataset.pipeline import cycle_, parse_record
from src.datasets.meta_dataset.transform import get_transforms


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
            num_ways,
            num_shots,
            num_shots_test,
            meta_train=meta_train,
            meta_val=meta_val,
            meta_test=meta_test,
            meta_split=meta_split
        )
        super().__init__(
            dataset,
            num_ways,
            target_transform=None,
            dataset_transform=None
        )

    def __getitem__(self, index):
        support_images, query_images = [], []
        support_tasks, query_tasks = [], []
        targets = torch.randperm(self.num_ways).unsqueeze(1)

        for class_id in index:
            used_ids = set()
            images = []
            classes = []
            while len(images) < self.num_shots + self.num_shots_test:
                sample_dict = self.dataset._get_next(class_id)
                if sample_dict['id'] in used_ids:
                    continue
                used_ids.add(sample_dict['id'])

                sample_dict = parse_record(sample_dict)
                images.append(self.dataset.transform(sample_dict['image']))
                classes.append(class_id)

            support_images.extend(images[:self.num_shots])
            query_images.extend(images[self.num_shots:])
            support_tasks.extend(classes[:self.num_shots])
            query_tasks.extend(classes[self.num_shots:])

        support_images = torch.stack(support_images, dim=0)
        support_labels = targets.repeat((1, self.num_shots)).view(-1)

        query_images = torch.stack(query_images, dim=0)
        query_labels = targets.repeat((1, self.num_shots_test)).view(-1)

        return OrderedDict([
            ('train', (support_images, support_labels, support_tasks)),
            ('test', (query_images, query_labels, query_tasks))
        ])


class SingleMetaDatasetClassDataset(ClassDataset):
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
        self.root = os.path.expanduser(os.path.join(root, source))
        self.source = source
        super().__init__(
            meta_train=meta_train,
            meta_val=meta_val,
            meta_test=meta_test,
            meta_split=meta_split,
            class_augmentations=None
        )

        dataset_spec, data_config, _ = get_dataspecs(
            self.root, num_ways, num_shots, num_shots_test, source
        )
        if self.meta_train:
            split = Split.TRAIN
        elif self.meta_val:
            split = Split.VALID
        elif self.meta_test:
            split = Split.TEST
        else:
            raise ValueError('Unknown split')

        self.episode_reader = Reader(dataset_spec=dataset_spec,
                                     split=split,
                                     shuffle=data_config.shuffle,
                                     offset=0)
        self._class_datasets = self.episode_reader.construct_class_datasets()
        self.transform = get_transforms(data_config, split)

    def __getitem__(self, index):
        return self._class_datasets[index]

    def _get_next(self, index):
        try:
            sample_dict = next(self[index])
        except (StopIteration, TypeError):
            self._class_datasets[index] = cycle_(self[index])
            sample_dict = next(self[index])
        return sample_dict

    @property
    def num_classes(self):
        return self.episode_reader.num_classes
