import os
import sys
import warnings
import torch
import random

from torchmeta.utils.data import CombinationMetaDataset
from itertools import combinations

from src.datasets.meta_dataset.utils import Split
from src.datasets.meta_dataset.loader import get_dataspecs
from src.datasets.meta_dataset.pipeline import EpisodicDataset, cycle_, parse_record
from src.datasets.meta_dataset.sampling import EpisodeDescriptionSampler
from src.datasets.meta_dataset.transform import get_transforms


def _text_to_split(split):
    if split == 'train':
        return Split.TRAIN
    elif split == 'val':
        return Split.VALID
    elif split == 'test':
        return Split.TEST
    else:
        raise ValueError(f'Unknown split: {split}')


class MetaDataset(CombinationMetaDataset):
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
        self.root = os.path.expanduser(root)
        self.source = source
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_shots_test = num_shots_test
        if meta_train + meta_val + meta_test == 0:
            if meta_split is None:
                raise ValueError('The meta-split is undefined. Use either the '
                    'argument `meta_train=True` (or `meta_val`/`meta_test`), or '
                    'the argument `meta_split="train"` (or "val"/"test").')
            elif meta_split not in ['train', 'val', 'test']:
                raise ValueError('Unknown meta-split name `{0}`. The meta-split '
                    'must be in [`train`, `val`, `test`].'.format(meta_split))
            meta_train = (meta_split == 'train')
            meta_val = (meta_split == 'val')
            meta_test = (meta_split == 'test')
        elif meta_train + meta_val + meta_test > 1:
            raise ValueError('Multiple arguments among `meta_train`, `meta_val` '
                'and `meta_test` are set to `True`. Exactly one must be set to '
                '`True`.')
        self.meta_train = meta_train
        self.meta_val = meta_val
        self.meta_test = meta_test
        self._meta_split = meta_split

        dataset_spec, data_config, _ = get_dataspecs(
            self.root, num_ways, num_shots, num_shots_test, source
        )
        split = _text_to_split(self.meta_split)

        self.episode_reader = reader.Reader(dataset_spec=dataset_spec,
                                            split=split,
                                            shuffle=data_config.shuffle,
                                            offset=0)
        self._class_datasets = self.episode_reader.construct_class_datasets()
        self.transform = get_transforms(data_config, split)

    def __iter__(self):
        for index in combinations(self.num_classes, self.num_ways):
            yield self[index]

    def __getitem__(self, index):
        support_images, query_images = [], []
        targets = torch.randperm(self.num_ways).unsqueeze(1)

        for class_id in index:
            used_ids = set()
            images = []

            while len(images) < self.num_shots + self.num_shots_test:
                sample_dict = self._get_next(class_id)
                if sample_dict['id'] in used_ids:
                    continue
                used_ids.add(sample_dict['id'])

                sample_dict = parse_record(sample_dict)
                images.append(self.transform(sample_dict['image']))

            support_images.extend(images[:self.num_shots])
            query_images.extand(images[self.num_shots:])

        support_images = torch.stack(support_images, dim=0)
        support_labels = targets.repeat((1, self.num_shots)).view(-1)

        query_images = torch.stack(query_images, dim=0)
        query_labels = targets.repeat((1, self.num_shots_test)).view(-1)

        return ({
            'train': (support_images, support_labels),
            'test': (query_images, query_labels)
        })

    def _get_next(self, class_id):
        try:
            sample_dict = next(self._class_datasets[class_id])
        except (StopIteration, TypeError):
            self._class_datasets[class_id] = cycle_(self._class_datasets[class_id])
            sample_dict = next(self._class_datasets[class_id])
        return sample_dict

    def sample_task(self):
        index = random.sample(range(self.num_classes), self.num_ways)
        return self[index]

    @property
    def num_classes(self):
        return self.episode_reader.num_classes

    def __len__(self):
        length = 1
        for i in range(1, self.num_ways + 1):
            length *= (self.num_classes - i + 1) / i

        if length > sys.maxsize:
            warnings.warn('The number of possible tasks in {0} is '
                'combinatorially large (equal to C({1}, {2})), and exceeds '
                'machine precision. Setting the length of the dataset to the '
                'maximum integer value, which undervalues the actual number of '
                'possible tasks in the dataset. Therefore the value returned by '
                '`len(dataset)` should not be trusted as being representative '
                'of the true number of tasks.'.format(self, self.num_classes,
                self.num_ways), UserWarning, stacklevel=2)
            length = sys.maxsize
        return int(length)
