import torch
import torch.nn.functional as F
import random
import numpy as np
import os

from collections import namedtuple, OrderedDict
from src.datasets import Omniglot, MiniImagenet
from torchmeta.toy import Sinusoid
from torchmeta.transforms import ClassSplitter, Categorical, Rotation
from torchvision.transforms import ToTensor, Resize, Compose
from torchvision import transforms

Benchmark = namedtuple('Benchmark', 'meta_train_dataset meta_val_dataset '
                       'meta_test_dataset model loss_function')


def seed_everything(seed=0):
    """Set random seed"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_accuracy(logits, targets):
    """Compute the accuracy"""
    with torch.no_grad():
        _, predictions = torch.max(logits, dim=1)
        accuracy = torch.mean(predictions.eq(targets).float())
    return accuracy.item()


def tensors_to_device(tensors, device=torch.device('cpu')):
    """Place a collection of tensors in a specific device"""
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, (list, tuple)):
        return type(tensors)(tensors_to_device(tensor, device=device)
                             for tensor in tensors)
    elif isinstance(tensors, (dict, OrderedDict)):
        return type(tensors)([(name, tensors_to_device(tensor, device=device))
                              for (name, tensor) in tensors.items()])
    else:
        raise NotImplementedError()


class ToTensor1D(object):
    """Convert a `numpy.ndarray` to tensor. Unlike `ToTensor` from torchvision,
    this converts numpy arrays regardless of the number of dimensions.
    Converts automatically the array to `float32`.
    """

    def __call__(self, array):
        return torch.from_numpy(array.astype('float32'))

    def __repr__(self):
        return self.__class__.__name__ + '()'


def get_benchmark_by_name(model_name,
                          name,
                          folder,
                          num_ways,
                          num_shots,
                          num_shots_test,
                          hidden_size=None,
                          test_dataset=None,
                          metaoptnet_embedding='ResNet',
                          metaoptnet_head='SVM-CS',
                          use_random_crop=False,
                          use_color_jitter=False):
    """Get dataset, model and loss function"""
    from src.maml.model import ModelConvOmniglot, ModelConvMiniImagenet, ModelMLPSinusoid
    from src.reptile.model import ModelConvOmniglot as ModelConvOmniglotReptile
    from src.reptile.model import ModelConvMiniImagenet as ModelConvMiniImagenetReptile
    from src.protonet.model import Protonet_Omniglot, Protonet_MiniImagenet
    from src.protonet.metalearners.loss import prototypical_loss
    from src.matching_networks.model import MatchingNetwork
    from src.cnaps.model import Cnaps
    from src.cnaps.metalearners.loss import CNAPsLoss
    from src.metaoptnet.model import MetaOptNet
    dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=num_shots,
                                      num_test_per_class=num_shots_test)
    if test_dataset is not None:
        folder = test_dataset

    if name == 'sinusoid':
        transform = ToTensor1D()

        meta_train_dataset = Sinusoid(num_shots + num_shots_test,
                                      num_tasks=1000000,
                                      transform=transform,
                                      target_transform=transform,
                                      dataset_transform=dataset_transform)
        meta_val_dataset = Sinusoid(num_shots + num_shots_test,
                                    num_tasks=1000000,
                                    transform=transform,
                                    target_transform=transform,
                                    dataset_transform=dataset_transform)
        meta_test_dataset = Sinusoid(num_shots + num_shots_test,
                                     num_tasks=1000000,
                                     transform=transform,
                                     target_transform=transform,
                                     dataset_transform=dataset_transform)

        if model_name in ['maml', 'reptile']:
            model = ModelMLPSinusoid(hidden_sizes=[40, 40])
            loss_function = F.mse_loss
        if model_name in ['protonet', 'matching_networks', 'cnaps', 'metaoptnet']:
            raise NotImplementedError(f"Not implemented for {model_name} on sinusoid dataset")

    elif name == 'omniglot':
        class_augmentations = [Rotation([90, 180, 270])]
        transform = []
        image_size = 28
        if model_name == 'cnaps':
            image_size = 84
        if use_random_crop:
            transform.append(transforms.RandomResizedCrop(image_size))
        if use_color_jitter:
            transform.append(transforms.ColorJitter(brightness=0.5,
                                                    contrast=0.5,
                                                    saturation=0.3))
        transform.append(Resize(image_size))
        transform.append(ToTensor())
        transform = Compose(transform)
        try:
            meta_train_dataset = Omniglot(folder,
                                          transform=transform,
                                          target_transform=Categorical(num_ways),
                                          num_classes_per_task=num_ways,
                                          meta_train=True,
                                          class_augmentations=class_augmentations,
                                          dataset_transform=dataset_transform,
                                          download=False)
        except Exception:
            meta_train_dataset = Omniglot(folder,
                                          transform=transform,
                                          target_transform=Categorical(num_ways),
                                          num_classes_per_task=num_ways,
                                          meta_train=True,
                                          class_augmentations=class_augmentations,
                                          dataset_transform=dataset_transform,
                                          download=True)
        meta_val_dataset = Omniglot(folder,
                                    transform=transform,
                                    target_transform=Categorical(num_ways),
                                    num_classes_per_task=num_ways,
                                    meta_val=True,
                                    class_augmentations=class_augmentations,
                                    dataset_transform=dataset_transform)
        meta_test_dataset = Omniglot(folder,
                                     transform=transform,
                                     target_transform=Categorical(num_ways),
                                     num_classes_per_task=num_ways,
                                     meta_test=True,
                                     dataset_transform=dataset_transform)

        if model_name == 'maml':
            model = ModelConvOmniglot(num_ways, hidden_size=hidden_size)
            loss_function = F.cross_entropy
        elif model_name == 'reptile':
            model = ModelConvOmniglotReptile(num_ways, hidden_size=hidden_size)
            loss_function = F.cross_entropy
        elif model_name == 'protonet':
            model = Protonet_Omniglot()
            loss_function = prototypical_loss
        elif model_name == 'matching_networks':
            model = MatchingNetwork(keep_prob=0, batch_size=32, num_channels=1, fce=False, num_classes_per_set=num_ways,
                                    num_samples_per_class=num_shots, image_size=28)
            loss_function = torch.nn.NLLLoss
        elif model_name == 'cnaps':
            model = Cnaps()
            loss_function = CNAPsLoss
        elif model_name == 'metaoptnet':
            model = MetaOptNet(name, metaoptnet_embedding, metaoptnet_head,
                               num_ways, num_shots, num_shots_test)
            loss_function = torch.nn.NLLLoss

    elif name == 'miniimagenet':
        transform = []
        if use_random_crop:
            transform.append(transforms.RandomResizedCrop(84))
        if use_color_jitter:
            transform.append(transforms.ColorJitter(brightness=0.5,
                                                    contrast=0.5,
                                                    saturation=0.3))
        transform.append(Resize(84))
        transform.append(ToTensor())
        transform = Compose(transform)
        try:
            meta_train_dataset = MiniImagenet(folder,
                                              transform=transform,
                                              target_transform=Categorical(num_ways),
                                              num_classes_per_task=num_ways,
                                              meta_train=True,
                                              dataset_transform=dataset_transform,
                                              download=False)
        except Exception:
            meta_train_dataset = MiniImagenet(folder,
                                              transform=transform,
                                              target_transform=Categorical(num_ways),
                                              num_classes_per_task=num_ways,
                                              meta_train=True,
                                              dataset_transform=dataset_transform,
                                              download=True)
        meta_val_dataset = MiniImagenet(folder,
                                        transform=transform,
                                        target_transform=Categorical(num_ways),
                                        num_classes_per_task=num_ways,
                                        meta_val=True,
                                        dataset_transform=dataset_transform)
        meta_test_dataset = MiniImagenet(folder,
                                         transform=transform,
                                         target_transform=Categorical(num_ways),
                                         num_classes_per_task=num_ways,
                                         meta_test=True,
                                         dataset_transform=dataset_transform)

        if model_name == 'maml':
            model = ModelConvMiniImagenet(num_ways, hidden_size=hidden_size)
            loss_function = F.cross_entropy
        elif model_name == 'reptile':
            model = ModelConvMiniImagenetReptile(num_ways, hidden_size=hidden_size)
            loss_function = F.cross_entropy
        elif model_name == 'protonet':
            model = Protonet_MiniImagenet()
            loss_function = prototypical_loss
        elif model_name == 'matching_networks':
            model = MatchingNetwork(keep_prob=0, batch_size=32, num_channels=3, fce=False, num_classes_per_set=num_ways,
                                    num_samples_per_class=num_shots, image_size=84)
            loss_function = torch.nn.NLLLoss
        elif model_name == 'cnaps':
            model = Cnaps()
            loss_function = CNAPsLoss
        elif model_name == 'metaoptnet':
            model = MetaOptNet(name, metaoptnet_embedding, metaoptnet_head,
                               num_ways, num_shots, num_shots_test)
            loss_function = torch.nn.NLLLoss

    else:
        raise NotImplementedError('Unknown dataset `{0}`.'.format(name))

    return Benchmark(meta_train_dataset=meta_train_dataset,
                     meta_val_dataset=meta_val_dataset,
                     meta_test_dataset=meta_test_dataset,
                     model=model,
                     loss_function=loss_function)
