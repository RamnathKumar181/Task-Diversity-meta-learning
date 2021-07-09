import torch
import torch.nn.functional as F
import random
import numpy as np
import os

from collections import namedtuple, OrderedDict
from torchmeta.datasets import Omniglot, MiniImagenet
from torchmeta.toy import Sinusoid
from torchmeta.transforms import ClassSplitter, Categorical, Rotation
from torchvision.transforms import ToTensor, Resize, Compose

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
                          hidden_size=None):
    """Get dataset, model and loss function"""
    from src.maml.model import ModelConvOmniglot, ModelConvMiniImagenet, ModelMLPSinusoid
    from src.protonet.model import Protonet_Omniglot, Protonet_MiniImagenet
    from src.protonet.metalearners.loss import prototypical_loss

    dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=num_shots,
                                      num_test_per_class=num_shots_test)
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
        if model_name == 'protonet':
            raise NotImplementedError("Not implemented for protonet on sinusoid dataset")

    elif name == 'omniglot':
        class_augmentations = [Rotation([90, 180, 270])]
        transform = Compose([Resize(28), ToTensor()])
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
                                          download=False)
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

        if model_name in ['maml', 'reptile']:
            model = ModelConvOmniglot(num_ways, hidden_size=hidden_size)
            loss_function = F.cross_entropy
        if model_name == 'protonet':
            model = Protonet_Omniglot()
            loss_function = prototypical_loss

    elif name == 'miniimagenet':
        transform = Compose([Resize(84), ToTensor()])
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

        if model_name in ['maml', 'reptile']:
            model = ModelConvMiniImagenet(num_ways, hidden_size=hidden_size)
            loss_function = F.cross_entropy
        if model_name == 'protonet':
            model = Protonet_MiniImagenet()
            loss_function = prototypical_loss

    else:
        raise NotImplementedError('Unknown dataset `{0}`.'.format(name))

    return Benchmark(meta_train_dataset=meta_train_dataset,
                     meta_val_dataset=meta_val_dataset,
                     meta_test_dataset=meta_test_dataset,
                     model=model,
                     loss_function=loss_function)
