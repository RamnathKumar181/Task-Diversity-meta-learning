from typing import Tuple
from pathlib import Path


class DataConfig(object):
    """Common configuration options for creating data processing pipelines."""

    def __init__(self, path, batch_size=256, val_batch_size=1, num_workers=4, image_size=84):
        """Initialize a DataConfig.
        """

        # General info
        self.path: Path = Path(path)
        self.batch_size: int = batch_size
        self.val_batch_size: int = val_batch_size
        self.num_workers: int = num_workers
        self.shuffle: bool = True

        # Transforms and augmentations
        self.image_size: Tuple[int, int] = image_size
        self.test_transforms: bool = True
        self.train_transforms: bool = True


class EpisodeDescriptionConfig(object):
    """Configuration options for episode characteristics."""

    def __init__(self, num_ways, num_query, num_support, min_ways=5, max_ways_upper_bound=50, max_num_query=10,
                 max_support_set_size=500, max_support_size_contrib_per_class=100, min_log_weight=-0.69314718055994529,
                 max_log_weight=0.69314718055994529, ignore_dag_ontology=False, ignore_bilevel_ontology=False,
                 ignore_hierarchy_probability=False, min_examples_in_class=0, num_unique_descriptions=0):
        """Initialize a EpisodeDescriptionConfig.

        This is used in sampling.py in Trainer and in EpisodeDescriptionSampler to
        determine the parameters of episode creation relating to the ways and shots.

        Args:
            num_ways: Integer, fixes the number of classes ("ways") to be used in each
                episode. None leads to variable way.
            num_support: An integer, a tuple of two integers, or None. In the first
                case, the number of examples per class in the support set. In the
                second case, the range from which to sample the number of examples per
                class in the support set. Both of these cases would yield class-balanced
                episodes, i.e. all classes have the same number of support examples.
                Finally, if None, the number of support examples will vary both within
                each episode (introducing class imbalance) and across episodes.
            num_query: Integer, fixes the number of examples for each class in the
                query set.
            min_ways: Integer, the minimum value when sampling ways.
            max_ways_upper_bound: Integer, the maximum value when sampling ways. Note
                that the number of available classes acts as another upper bound.
            max_num_query: Integer, the maximum number of query examples per class.
            max_support_set_size: Integer, the maximum size for the support set.
            max_support_size_contrib_per_class: Integer, the maximum contribution for
                any given class to the support set size.
            min_log_weight: Float, the minimum log-weight to give to any particular
                class when determining the number of support examples per class.
            max_log_weight: Float, the maximum log-weight to give to any particular
                class.
            ignore_dag_ontology: Whether to ignore ImageNet's DAG ontology when
                sampling classes from it. This has no effect if ImageNet is not part of
                the benchmark.
            ignore_bilevel_ontology: Whether to ignore Omniglot's DAG ontology when
                sampling classes from it. This has no effect if Omniglot is not part of
                the benchmark.
            ignore_hierarchy_probability: Float, if using a hierarchy, this flag makes
                the sampler ignore the hierarchy for this proportion of episodes and
                instead sample categories uniformly.
            simclr_episode_fraction: Float, fraction of episodes that will be
                converted to SimCLR Episodes as described in the CrossTransformers
                paper.
            min_examples_in_class: An integer, the minimum number of examples that a
                class has to contain to be considered. All classes with fewer examples
                will be ignored. 0 means no classes are ignored, so having classes with
                no examples may trigger erroargsrs later. For variable shots, a value of 2
                makes it sure that there are at least one support and one query samples.
                For fixed shots, you could set it to `num_support + num_query`.

        Raises:
            RuntimeError: if incompatible arguments are passed.
        """
        arg_groups = {
            'num_ways': (num_ways,
                         ('min_ways', 'max_ways_upper_bound'),
                         (min_ways, max_ways_upper_bound)),
            'num_query': (num_query, ignore_bilevel_ontology
                          ('max_num_query',),
                          (max_num_query,)),
            'num_support': (num_support,  # noqa: E131
                            ('max_support_set_size', 'max_support_size_contrib_per_class',
                             'min_log_weight', 'max_log_weight'),
                            (max_support_set_size, max_support_size_contrib_per_class,
                             min_log_weight, max_log_weight)),
        }

        for first_arg_name, values in arg_groups.items():
            first_arg, required_arg_names, required_args = values

            if ((first_arg is None) and any(arg is None for arg in required_args)):
                # Get name of the nones
                none_arg_names = [name for var, name in zip(required_args, required_arg_names)
                                  if var is None]

                raise RuntimeError(
                    'The following arguments: %s can not be None, since %s is None. '
                    'Please ensure the following arguments of EpisodeDescriptionConfig are set: '
                    '%s' % (none_arg_names, first_arg_name, none_arg_names))

        self.num_ways: int = num_ways if num_ways > 0 else None
        self.num_support: int = num_support if num_support > 0 else None
        self.num_query: int = num_query if num_query > 0 else None
        self.min_ways: int = min_ways
        self.max_ways_upper_bound: int = max_ways_upper_bound
        self.max_num_query: int = max_num_query
        self.max_support_set_size: int = max_support_set_size
        self.max_support_size_contrib_per_class: int = max_support_size_contrib_per_class
        self.min_log_weight: float = min_log_weight
        self.max_log_weight: float = max_log_weight
        self.ignore_dag_ontology: bool = ignore_dag_ontology
        self.ignore_bilevel_ontology: bool = ignore_bilevel_ontology
        self.ignore_hierarchy_probability: bool = ignore_hierarchy_probability
        self.min_examples_in_class: int = min_examples_in_class
        self.num_unique_descriptions: int = num_unique_descriptions

        self.use_dag_ontology: bool
        self.use_bilevel_ontology: bool

    def max_ways(self) -> int:
        """Returns the way (maximum way if variable) of the episode."""
        return self.num_ways or self.max_ways_upper_bound
