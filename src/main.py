import argparse
from src.logger import Logger
import gc
import json
from glob import glob
from src.utils import seed_everything
from src.maml import MAMLTrainer, MAMLTester


def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser('Task_Diversity')
    # General
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of experimental runs (default: 5).')
    parser.add_argument('--model', type=str,
                        choices=['maml', 'protonet'],
                        default='maml',
                        help='Name of the model to be used (default: MAML).')
    parser.add_argument('--task_sampler', type=str,
                        choices=['random'],
                        default='random',
                        help='Type of task sampler to be used '
                        '(default: random).')
    parser.add_argument('folder', type=str,
                        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--dataset', type=str,
                        choices=['sinusoid', 'omniglot', 'miniimagenet'],
                        default='omniglot',
                        help='Name of the dataset (default: omniglot).')
    parser.add_argument('--output-folder', type=str, default=None,
                        help='Path to the output folder to save the model.')
    parser.add_argument('--num-ways', type=int, default=5,
                        help='Number of classes per task (N in "N-way",'
                        ' default: 5).')
    parser.add_argument('--num-shots', type=int, default=5,
                        help='Number of training example per class '
                        '(k in "k-shot", default: 5).')
    parser.add_argument('--num-shots-test', type=int, default=15,
                        help='Number of test example per class.'
                        ' If negative, same as the number '
                        'of training examples `--num-shots` (default: 15).')

    # Model
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='Number of channels in each convolution '
                        'layer of the VGG network (default: 64).')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=25,
                        help='Number of tasks in a batch of tasks '
                        '(default: 25).')
    parser.add_argument('--num-steps', type=int, default=1,
                        help='Number of fast adaptation steps, '
                        'ie. gradient descent updates (default: 1).')
    parser.add_argument('--num-epochs', type=int, default=50,
                        help='Number of epochs of meta-training '
                        '(default: 50).')
    parser.add_argument('--num-batches', type=int, default=100,
                        help='Number of batch of tasks per epoch '
                        '(default: 100).')
    parser.add_argument('--step-size', type=float, default=0.1,
                        help='Size of the fast adaptation step,'
                        ' ie. learning rate in the '
                        'gradient descent update (default: 0.1).')
    parser.add_argument('--first-order', action='store_true',
                        help='Use the first order approximation,'
                        ' do not use higher-order derivatives during '
                        'meta-optimization.')
    parser.add_argument('--meta-lr', type=float, default=0.001,
                        help='Learning rate for the meta-optimizer '
                        '(optimization of the outer loss). The default '
                        'optimizer is Adam (default: 1e-3).')

    # Misc
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of workers to use for data-loading '
                        '(default: 1).')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--train', action='store_true')

    args = parser.parse_args()

    if args.num_shots_test <= 0:
        args.num_shots_test = args.num_shots
    return args


if __name__ == '__main__':
    args = parse_args()
    log = Logger(args.runs)
    if args.train:
        log.reset(args.runs, info="Training Accuracy")
        if args.model == 'maml':
            """
            MAML Trainer
            """
            for run in range(args.runs):
                gc.collect()
                seed_everything(run)
                maml_trainer = MAMLTrainer(args)
                log.add_result(run, maml_trainer.get_result())
        print(f"Average Performance of {args.model} on {args.dataset}:")
        log.print_statistics()
    else:
        log.reset(args.runs, info="Testing Accuracy")
        for run, config_file in glob(f'{args.output_folder}/*/config.json'):
            with open(config_file, 'r') as f:
                config = json.load(f)
            if args.folder is not None:
                config['folder'] = args.folder
            if args.num_steps > 0:
                config['num_steps'] = args.num_steps
            if args.num_batches > 0:
                config['num_batches'] = args.num_batches
            config['verbose'] = args.verbose
            if args.model == 'maml':
                """
                MAML Test
                """
                maml_tester = MAMLTester(config)
                log.add_result(run, maml_tester.get_result())
        print(f"Average Performance of {args.model} on {args.dataset}:")
        log.print_statistics()
        pass
