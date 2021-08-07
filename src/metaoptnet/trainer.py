import json
import time
import os
import logging
import torch
from src.metaoptnet.metalearners import MetaOptNet
from src.utils import get_benchmark_by_name
from torchmeta.utils.data import BatchMetaDataLoader as BMD
import wandb


class MetaOptNetTrainer():
    def __init__(self, args):
        self.args = args
        self.highest_val = 0
        self.device = self._device()
        logging.basicConfig(level=logging.DEBUG if self.args.verbose else logging.INFO)
        logging.info(f"Configuration while training: {args}")
        self._build()

    def _build(self):
        self._create_config_file()
        self._build_loaders_and_optim()
        self._build_metalearner()
        self._train()
        wandb.save(f"{self.args.model_path}")

    def _create_config_file(self):
        if (self.args.output_folder is not None):
            if not os.path.exists(self.args.output_folder):
                os.makedirs(self.args.output_folder)
                logging.debug('Creating folder `{0}`'.format(self.args.output_folder))
            folder = os.path.join(self.args.output_folder,
                                  time.strftime('%Y-%m-%d-%H%M%S'))
            os.makedirs(folder)
            logging.debug('Creating folder `{0}`'.format(folder))

            self.args.folder = os.path.abspath(self.args.folder)
            model_dest = '{0}_model.th'.format(self.args.model)
            self.args.model_path = os.path.abspath(os.path.join(folder, model_dest))

            # Save configurations in a config.json file
            with open(os.path.join(folder, 'config.json'), 'w') as f:
                json.dump(vars(self.args), f, indent=2)
            logging.info('Saving configuration file in `{0}`'.format(
                os.path.abspath(os.path.join(folder, 'config.json'))))

    def _build_loaders_and_optim(self):
        self.benchmark = get_benchmark_by_name(self.args.model,
                                               self.args.dataset,
                                               self.args.folder,
                                               self.args.num_ways,
                                               self.args.num_shots,
                                               self.args.num_shots_test,
                                               hidden_size=self.args.hidden_size)
        if self.args.task_sampler == 'no_diversity_task':
            logging.info("Using no_diversity_task sampler:\n\n")
            from src.task_sampler import BatchMetaDataLoaderNDT as BMD_NDT
            self.meta_train_dataloader = BMD_NDT(self.benchmark.meta_train_dataset,
                                                 batch_size=self.args.batch_size,
                                                 shuffle=True,
                                                 num_workers=self.args.num_workers,
                                                 pin_memory=True)
        elif self.args.task_sampler == 'no_diversity_batch':
            logging.info("Using no_diversity_batch sampler:\n\n")
            from src.task_sampler import BatchMetaDataLoaderNDB as BMD_NDB
            self.meta_train_dataloader = BMD_NDB(self.benchmark.meta_train_dataset,
                                                 batch_size=self.args.batch_size,
                                                 shuffle=True,
                                                 num_workers=self.args.num_workers,
                                                 pin_memory=True)
        elif self.args.task_sampler == 'no_diversity_tasks_per_batch':
            logging.info("Using no_diversity_tasks_per_batch sampler:\n\n")
            from src.task_sampler import BatchMetaDataLoaderNDTB as BMD_NDTB
            self.meta_train_dataloader = BMD_NDTB(self.benchmark.meta_train_dataset,
                                                  batch_size=self.args.batch_size,
                                                  shuffle=True,
                                                  num_workers=self.args.num_workers,
                                                  pin_memory=True)
        else:
            logging.info("Using uniform_task sampler:\n\n")
            self.meta_train_dataloader = BMD(self.benchmark.meta_train_dataset,
                                             batch_size=self.args.batch_size,
                                             shuffle=True,
                                             num_workers=self.args.num_workers,
                                             pin_memory=True)
        self.meta_val_dataloader = BMD(self.benchmark.meta_val_dataset,
                                       batch_size=self.args.batch_size,
                                       shuffle=True,
                                       num_workers=self.args.num_workers,
                                       pin_memory=True)

        self.meta_optimizer = torch.optim.SGD([{'params': self.benchmark.model.embedding_network.parameters()},
                                               {'params': self.benchmark.model.cls_head.parameters()}],
                                              lr=self.args.meta_lr,
                                              momentum=self.args.momentum,
                                              weight_decay=self.args.weight_decay,
                                              nesterov=True)

        def lambda_epoch(e): return 1.0 if e < 20 else (
            0.06 if e < 40 else 0.012 if e < 50 else (0.0024))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.meta_optimizer, lr_lambda=lambda_epoch, last_epoch=-1)
        wandb.watch(self.benchmark.model)

    def _build_metalearner(self):
        self.metalearner = MetaOptNet(self.benchmark.model,
                                      self.meta_optimizer,
                                      scheduler=self.scheduler,
                                      num_adaptation_steps=self.args.num_steps,
                                      step_size=self.args.step_size,
                                      loss_function=self.benchmark.loss_function,
                                      device=self.device,
                                      num_ways=self.args.num_ways,
                                      num_shots=self.args.num_shots,
                                      num_shots_test=self.args.num_shots_test)

    def _train(self):
        best_value = None
        for epoch in range(self.args.num_epochs):
            self.metalearner.train(self.meta_train_dataloader,
                                   max_batches=self.args.num_batches,
                                   verbose=self.args.verbose,
                                   desc='Training',
                                   leave=False)
            results = self.metalearner.evaluate(self.meta_val_dataloader,
                                                max_batches=self.args.num_batches,
                                                verbose=self.args.verbose,
                                                desc='Validation')
            if (epoch+1) % self.args.log_interval == 0:
                wandb.log({"Accuracy": results['accuracies']})
            # Save best model
            if ((best_value is None)
                    or (best_value < results['accuracies'])):
                best_value = results['accuracies']
                save_model = True
            else:
                save_model = False

            if save_model and (self.args.output_folder is not None):
                with open(self.args.model_path, 'wb') as f:
                    torch.save(self.benchmark.model.state_dict(), f)
        self.highest_val = best_value

        if hasattr(self.benchmark.meta_train_dataset, 'close'):
            self.benchmark.meta_train_dataset.close()
            self.benchmark.meta_val_dataset.close()

    def get_result(self):
        return tuple([self.highest_val])

    def _device(self):
        return torch.device('cuda' if self.args.use_cuda
                            and torch.cuda.is_available() else 'cpu')


class MetaOptNetTester():
    def __init__(self, config):
        self.config = config
        self.highest_test = 0
        self.device = self._device()
        logging.basicConfig(level=logging.DEBUG if self.config['verbose'] else logging.INFO)
        logging.info(f"Configuration while testing: {config}")
        self._build()

    def _build(self):
        self._build_loader()
        self._build_metalearner()
        self._test()

    def _build_loader(self):
        self.benchmark = get_benchmark_by_name(self.config['model'],
                                               self.config['dataset'],
                                               self.config['folder'],
                                               self.config['num_ways'],
                                               self.config['num_shots'],
                                               self.config['num_shots_test'],
                                               hidden_size=self.config['hidden_size'])

        self.meta_test_dataloader = BMD(self.benchmark.meta_test_dataset,
                                        batch_size=self.config['batch_size'],
                                        shuffle=True,
                                        num_workers=self.config['num_workers'],
                                        pin_memory=True)

        with open(self.config['model_path'], 'rb') as f:
            self.benchmark.model.load_state_dict(torch.load(f, map_location=self.device))

    def _build_metalearner(self):

        self.metalearner = MetaOptNet(self.benchmark.model,
                                      num_adaptation_steps=self.config['num_steps'],
                                      step_size=self.config['step_size'],
                                      loss_function=self.benchmark.loss_function,
                                      device=self.device,
                                      num_ways=self.config['num_ways'])

    def _test(self):
        results = self.metalearner.evaluate(self.meta_test_dataloader,
                                            max_batches=self.config['num_batches'],
                                            verbose=self.config['verbose'],
                                            desc='Testing')
        dirname = os.path.dirname(self.config['model_path'])
        with open(os.path.join(dirname, 'results.json'), 'w') as f:
            json.dump(results, f)

        self.highest_test = results['accuracies']

    def get_result(self):
        return tuple([self.highest_test])

    def _device(self):
        return torch.device('cuda' if self.config['use_cuda']
                            and torch.cuda.is_available() else 'cpu')
