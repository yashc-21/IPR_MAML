# Base code is from https://github.com/cs230-stanford/cs230-code-examples
from torch.autograd import Variable
from tqdm import tqdm
from evaluate import evaluate

import logging
import copy
import argparse
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from src.data_loader import fetch_dataloaders,FewShotDataset
from src.data_loader import datasets_test
from torch.utils.data import DataLoader
import os
import utils
import pickle
import math
from src.model import metrics
from src.data_loader import split_emotions
from src.data_loader import SER
from src.model import MetaLearner

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_dir',
    default='data/Omniglot',
    help="Directory containing the dataset")
parser.add_argument(
    '--model_dir',
    default='experiments/base_model',
    help="Directory containing params.json")
parser.add_argument(
    '--restore_file',
    default=None,
    help="Optional, name of the file in --model_dir containing weights to \
          reload before training")  # 'best' or 'train'


def train_single_task(model, task_lr, loss_fn, dataloaders, params):
    """
    Train the model on a single few-shot task.
    We train the model with single or multiple gradient update.
    
    Args:
        model: (MetaLearner) a meta-learner to be adapted for a new task
        task_lr: (float) a task-specific learning rate
        loss_fn: a loss function
        dataloaders: (dict) a dict of DataLoader objects that fetches both of 
                     support set and query set
        params: (Params) hyperparameters
    """
    # extract params
    num_train_updates = params.num_train_updates

    # set model to training mode
    model.train()

    # support set and query set for a single few-shot task
    dl_sup = dataloaders['train']
    X_sup, Y_sup = next(iter(dl_sup))

    # move to GPU if available
    if params.cuda:
        X_sup, Y_sup = X_sup.cuda(1), Y_sup.cuda(1)

    # compute model output and loss
    Y_sup_hat = model(X_sup)
    loss = loss_fn(Y_sup_hat, Y_sup)

    # clear previous gradients, compute gradients of all variables wrt loss
    def zero_grad(params):
        for p in params:
            if p.grad is not None:
                p.grad.zero_()

    # NOTE if we want approx-MAML, change create_graph=True to False
    # optimizer.zero_grad()
    # loss.backward(create_graph=True)
    zero_grad(model.parameters())
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)

    # performs updates using calculated gradients
    # we manually compute adpated parameters since optimizer.step() operates in-place
    adapted_state_dict = model.cloned_state_dict()  # NOTE what about just dict
    adapted_params = OrderedDict()
    for (key, val), grad in zip(model.named_parameters(), grads):
        adapted_params[key] = val - task_lr * grad
        adapted_state_dict[key] = adapted_params[key]

    for _ in range(1, num_train_updates):
        Y_sup_hat = model(X_sup, adapted_state_dict)
        loss = loss_fn(Y_sup_hat, Y_sup)
        zero_grad(adapted_params.values())
        # optimizer.zero_grad()
        # loss.backward(create_graph=True)
        grads = torch.autograd.grad(
            loss, adapted_params.values(), create_graph=True)
        for (key, val), grad in zip(adapted_params.items(), grads):
            adapted_params[key] = val - task_lr * grad
            adapted_state_dict[key] = adapted_params[key]

    return adapted_state_dict


def train_and_evaluate(model,
                       meta_train_classes,
                       meta_test_classes,
                       task_type,
                       meta_optimizer,
                       loss_fn,
                       metrics,
                       params,
                       model_dir,
                       restore_file=None):
    """
    Train the model and evaluate every `save_summary_steps`.

    Args:
        model: (MetaLearner) a meta-learner for MAML algorithm
        meta_train_classes: (list) the classes for meta-training
        meta_train_classes: (list) the classes for meta-testing
        task_type: (subclass of FewShotTask) a type for generating tasks
        meta_optimizer: (torch.optim) an meta-optimizer for MetaLearner
        loss_fn: a loss function
        metrics: (dict) a dictionary of functions that compute a metric using 
                 the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from
                      (without its extension .pth.tar)
    TODO Validation classes
    """
    def zero_grad(params):
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir,
                                    args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, meta_optimizer)

    # params information
    num_classes = params.num_classes
    num_samples = params.num_samples
    num_query = params.num_query
    num_inner_tasks = params.num_inner_tasks
    task_lr = params.task_lr
    meta_lr = params.meta_lr
    reg_lr = 0.01 # hyperparameter for regularizing with generalising gradient

    # TODO validation accuracy
    best_test_acc = 0.0

    # For plotting to see summerized training procedure
    plot_history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    with tqdm(total=params.num_episodes) as t:
        for episode in range(params.num_episodes):
            # Run one episode
            logging.info("Episode {}/{}".format(episode + 1,
                                                params.num_episodes))

            # Run inner loops to get adapted parameters (theta_t`)
            adapted_state_dicts = []
            dataloaders_list = []
            for n_task in range(num_inner_tasks):
                task = task_type(meta_train_classes, num_classes, num_samples,
                                 num_query)
                dataloaders = fetch_dataloaders(['train', 'test','meta'],
                                                task,params)
                # Perform a gradient descent to meta-learner on the task
                a_dict = train_single_task(model, task_lr, loss_fn,
                                           dataloaders, params)
                # Store adapted parameters
                # Store dataloaders for meta-update and evaluation
                adapted_state_dicts.append(a_dict)
                dataloaders_list.append(dataloaders)

            # Update the parameters of meta-learner
            # Compute losses with adapted parameters along with corresponding tasks
            # Updated the parameters of meta-learner using sum of the losses
            meta_loss = 0
            net_clone0 = copy.deepcopy(model)
            net_clone1 = copy.deepcopy(model)
            for p in model.parameters():
                if p.grad is None:
                    if params.cuda:
                        p.grad = Variable(torch.zeros(p.size())).cuda(1)
                    else:
                        p.grad = Variable(torch.zeros(p.size()))
                p.grad.data.zero_()  # not sure this is required

            for n_task in range(num_inner_tasks):
                dataloaders = dataloaders_list[n_task]
                dl_meta = dataloaders['meta']
                X_meta, Y_meta = next(iter(dl_meta))
                if params.cuda:
                    X_meta, Y_meta = X_meta.cuda(1), Y_meta.cuda(1)

                a_dict = adapted_state_dicts[n_task]
                net_clone1.load_state_dict(a_dict)
                zero_grad(net_clone1.parameters())
                Y_meta_hat1 = net_clone1(X_meta)
                loss_t1 = loss_fn(Y_meta_hat1, Y_meta)
                
                grads1 = torch.autograd.grad(loss_t1, net_clone1.parameters(), create_graph=False)

                zero_grad(net_clone0.parameters())
                Y_meta_hat0 = net_clone0(X_meta)
                loss_t0 = loss_fn(Y_meta_hat0, Y_meta)

                grads0 = torch.autograd.grad(loss_t0, net_clone0.parameters(), create_graph=False)
                for p, (g0,g1) in zip(model.parameters(), zip(grads0,grads1)):
                    if p.grad is None:
                        if params.cuda:
                            p.grad = Variable(torch.zeros(p.size())).cuda()
                        else:
                            p.grad = Variable(torch.zeros(p.size()))     
                    p.grad.data.add_((g1+(reg_lr*(g1-g0))/task_lr)/float(num_inner_tasks))
            
            meta_optimizer.step()

            # Evaluate model on new task
            # Evaluate on train and test dataset given a number of tasks (params.num_steps)
            if (episode + 1) % 100 == 0:
                train_metrics = evaluate(model, loss_fn, meta_train_classes,
                                         task_lr, task_type, metrics, params,args,
                                         'train')
                test_metrics = evaluate(model, loss_fn, meta_test_classes,
                                        task_lr, task_type, metrics, params,args,
                                        'test')

                train_loss = train_metrics['loss']
                test_loss = test_metrics['loss']
                train_acc = train_metrics['accuracy']
                test_acc = test_metrics['accuracy']

                is_best = test_acc >= best_test_acc

                # Save weights
                utils.save_checkpoint({
                    'episode': episode + 1,
                    'state_dict': model.state_dict(),
                    'optim_dict': meta_optimizer.state_dict()
                },
                                      is_best=is_best,
                                      checkpoint=model_dir)

                # If best_test, best_save_path
                if is_best:
                    logging.info("- Found new best accuracy")
                    best_test_acc = test_acc

                    # Save best test metrics in a json file in the model directory
                    best_train_json_path = os.path.join(
                        model_dir, "metrics_train_best_weights.json")
                    utils.save_dict_to_json(train_metrics,
                                            best_train_json_path)
                    best_test_json_path = os.path.join(
                        model_dir, "metrics_test_best_weights.json")
                    utils.save_dict_to_json(test_metrics, best_test_json_path)

                # Save latest test metrics in a json file in the model directory
                last_train_json_path = os.path.join(
                    model_dir, "metrics_train_last_weights.json")
                utils.save_dict_to_json(train_metrics, last_train_json_path)
                last_test_json_path = os.path.join(
                    model_dir, "metrics_test_last_weights.json")
                utils.save_dict_to_json(test_metrics, last_test_json_path)

                plot_history['train_loss'].append(train_loss)
                plot_history['train_acc'].append(train_acc)
                plot_history['test_loss'].append(test_loss)
                plot_history['test_acc'].append(test_acc)

                t.set_postfix(
                    tr_acc='{:05.3f}'.format(train_acc),
                    te_acc='{:05.3f}'.format(test_acc),
                    tr_loss='{:05.3f}'.format(train_loss),
                    te_loss='{:05.3f}'.format(test_loss))
                print('\n')

            t.update()

    utils.plot_training_results(args.model_dir, plot_history)


if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    SEED = params.SEED
    meta_lr = params.meta_lr
    num_episodes = params.num_episodes

    # Use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(SEED)
    if params.cuda: torch.cuda.manual_seed(SEED)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    params.in_channels = 3
    # NOTE These params are only applicable to pre-specified model architecture.
    # Split meta-training and meta-testing characters
    
    meta_train_classes, meta_test_classes = split_emotions(SEED)
    task_type = SER

    # Define the model and optimizer
    if params.cuda:
        model = MetaLearner(params).cuda(1)
    else:
        model = MetaLearner(params)
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)

    # fetch loss function and metrics
    loss_fn = nn.NLLLoss()
    model_metrics = metrics

    # Train the model
    logging.info("Starting training for {} episode(s)".format(num_episodes))
    train_and_evaluate(model, meta_train_classes, meta_test_classes, task_type,
                       meta_optimizer, loss_fn, model_metrics, params,
                       args.model_dir, args.restore_file)