from enum import Enum
import re
import math
import os
import time
from datetime import timedelta
from collections import deque
import numpy as np
import h5py
import torch
torch.set_default_dtype(torch.float64)
import torch_geometric as tg
import e3nn
import e3nn.point.data_helpers as dh
import training_config
from stopwatch import Stopwatch
import matplotlib
import matplotlib.pyplot as plt

### Code to Generate Molecules ###

# all expected elements
all_elements = training_config.all_elements
n_elements = len(all_elements)

# so we can normalize training data for the nuclei to be predicted
relevant_elements = training_config.relevant_elements

# cpu or gpu
device = training_config.device

# other parameters
n_norm = training_config.n_norm
training_size = training_config.training_size
testing_size = training_config.testing_size
batch_size = training_config.batch_size

### Functions for Training ###

# saves a model and optimizer to disk
def checkpoint(model_kwargs, model, filename, optimizer):
    model_dict = {
        'state_dict' : model.state_dict(),
        'model_kwargs' : model_kwargs,
        'optimizer_state_dict' : optimizer.state_dict()
    }
    print("                                                                                                      ", end="\r", flush=True)
    print(f"Checkpointing to {filename}...", end='', flush=True)
    torch.save(model_dict, filename)
    file_size = os.path.getsize(filename) / 1E6
    print(f"occupies {file_size:.2f} MB.", end='\r', flush=True)

# mean-squared loss (not RMS!)
def loss_function(output, data):
    predictions = output
    observations = data.y
    weights = data.weights
    normalization = weights.sum()
    residuals = (predictions-observations)
    loss = residuals.square() * weights
    loss = loss.sum() / normalization
    return loss, residuals

### Training Code ###

class TrainingHistory():
    # running: whether the clock should be running
    # training_window_size: moving average for training_loss over this many minibatches
    def __init__(self, training_window_size=10):
        self.training_window_size = training_window_size
        self.cumulative_stopwatch = Stopwatch(running=True)

        # parallel indexed lists
        self.epochs = []             # which epoch the data point was recorded during
        self.elapsed_times = []      # how much time has passed since training started
                                     # (only includes times when the internal stopwatch
                                     #  was running)
        self.minibatches = []        # how many cumulative minibatches have been seen
                                     # across all epochs
        self.training_losses = []    # training loss using moving average method
        self.testing_losses = []     # testing losses 

        # moving average window for training losses
        self.minibatch_loss_buffer = deque()                # training loss for recent minibatches
        self.minibatch_training_time_buffer = deque()       # time taken to evaluate recent minibatches

        # detailed residual stats for test set
        self.residuals_by_molecule = None   # { molecule name : residuals (n_examples,n_atoms) }
        self.residuals_by_site_label = None # { molecule name : residuals }
        self.stats_by_element = None        # { element symbol : (mean error, RMSE) }

    def print_training_status_update(self, epoch, minibatch, n_minibatches, t_wait, t_batch, q1, q2):
        losses = np.array(self.minibatch_loss_buffer)
        train_loss = np.mean(losses)
        times = np.array(self.minibatch_training_time_buffer)
        time_per_batch = np.mean(times)
        elapsed = self.cumulative_stopwatch.get_elapsed()
        delta = timedelta(seconds=elapsed)
        delta = str(delta)
        delta = delta[:-5]

        # t_train is the average training time for the recent batches
        # wait time is the time spent waiting to accumulate the batch from the queue
        # in the last batch only, same goes for t_batch
        print(f"{epoch} : {minibatch} / {n_minibatches}  train_loss = {train_loss:10.3f}  t_train = {time_per_batch:.2f} s  t_wait = {t_wait:.2f} s  t_batch = {t_batch:.2f}  mol_q = {q1}  dn_q = {q2}  t = {delta}      ", end="\r", flush=True)

    def log_minibatch_loss(self, minibatch_loss, training_time):
        losses, times = self.minibatch_loss_buffer, self.minibatch_training_time_buffer
        losses.append(minibatch_loss)
        times.append(training_time)
        if len(losses) > self.training_window_size:
            losses.popleft()
            times.popleft()

    def log_testing_loss(self, epoch, minibatches, testing_loss,
                         residuals_by_molecule, residuals_by_site_label, RMSEs_by_element):
        self.epochs.append(epoch)
        self.elapsed_times.append(self.cumulative_stopwatch.get_elapsed())
        self.testing_losses.append(testing_loss)
        losses = np.array(self.minibatch_loss_buffer)
        training_loss = np.mean(losses)
        self.training_losses.append(training_loss)
        self.minibatches.append(minibatches)
        self.residuals_by_molecule = residuals_by_molecule
        self.residuals_by_site_label = residuals_by_site_label

    def write_files(self, checkpoint_prefix):
        # save training/test loss graph
        print("                                                                                                  ", end="\r", flush=True)
#        print("Saving test/train loss graph...", end="", flush=True)
#        graph_filename = f"{checkpoint_prefix}-graph.png"
#        n_minibatches = math.ceil(training_size/batch_size)
#        x_test = []
#        for epoch, minibatches in zip(self.epochs, self.minibatches):
#            fractional_epoch = epoch + minibatches/n_minibatches - 1
#            x_test.append(fractional_epoch)
#        x_test = np.array(x_test)
#        x_train = x_test - self.training_window_size / n_minibatches
#        plt.figure(figsize=(12,8))
#        plt.plot(x_train, self.training_losses, "ro-", label="train")
#        plt.plot(x_test, self.testing_losses, "bo-", label="test")
#        plt.legend(loc="best")
#        plt.savefig(graph_filename)
#
        # save raw data
        history_filename = f"{checkpoint_prefix}-training_history.torch"
        print("saving training history...", end="", flush=True)
        torch.save(self, history_filename)
        print("done", end="\r", flush=True)

# train a single batch
def train_batch(data_list, model, optimizer, training_history):
    # set model to training mode (for batchnorm)
    model.train()

    # forward pass
    time1 = time.time()
    data = tg.data.Batch.from_data_list(data_list)
    time2 = time.time()
    data.to(device)
    output = model(data.x, data.edge_index, data.edge_attr, n_norm=n_norm)
    loss, _ = loss_function(output,data)

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # update results
    minibatch_loss = np.sqrt(loss.item())  # RMSE
    training_time = time.time() - time2
    t_batch = time2 - time1
    training_history.log_minibatch_loss(minibatch_loss, training_time)
    return t_batch

# compute the testing losses
# molecules_dict: name -> Molecule
def compute_testing_loss(model, testing_dataloader, training_history, molecules_dict,
                         epoch, minibatches_seen):
    print("\ntesting...", end="\r", flush=True)

    # set model to testing mode (for batchnorm)
    model.eval()
    time1 = time.time()
    n_minibatches = math.ceil(testing_size/batch_size)
    testing_loss = 0.0
    n_testing_examples_seen = 0
    residuals_by_molecule = {}
    residuals_by_site_label = {}
    stats_by_element = {}
    for minibatch_index, data_list in enumerate(testing_dataloader):
        data = tg.data.Batch.from_data_list(data_list)
        data.to(device)
        n_examples_this_minibatch = len(data)

        with torch.no_grad():
            # run model
            output = model(data.x, data.edge_index, data.edge_attr, n_norm=n_norm)

            # compute MSE
            loss, residuals = loss_function(output, data)
            minibatch_loss = np.sqrt(loss.item())
            testing_loss = testing_loss * n_testing_examples_seen + \
                           minibatch_loss * n_examples_this_minibatch
            n_testing_examples_seen += n_examples_this_minibatch
            testing_loss = testing_loss / n_testing_examples_seen

            # store residuals
            residuals = residuals.squeeze(-1).cpu().numpy()
            i = 0
            for name in data.name:
                molecule = molecules_dict[name]
                n_atoms = molecule.n_atoms
                if name not in residuals_by_molecule:
                    residuals_by_molecule[name] = []
                subset = residuals[i:i+n_atoms]
                residuals_by_molecule[name].append(subset)
                i += n_atoms

        # interim status update
        print(f"testing  {minibatch_index+1:5d} / {n_minibatches:5d}   minibatch_test_loss = {minibatch_loss:<10.3f}   overall_test_loss = {testing_loss:<10.3f}", end="\r", flush=True)

    # reshape residual data
    all_residuals = { element : [] for element in relevant_elements }  # element -> [residuals]
    for name, results in residuals_by_molecule.items():
        results = np.array(results).T
        molecule = molecules_dict[name]
        atomic_symbols = molecule.atomic_symbols
        for atomic_index, this_result in enumerate(results):
            element = atomic_symbols[atomic_index]
            if element not in relevant_elements:
                continue
            site_label = "f{name}_{element}{atomic_index+1}"
            residuals_by_site_label[site_label] = this_result
            all_residuals[element].extend(this_result)

    # compute mean errors and RMSEs
    for element, residuals in all_residuals.items():
        residuals = np.array(residuals)
        mean_error = np.mean(residuals)
        RMSE = np.sqrt(np.mean(np.square(residuals)))
        stats_by_element[element] = (mean_error,RMSE)

    # log results
    training_history.log_testing_loss(epoch, minibatches_seen, testing_loss,
                                      residuals_by_molecule, residuals_by_site_label,
                                      stats_by_element)

    # print update
    elapsed = time.time() - time1
    print(f"           testing_loss = {testing_loss:10.3f}   t_test = {elapsed:.2f} s                                        ")
    print("          means / RMSEs =   ", end="")
    for element, (mean_error,RMSE) in stats_by_element.items():
        print(f"{element} : {mean_error:.3f} / {RMSE:.3f}    ", end="")
    print(flush=True)
