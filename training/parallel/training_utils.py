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
testing_size = training_config.testing_size

### Functions for Training ###

# saves a model and optimizer to disk
def checkpoint(model_kwargs, model, filename, optimizer):
    model_dict = {
        'state_dict' : model.state_dict(),
        'model_kwargs' : model_kwargs,
        'optimizer_state_dict' : optimizer.state_dict()
    }
    printf("Checkpointing to {filename}...", end='', flush=True)
    torch.save(model_dict, filename)
    file_size = os.path.getsize(filename) / 1E6
    printf("occupies {file_size:.2f} MB.")

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

### Code for Storing Training Data ###

# represents a molecule and all its jiggled training examples 
class Molecule():
    def __init__(self, name,
                 atomic_symbols,
                 symmetrical_atoms,        # list of lists of 0-indexed atom numbers
                 perturbed_geometries,
                 perturbed_shieldings):
        self.name = name                                       # name of molecule
        self.atomic_symbols = atomic_symbols                   # vector of strings of length n_atoms
        self.n_atoms = len(atomic_symbols)                     # number of atoms
        self.perturbed_geometries = perturbed_geometries       # (n_examples, n_atoms, 3)

        # zero out shieldings for irrelevant atoms
        for i,a in enumerate(atomic_symbols):
            if a not in relevant_elements:
                perturbed_shieldings[:,i]=0.0
        self.perturbed_shieldings = perturbed_shieldings                # (n_examples, n_atoms, 1)
        self.features = get_one_hots(atomic_symbols)                    # (n_atoms, n_elements)
        self.weights = get_weights(atomic_symbols, symmetrical_atoms)   # (n_atoms,)

def str2array(s):
    # https://stackoverflow.com/questions/35612235/how-to-read-numpy-2d-array-from-string
    s = re.sub('\[ +', '[', s.strip())
    s = re.sub('[,\s]+', ', ', s)
    a = ast.literal_eval(s)
    if len(a) == 0 or a is None:
        return []
    else:
        for i, b in enumerate(a):
            for j, _ in enumerate(b):
                 a[i][j] += -1
        return a

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
        self.RMSEs_by_element = None        # { element symbol : RMSE }

    def print_training_status_update(self, epoch, minibatch, n_minibatches):
        losses = np.array(self.minibatch_loss_buffer)
        train_loss = np.mean(losses)
        times = np.array(self.minibatch_training_time_buffer)
        time_per_batch = np.mean(times)
        elapsed = self.cumulative_stopwatch.get_elapsed()
        delta = timedelta(seconds=elapsed)
        delta = str(delta)
        delta = delta[:-5]
        print(f"Epoch {epoch}  Batch {minibatch:5d} / {n_minibatches:5d}   est_train_loss = {train_loss:10.3f}  time_per_batch = {time_per_batch:.2f} s  elapsed = {delta}", end="\r", flush=True)

    def print_testing_status_update(self, epoch, minibatch, n_minibatches):
        pass

    def log_minibatch_loss(self, minibatch_loss, training_time):
        losses, times = self.minibatch_loss_buffer, self.minibatch_training_time_buffer
        losses.append(minibatch_loss)
        times.append(training_time)
        if len(losses) > self.training_window_size:
            losses.popleft()
            times.popleft()

    def log_testing_loss(self, epoch, minibatches, testing_loss,
                         residuals_by_molecule, residuals_by_site_label, RMSEs_by_element):
        pass

# train a single batch
def train_batch(data_list, model, optimizer, training_history):
    # forward pass
    time1 = time.time()
    data = tg.data.Batch.from_data_list(data_list)
    data.to(device)
    output = model(data.x, data.edge_index, data.edge_attr, n_norm=n_norm)
    loss, _ = loss_function(output,data)

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # update results
    minibatch_loss = np.sqrt(loss.item())  # RMSE
    training_time = time.time() - time1
    training_history.log_minibatch_loss(minibatch_loss, training_time)

# compute the testing losses
# testing_molecule_dict: name -> Molecule
# returns: testing_loss,
#          results_dict (molecule name -> residuals (n_examples,n_atoms),
#          results_dict2 (site label -> residuals)
#          RMSE_dict (element -> RMSE)
def compute_testing_loss(testing_dataloader, training_history, epoch, molecule_dict):
    n_minibatches = math.ceil(testing_size/batch_size)
    testing_loss = 0.0
    n_testing_eaxmples_seen = 0
    for minibatch_index, data_list in enumerate(testing_dataloader):
        n_examples_this_minibatch = len(data)
        data = tg.data.Batch.from_data_list(data_list)
        data.to(device)

        with torch.no_grad():
            # run model
            output = model(data.x, data.edge_index, data.edge_attr)

            # compute MSE
            loss, residuals = loss_function(output_data)
            minibatch_loss = np.sqrt(loss.item())
            testing_loss = testing_loss * n_testing_examples_seen + \
                           minibatch_loss * n_examples_this_minibatch
            n_testing_examples_seen += n_examples_this_minibatch
            testing_loss = testing_loss / n_testing_examples_seen

            # store residuals


    # reshape residual data

    training_history.log_testing_loss(epoch, testing_loss)
