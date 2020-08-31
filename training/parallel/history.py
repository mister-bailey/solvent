from stopwatch import Stopwatch
from collections import deque
import numpy as np
import torch
from datetime import timedelta
import matplotlib
import matplotlib.pyplot as plt
import math
import time
from datetime import timedelta
from training_utils import loss_function
from training_config import Config


class TrainTestHistory:
    def __init__(self, batches_per_epoch, checkpoint_prefix, testing_batches, relevant_elements,
                 molecule_dict, device, number_to_symbol=None, smoothing_window=10, store_residuals=False):
        self.train = TrainingHistory(batches_per_epoch, smoothing_window)
        self.test = TestingHistory(testing_batches, relevant_elements, molecule_dict, device,
                                  number_to_symbol=number_to_symbol, store_residuals=store_residuals)
        self.checkpoint_prefix = checkpoint_prefix
    
    # save raw data
    def save(self):
        history_filename = f"{self.checkpoint_prefix}-history.torch"
        print("Saving train/test history... ", end="", flush=True)
        torch.save(self, history_filename)
        print("Done.", end="\r", flush=True)

    def plot(self, figure=None, x_axis='batch_number', y_axis='smoothed_loss'): # x_axis = 'time' is also allowed
        if figure is None:
            plt.figure(figsize=(12,8))
        self.train.plot(figure, x_axis, y_axis)
        self.test.plot(figure, x_axis)
        if figure is None:
            plt.legend(loc="best")

    def log_batch(self, *args, **kwargs):
        self.train.log_batch(*args, **kwargs)

    def run_test(self, model, batch_number=None, epoch=None, batch_in_epoch=None, elapsed_time=None, *args, **kwargs):
        if batch_number is None: batch_number = self.train.batch_number[-1]
        if epoch is None: epoch = self.train.epoch[-1]
        if batch_in_epoch is None: batch_in_epoch = self.train.batch_in_epoch[-1]
        if elapsed_time is None: elapsed_time = self.train.elapsed_time[-1]
        self.test.run_test(model, batch_number, epoch, batch_in_epoch, elapsed_time, *args, **kwargs)
        


# some utility functions for incrementing from an empty list
def last(seq, min=0):
    if len(seq) == 0:
        return min
    return seq[-1]

def new_or_inc(seq, next, min=0, inc=1):
    if next is None:
        seq.append(last(seq, min) + inc)
    else:
        seq.append(next)

def new_or_old(seq, next, min=0):
    if next is None:
        seq.append(last(seq, min))
    else:
        seq.append(next)


class TrainingHistory:
    def __init__(self, batches_per_epoch, smoothing_window=10):
        self.batches_per_epoch = batches_per_epoch
        self.smoothing_window = smoothing_window

        # initialize the lists we will be accumulating
        self.batch_number=[]
        self.epoch=[]
        self.batch_in_epoch=[]
        self.elapsed_time=[]

        self.molecules_per_batch=[]
        self.atoms_per_batch=[]

        self.loss=[]
        self.smoothed_loss=[]

    def log_batch(self, batch_time, wait_time, molecules_per_batch, atoms_per_batch, loss,
                  batch_number=None, epoch=None, batch_in_epoch=None, verbose=True):
        self.elapsed_time.append(last(self.elapsed_time) + batch_time)

        # if you don't provide batch numbers or epoch numbers, we will make reasonable assumptions:
        new_or_inc(self.batch_number, batch_number, min=0)
        if batch_in_epoch is None:
            if epoch != last(self.epoch):
                batch_in_epoch = 1
            else:
                batch_in_epoch = last(self.batch_in_epoch) + 1
        self.batch_in_epoch.append(batch_in_epoch)
        new_or_old(self.epoch, epoch, min=1)


        self.elapsed_time.append(last(self.elapsed_time) + batch_time)
        self.molecules_per_batch.append(molecules_per_batch)
        self.atoms_per_batch.append(atoms_per_batch)
        self.loss.append(loss)
        window = min(len(self.loss), self.smoothing_window)
        self.smoothed_loss.append(sum(self.loss[-window:]) / window)

        if verbose:
            print(f"{self.epoch[-1]} : {self.batch_in_epoch[-1]} / {self.batches_per_epoch}  train_loss = {self.smoothed_loss[-1]:10.3f}"
                  f"  t_train = {batch_time:.2f} s  t_wait = {wait_time:.2f} s  t = {str(timedelta(seconds=self.elapsed_time[-1]))[:-5]}   ",
                   end="\r", flush=True)
        
    # x_axis = 'time' and y_axis = 'loss' are also allowed
    def plot(self, figure=None, x_axis='batch_number', y_axis='smoothed_loss'):
        x_axis = self.batch_number if x_axis=='batch number' else self.elapsed_time
        y_axis = self.smoothed_loss if y_axis=='smoothed_loss' else self.loss
        
        if figure is None:
            plt.figure(figsize=(12,8))
        plt.plot(np.array(x_axis), np.array(y_axis), "ro-", label="train")
        if figure is None:
            plt.legend(loc="best")



class TestingHistory():

    # training_window_size: moving average for training_loss over this many minibatches
    def __init__(self, testing_batches, relevant_elements, molecule_dict, device,
            number_to_symbol=None, store_residuals=False):
        self.testing_batches = testing_batches
        self.device = device
        self.relevant_elements = relevant_elements
        self.number_to_symbol = number_to_symbol if number_to_symbol else Config().number_to_symbol
        self.store_residuals = store_residuals

        # for each relevant element, gives you a list of atom indices in the testing set
        atom_indices = {e:[] for e in relevant_elements}
        atom_index = 0
        for batch in testing_batches:
            for example in batch.example_list:
                for e in molecule_dict[example.ID].atomic_numbers:
                    if e in relevant_elements:
                        atom_indices[e].append(atom_index)
                    atom_index += 1
        self.atom_indices = {e:np.array(ai) for e,ai in atom_indices.items()}

        # precompute weight per testing batch and total testing weight
        self.batch_weights = torch.tensor([torch.sum(batch.weights) for batch in testing_batches])
        self.total_weight = sum(self.batch_weights)

        # initialize the lists we will be accumulating
        self.batch_number = []
        self.epoch = []
        self.batch_in_epoch = []
        self.elapsed_time = []
        self.loss = []
        self.mean_error_by_element = []
        self.RMSE_by_element = []


    def plot(self, figure=None, x_axis='batch_number'): # x_axis = 'time' is also allowed
        x_axis = self.batch_number if x_axis=='batch number' else self.elapsed_time
        if figure is None:
            plt.figure(figsize=(12,8))
        plt.plot(np.array(x_axis), np.array(self.loss), "bo-", label="test")
        if figure is None:
            plt.legend(loc="best")

    def run_test(self, model, batch_number, epoch, batch_in_epoch, elapsed_time, verbose=True, log=True):
        if verbose: print("\nTesting batches...", end="\r", flush=True)

        time0 = time.time()

        losses = []
        residual_chunks = []
        model.eval() # don't compute running means
        with torch.no_grad(): # don't compute gradients
            for batch in self.testing_batches:
                batch.to(self.device)
                loss, chunk = loss_function(model(batch.x, batch.edge_index, batch.edge_attr), batch)
                losses.append(loss)
                residual_chunks.append(chunk)

        if verbose: print("Collating batch results...", end="\r", flush=True)
        loss = (torch.dot(torch.tensor(losses), self.batch_weights) / self.total_weight).sqrt()
        residuals = torch.cat(residual_chunks)

        if verbose: print("Calculating stats by element...", end="\r", flush=True)
        residuals_by_element = {e:residuals[self.atom_indices[e]] for e in self.relevant_elements}

        # compute mean errors and RMSEs
        mean_error_by_element = {e:residuals_by_element[e].mean() for e in self.relevant_elements}
        RMSE_by_element = {e:residuals_by_element[e].square().mean().sqrt() for e in self.relevant_elements}

        time1 = time.time()
        test_time = time1 - time0

        if verbose:
            print(f"  Test loss = {loss:6.3f}   Test time = {test_time:.2f}")
            print(f"  Element   Mean Error    RMSE")
            #print(f" <5> Ee <6>  012.345 <5> 012.345")
            for e in self.relevant_elements:
                print(f"     {self.number_to_symbol[e].rjust(2)}      {mean_error_by_element[e]:3.3f}     {RMSE_by_element[e]:3.3f}")

        if log:
            self.log_test(batch_number, epoch, batch_in_epoch, elapsed_time,
                    loss, mean_error_by_element, RMSE_by_element, residuals_by_element)

    def log_test(self, batch_number, epoch, batch_in_epoch, elapsed_time,
            loss, mean_error_by_element, RMSE_by_element, residuals_by_element):
        self.batch_number.append(batch_number)
        self.epoch.append(epoch)
        self.batch_in_epoch.append(batch_in_epoch)
        self.elapsed_time.append(elapsed_time)
        self.loss.append(loss)
        self.mean_error_by_element.append(mean_error_by_element)
        self.RMSE_by_element.append(RMSE_by_element)
        if self.store_residuals:
            self.residuals_by_element = residuals_by_element



        




def compute_testing_loss(model, testing_batches, device, relevant_elements, training_history,
                        molecules_dict, epoch, minibatches_seen):
    print("\ntesting...", end="\r", flush=True)

    # set model to testing mode (for batchnorm)
    model.eval()
    time1 = time.time()
    n_minibatches = len(testing_batches)
    testing_loss = 0.0
    n_testing_examples_seen = 0
    residuals_by_molecule = {}
    residuals_by_site_label = {}
    stats_by_element = {}
    for minibatch_index, minibatch in enumerate(testing_batches):
        minibatch.to(device)

        with torch.no_grad():
            # run model
            output = model(minibatch.x, minibatch.edge_index, minibatch.edge_attr)

            # compute MSE
            loss, residuals = loss_function(output, minibatch)
            minibatch_loss = np.sqrt(loss.item())
            testing_loss = testing_loss * n_testing_examples_seen + \
                        minibatch_loss * minibatch.n_examples
            n_testing_examples_seen += minibatch.n_examples
            testing_loss = testing_loss / n_testing_examples_seen

            # store residuals
            residuals = residuals.squeeze(-1).cpu().numpy()
            atom_tally = 0
            for example in minibatch.example_list:
                molecule = molecules_dict[example.ID]
                n_atoms = molecule.n_atoms
                if example.ID not in residuals_by_molecule:
                    residuals_by_molecule[example.ID] = []
                subset = residuals[atom_tally:atom_tally+n_atoms]
                residuals_by_molecule[example.ID].append(subset)
                atom_tally += n_atoms
            assert atom_tally == residuals.shape[0], "Testing atom count mismatch!"

        # interim status update
        print(f"testing  {minibatch_index+1:5d} / {n_minibatches:5d}   minibatch_test_loss = {minibatch_loss:<10.3f}   overall_test_loss = {testing_loss:<10.3f}", end="\r", flush=True)
    #testing_loss /= n_minibatches # ???????????????????

    # reshape residual data
    all_residuals = { element : [] for element in relevant_elements }  # element -> [residuals]
    for ID, results in residuals_by_molecule.items():
        results = np.array(results).T
        molecule = molecules_dict[ID]
        for atomic_index, this_result in enumerate(results):
            element = molecule.atomic_numbers[atomic_index]
            if element not in relevant_elements:
                continue
            site_label = "f{ID}_{number_to_symbol[element]}{atomic_index+1}"
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
        print(f"{number_to_symbol[element]} : {mean_error:.3f} / {RMSE:.3f}    ", end="")
    print(flush=True)

