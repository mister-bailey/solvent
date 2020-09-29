from stopwatch import Stopwatch
from collections import deque
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import math
import time
from datetime import timedelta
from training_utils import loss_function
from training_config import Config


class TrainTestHistory:
    def __init__(self, examples_per_epoch, save_prefix, testing_batches, relevant_elements,
                 molecule_dict, device, number_to_symbol=None, smoothing_window=10, store_residuals=False):
        self.train = TrainingHistory(examples_per_epoch, smoothing_window)
        self.test = TestingHistory(testing_batches, relevant_elements, molecule_dict, device,
                                  number_to_symbol=number_to_symbol, store_residuals=store_residuals)
        self.save_prefix = save_prefix
    
    # save raw data
    def save(self, file=None):
        if file is None:
            file = f"{self.save_prefix}-history.torch"
        print("Saving train/test history... ", end="", flush=True)
        torch.save(self, file)
        print("Done.", end="\r", flush=True)

    @staticmethod
    def load(testing_batches=[], file=None, prefix=None):
        if file is None:
            file = prefix + "-history.torch"
        h = torch.load(file)
        assert isinstance(h, TrainTestHistory), f"File {file} doesn't contain a train/test history!"
        h.test.testing_batches = testing_batches
        return h

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
        if batch_number is None: batch_number = len(self.train.loss)
        if epoch is None: epoch = self.train.epoch[-1]
        if batch_in_epoch is None: batch_in_epoch = self.train.batch_in_epoch[-1]
        if elapsed_time is None: elapsed_time = self.train.elapsed_time[-1]
        self.test.run_test(model, batch_number, epoch, batch_in_epoch, elapsed_time, *args, **kwargs)

    def elapsed_time(self, batch=-1):
        return self.train.elapsed_time[batch]
        

# returns the first non-None argument
# eg., user-specified value could be arg1, default could be arg2
def alt(*args):
    for arg in args:
        if arg is not None:
            return arg
    return None

class TrainingHistory:
    def __init__(self, examples_per_epoch, smoothing_window=10):
        self.examples_per_epoch = examples_per_epoch
        self.smoothing_window = smoothing_window

        # initialize the lists we will be accumulating
        # these lists correspond to each other
        # one entry per batch
        # batch 0 is a dummy batch
        self.epoch=[0]

        # batch n covers example[example_number[n-1]:example_number[n]]
        self.example_number=[examples_per_epoch]

        self.batch_in_epoch=[0]
        self.elapsed_time=[0]

        self.atoms_per_batch=[0]

        self.loss=[None]
        self.smoothed_loss=[None]

        # this list has one entry per epoch
        # epoch e starts at index epoch_start[e]
        self.epoch_start=[0] # includes a dummy epoch 0 starting at batch 0



    def log_batch(self, batch_time, wait_time, examples_per_batch, atoms_per_batch, loss,
                  epoch=None, batch_in_epoch=None, verbose=True):
        self.elapsed_time.append(self.elapsed_time[-1] + batch_time)

        # if you don't provide batch numbers or epoch numbers, we will make reasonable assumptions:
        epoch = self.current_epoch() if epoch is None else epoch
        if epoch > self.epoch[-1]:
            self.epoch_start.append(len(self.epoch))
        self.epoch.append(epoch)

        batch_in_epoch = self.next_batch_in_epoch() if batch_in_epoch is None else batch_in_epoch
        self.batch_in_epoch.append(batch_in_epoch)

        #print(f"example_in_epoch: {self.example_in_epoch()}  ", end='')
        self.example_number.append(self.example_in_epoch() + examples_per_batch)

        self.elapsed_time.append(self.elapsed_time[-1] + batch_time)
        self.atoms_per_batch.append(atoms_per_batch)
        self.loss.append(loss)
        window = min(len(self.loss)-1, self.smoothing_window)
        self.smoothed_loss.append(sum(self.loss[-window:]) / window)
        #print(f"batches_remaining: {self.batches_remaining_in_epoch()}")

        if verbose:
            print(f"{self.epoch[-1]} : {self.batch_in_epoch[-1]} / {self.batch_in_epoch[-1] + self.batches_remaining_in_epoch()}"
                  f"  train_loss = {self.smoothed_loss[-1]:10.3f}  t_train = {batch_time:.2f} s"
                  f"  t_wait = {wait_time:.2f} s  t = {str(timedelta(seconds=self.elapsed_time[-1]))[:-5]}   ",
                   end="\r", flush=True)

    def examples_in_batch(self, batch):
        return self.example_number[batch] - self.example_in_epoch(batch-1)

    def num_batches(self):
        return len(self.epoch) - 1

    # number of epochs we have seen
    def num_epochs(self):
        return len(self.epoch_start) - 1

    def ends_epoch(self, batch):
        """
        True iff batch is last in epoch
        """
        return self.example_number[batch] >= self.examples_per_epoch

    def current_epoch(self, batch=-1):
        """
        epoch of the next incoming batch
        """
        return self.epoch[batch] if not self.ends_epoch(batch) else self.epoch[batch] + 1

    def next_batch_in_epoch(self, batch=-1):
        """
        batch number that would follow given batch
        wraps around to 1 if epoch ends
        """
        return self.batch_in_epoch[batch] + 1 if not self.ends_epoch(batch) else 1

    def batches_remaining_in_epoch(self, batch=-1):
        return math.ceil((self.examples_per_epoch - self.example_number[batch]) / self.examples_in_batch(batch))

    def example_in_epoch(self, batch=-1):
        """
        tells you which example number would start the following batch
        """
        return self.example_number[batch] if not self.ends_epoch(batch) else 0

    def total_examples(self, batch=-1):
        return (self.epoch[batch] - 1) * self.examples_per_epoch + self.example_number[batch]
        
    # x_axis = 'time' and y_axis = 'loss' are also allowed
    def plot(self, figure=None, x_axis='batch_number', y_axis='smoothed_loss'):
        x_axis = range(1, len(self.loss)) if x_axis=='batch number' else self.elapsed_time[1:]
        y_axis = self.smoothed_loss[1:] if y_axis=='smoothed_loss' else self.loss[1:]
        
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
        # Using average loss over batches, rather than weighted average, to better mirror
        # running average of testing loss:
        #loss = (torch.dot(torch.tensor(losses), self.batch_weights) / self.total_weight).sqrt()
        loss = torch.tensor(losses).sqrt().mean()
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
            #print(f"<4> Ee  <7>  012.345 <5> 012.345")
            for e in self.relevant_elements:
                print(f"    {self.number_to_symbol[e].rjust(2)}       {mean_error_by_element[e]:3.3f}     {RMSE_by_element[e]:3.3f}")

        if log:
            self.log_test(batch_number, epoch, batch_in_epoch, elapsed_time,
                    loss, mean_error_by_element, RMSE_by_element, residuals_by_element)

    def log_test(self, batch_number, epoch, batch_in_epoch, elapsed_time,
            loss, mean_error_by_element, RMSE_by_element, residuals_by_element=None):
        self.batch_number.append(batch_number)
        self.epoch.append(epoch)
        self.batch_in_epoch.append(batch_in_epoch)
        self.elapsed_time.append(elapsed_time)
        self.loss.append(loss)
        self.mean_error_by_element.append(mean_error_by_element)
        self.RMSE_by_element.append(RMSE_by_element)
        if self.store_residuals:
            self.residuals_by_element = residuals_by_element

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['testing_batches']
        return d



        


# this function is deprecated. I include it only for reference.

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

