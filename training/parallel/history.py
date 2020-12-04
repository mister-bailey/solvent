from stopwatch import Stopwatch
from collections import deque
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import math
import time
import bisect
from datetime import timedelta
from training_utils import loss_function
import training_config as config
from pipeline import Molecule
from resizable import H5Array as Array
import os
from shutil import copyfile
import h5py


class TrainTestHistory:
    def __init__(self, testing_batches=[], device='cuda', examples_per_epoch=None, relevant_elements=None, 
                 run_name=None, number_to_symbol=None, smoothing_window=10, failed=0,
                 use_tensor_constraint=False, store_residuals=False, sparse_logging=True,
                 wandb_log=None, file=None, save_prefix=None, hdf5=True, use_backup=True, load=True):
        assert hdf5, "Non-hdf5 histories not currently working."
        self.hdf5 = hdf5
        self.use_backup = use_backup

        self.file = file
        if file is None and save_prefix is not None:
            self.file = f"{save_prefix}-history.torch"
        if isinstance(self.file, str) and not os.path.isfile(self.file) and not os.path.isfile(self.file + '.bak'):
            load = False

        if not load:
            # create new history file
            if not isinstance(self.file, h5py.File):
                self.file = h5py.File(self.file, 'w')
            backup = self.file.filename + '.bak'
            if os.path.isfile(backup):
                os.remove(backup)
                
            if sparse_logging:
                self.train = SparseTrainingHistory(
                    examples_per_epoch, smoothing_window, failed, use_tensor_constraint,
                    wandb_log=wandb_log, file=self.file, hdf5=hdf5, load=False)
            else:
                raise ValueError("Currently only SparseTrainingHistory is working.")
                self.train = TrainingHistory(examples_per_epoch, smoothing_window, failed, use_tensor_constraint, wandb_log)
            self.test = TestingHistory(examples_per_epoch, testing_batches, relevant_elements, device, failed,
                                    number_to_symbol=number_to_symbol, store_residuals=store_residuals, wandb_log=wandb_log)
            self.failed = failed
            self.file.attrs['failed'] = failed
            self.name = run_name
            self.file.attrs['name'] = run_name

        else:
            # load history from file
            if not isinstance(self.file, h5py.File):
                self.file = h5py.File(self.file, 'a')
            filename = self.file.filename
            if os.path.isfile(filename + '.bak'):
                print("History file failed to close properly last time!")
                if input("Restore from backup? (y/n) ").strip().lower() == 'y':
                    self.file.close()
                    os.remove(filename)
                    os.rename(filename + '.bak', filename)
                    self.file = h5py.File(filename, 'a')
                    
            self.failed = self.file.attrs['failed']
            self.name = self.file.attrs['name']
            self.examples_per_epoch = self.file.attrs['examples_per_epoch']
            self.sparse_logging = self.file.attrs['sparse_logging']
            if self.sparse_logging:
                self.train = SparseTrainingHistory(
                    self.examples_per_epoch, failed=self.failed,
                    use_tensor_constraint=self.file.attrs['use_tensor_constraint'],
                    wandb_log=wandb_log, file=self.file['train'], hdf5=True, load=True)
            else:
                raise ValueError("Currently only SparseTrainingHistory is fully implemented.")
            self.test = TestingHistory(
                self.examples_per_epoch, testing_batches=testing_batches, device=device, failed=self.failed,
                wandb_log=wandb_log, file=self.file['test'], hdf5=True, load=True)
            
            if os.path.isfile(filename + '.bak'):
                os.remove(filename + '.bak')
    
    def save(self, verbose = True):
        assert self.hdf5, "Non-hdf5 histories not currently working."
        print("Saving " + ("and backing up " if self.use_backup else "") + "history...")
        self.file.flush()
        if self.use_backup:
            copyfile(self.file.filename, self.file.filename + ".bak")
        
    def close(self, verbose = True):
        self.save()
        self.file.close()
        if self.use_backup:
            backup = self.file.filename + '.bak'
            os.remove(backup)
                
            

    def plot(self, figure=None, x_axis='batch_number', y_axis='smoothed_loss'): # x_axis = 'time' is also allowed
        if figure is None:
            plt.figure(figsize=(12,8))
        self.train.plot(figure, x_axis, y_axis)
        self.test.plot(figure, x_axis)
        if figure is None:
            plt.legend(loc="best")

    def log_batch(self, *args, **kwargs):
        self.train.log_batch(*args, **kwargs)

    def elapsed_time(self):
        return self.train.elapsed_time[-1]

    def run_test(self, model, batch_number=None, epoch=None, batch_in_epoch=None, example_in_epoch=None, example=None, elapsed_time=None, *args, **kwargs):
        if batch_number is None: batch_number = len(self.train.loss)
        if epoch is None: epoch = self.train.epoch[-1]
        if batch_in_epoch is None: batch_in_epoch = self.train.batch_in_epoch[-1]
        if example_in_epoch is None: example_in_epoch = self.train.example_in_epoch[-1]
        if example is None: example = self.train.example[-1]
        if elapsed_time is None: elapsed_time = self.train.elapsed_time[-1]
        self.test.run_test(model, batch_number, epoch, batch_in_epoch, example_in_epoch, example, elapsed_time, *args, **kwargs)

class BaseHistory:

    # methods for finding comparison points

    def max_epoch_batch(self):
        return self.epoch[-1], self.batch[-1]

    def max_batch(self):
        raise Exception("This should be overloaded!")

    def max_epoch_example(self):
        return self.epoch[-1], self.example_in_epoch[-1]

    def max_example(self):
        return self.example[-1]

    def max_time(self):
        return self.elapsed_time[-1]
        

class TrainingHistory(BaseHistory):
    def __init__(self, examples_per_epoch, smoothing_window=10, failed=0, use_tensor_constraint=False,
                 wandb_log=None, file=None, hdf5=True, load=True):
        self.examples_per_epoch = examples_per_epoch
        self.failed = failed
        self.wandb_log = wandb_log
        self.file = file
        
        if hdf5:
            if not load:
                self.smoothing_window = smoothing_window
                file.attrs['smoothing_window'] = smoothing_window

                # initialize the lists we will be accumulating
                # these lists correspond to each other
                # one entry per batch
                # batch 0 is a dummy batch
                self.epoch=Array(file, 'epoch', [0])

                # batch n covers example[example_number[n-1]:example_number[n]]
                self.example_in_epoch=Array(file, 'example_in_epoch', [examples_per_epoch])
                self.example=Array(file, 'example', [0])
                
                self.examples_in_batch=Array(file, 'examples_in_batch', [0])

                self.batch_in_epoch=Array(file, 'batch_in_epoch', [0])
                self.elapsed_time=Array(file, 'elapsed_time', [0.0])

                self.atoms_in_batch=Array(file, 'atoms_in_batch', [0])

                self.loss=Array(file, 'loss', [float("inf")])
                self.smoothed_loss=Array(file, 'smoothed_loss', [float("inf")])
                
                if use_tensor_constraint:
                    self.tensor_loss=Array(file, 'tensor_loss', [float("inf")])
                    self.smoothed_tensor_loss=Array(file, 'smoothed_tensor_loss', [float("inf")])

                # this list has one entry per epoch
                # epoch e starts at index epoch_start[e]
                self.epoch_start=Array(file, 'epoch_start', [0]) # includes a dummy epoch 0 starting at batch 0
            else:
                self.smoothing_window = file.attrs['smoothing_window']
                self.epoch=Array(file, 'epoch')
                self.example_in_epoch=Array(file, 'example_in_epoch')
                self.example=Array(file, 'example')               
                self.examples_in_batch=Array(file, 'examples_in_batch')
                self.batch_in_epoch=Array(file, 'batch_in_epoch')
                self.elapsed_time=Array(file, 'elapsed_time')
                self.atoms_in_batch=Array(file, 'atoms_in_batch')
                self.loss=Array(file, 'loss')
                self.smoothed_loss=Array(file, 'smoothed_loss')              
                if use_tensor_constraint:
                    self.tensor_loss=Array(file, 'tensor_loss')
                    self.smoothed_tensor_loss=Array(file, 'smoothed_tensor_loss')
                self.epoch_start=Array(file, 'epoch_start') 
        else:
            raise ValueError("Non-hdf5 histories not currently working.")



    def log_batch(self, batch_time, wait_time, examples_in_batch, atoms_in_batch, scalar_loss,
                  tensor_loss=None, epoch=None, batch_in_epoch=None, verbose=True):

        # if you don't provide batch numbers or epoch numbers, we will make reasonable assumptions:
        epoch = self.current_epoch() if epoch is None else epoch
        if epoch > self.epoch[-1]:
            self.epoch_start.append(len(self.example))
        self.epoch.append(epoch)

        batch_in_epoch = self.next_batch_in_epoch() if batch_in_epoch is None else batch_in_epoch
        self.batch_in_epoch.append(batch_in_epoch)

        self.examples_in_batch.append(examples_in_batch)
        self.example_in_epoch.append((self.example_in_epoch[-1] + examples_in_batch - 1) % self.examples_per_epoch + 1)
        self.example.append(self.example[-1] + examples_in_batch)

        self.elapsed_time.append(self.elapsed_time[-1] + batch_time)
        self.atoms_in_batch.append(atoms_in_batch)
        
        self.loss.append(scalar_loss)
        window = min(len(self.loss)-1, self.smoothing_window)
        self.smoothed_loss.append(sum(self.loss[-window:]) / window)

        if tensor_loss is not None:
            self.tensor_loss.append(tensor_loss)
            window = min(len(self.tensor_loss)-1, self.smoothing_window)
            self.smoothed_tensor_loss.append(sum(self.tensor_loss[-window:]) / window)

        if verbose:
            print(f"{self.epoch[-1]} : {self.batch_in_epoch[-1]} / {self.batch_in_epoch[-1] + self.batches_remaining_in_epoch()}  " +
                  ("loss =" if tensor_loss is None else "scalar_loss =") + f"{self.smoothed_loss[-1]:8.3f}  " +
                  ("" if tensor_loss is None else f"tensor_loss ={self.smoothed_tensor_loss[-1]:8.3f}") +
                  f"  t_train = {batch_time:.2f} s  " +
                  (f"t_wait = {wait_time:.2f} s  " if tensor_loss is None else "") +
                  f"t = {str(timedelta(seconds=self.elapsed_time[-1]))[:-5]}   ",
                   end="\r", flush=True)

    def num_batches(self):
        return len(self.epoch) - 1

    # number of epochs we have seen
    def num_epochs(self):
        return len(self.epoch_start) - 1

    def current_epoch(self, batch=-1):
        """
        epoch of the next incoming batch
        """
        return self.epoch[batch] + self.example_in_epoch[batch] // self.examples_per_epoch
        
    def next_example_in_epoch(self, batch=-1):
        """
        example in epoch that would start next batch
        """
        return self.example_in_epoch[-1] % self.examples_per_epoch

    def next_batch_in_epoch(self, batch=-1):
        """
        batch number that would follow given batch
        wraps around to 1 if epoch ends
        """
        return 1 if self.example_in_epoch[batch] >= self.examples_per_epoch else self.batch_in_epoch[batch] + 1
        
    @property
    def examples_per_batch(self):
        return self.examples_in_batch[-1]

    def batches_remaining_in_epoch(self, batch=-1):
        return math.ceil((self.examples_per_epoch - self.example_in_epoch[batch]) / self.examples_per_batch)

    def max_batch(self):
        return len(self.loss) - 1
        
    # x_axis = 'time' and y_axis = 'loss' are also allowed
    def plot(self, figure=None, x_axis='example', y_axis='smoothed_loss'):
        x_axis = self.example[1:] if x_axis=='example' else self.elapsed_time[1:]
        y_axis = self.smoothed_loss[1:] if y_axis=='smoothed_loss' else self.loss[1:]
        
        if figure is None:
            plt.figure(figsize=(12,8))
        plt.plot(np.array(x_axis), np.array(y_axis), "ro-", label="train")
        if figure is None:
            plt.legend(loc="best")
            
    def __getstate__(self):
        d = self.__dict__.copy()
        del d['wandb_log']
        return d



class SparseTrainingHistory(TrainingHistory):
    """
    A reimagining of the training history class which gets
    sparse updates (i.e., nonconsecutive batches). Quantities
    can't be computed by aggregation any more.
    
    Intended for distributed computing, when we don't want
    all the processes vying to log at the same time. 
    GPU 0 will own and have exclusive access to this history.
    
    """    
    
    def __init__(self, examples_per_epoch, smoothing_window=10, failed=0, use_tensor_constraint=False,
                 wandb_log=None, file=None, hdf5=True, load=True):
        super().__init__(examples_per_epoch, smoothing_window, failed, use_tensor_constraint, wandb_log,
                         file=None, hdf5=True, load=True)
        

    def log_batch(self, batch_time, wait_time, examples_in_batch, atoms_in_batch, example_in_epoch,
                  example, scalar_loss, tensor_loss=None, epoch=None, batch_in_epoch=None, verbose=True):
        self.elapsed_time.append(self.elapsed_time[-1] + batch_time) # worry about this with sparse logging

        if example is None:
            example = (epoch-1) * self.examples_per_epoch + example_in_epoch
        elif example_in_epoch is None:
            example_in_epoch = (example - 1) % self.examples_per_epoch + 1

        epoch = self.epoch[-1] if epoch is None else epoch
        if example_in_epoch > self.examples_per_epoch:
            epoch += 1
        if epoch > self.epoch[-1]:
            self.epoch_start.append(len(self.example))
        self.epoch.append(epoch)

        batch_in_epoch = math.ceil(example_in_epoch / self.examples_per_batch)
        self.batch_in_epoch.append(batch_in_epoch)
        self.examples_in_batch.append(examples_in_batch)
        self.example_in_epoch.append(example_in_epoch)
        self.example.append(example)
        self.atoms_in_batch.append(atoms_in_batch)
        
        self.loss.append(scalar_loss)
        window = min(len(self.loss)-1, self.smoothing_window)
        self.smoothed_loss.append(sum(self.loss[-window:]) / window)
        
        if tensor_loss is not None:
            self.tensor_loss.append(tensor_loss)
            window = min(len(self.tensor_loss)-1, self.smoothing_window)
            self.smoothed_tensor_loss.append(sum(self.tensor_loss[-window:]) / window)            
        
        if self.wandb_log is not None:
            log_dict = {
                'elapsed_time':self.elapsed_time[-1],
                'epoch':epoch,
                'batch_in_epoch':batch_in_epoch,
                'example_in_epoch':example_in_epoch,
                'example':example,
                'examples_in_batch':examples_in_batch,
                'atoms_in_batch':atoms_in_batch,
                'scalar_loss':scalar_loss,
                'smoothed_scalar_loss':self.smoothed_loss[-1]
            }
            if tensor_loss is not None:
                log_dict['tensor_loss'] = tensor_loss
                log_dict['smoothed_tensor_loss'] = self.smoothed_tensor_loss[-1]
            self.wandb_log(log_dict)

        if verbose:
            print(f"{self.epoch[-1]} : {self.batch_in_epoch[-1]} / {self.batch_in_epoch[-1] + self.batches_remaining_in_epoch()}  " +
                  ("loss =" if tensor_loss is None else "scalar_loss =") + f"{self.smoothed_loss[-1]:8.3f}  " +
                  ("" if tensor_loss is None else f"tensor_loss ={self.smoothed_tensor_loss[-1]:8.3f}") +
                  f"  t_train = {batch_time:.2f} s  " +
                  (f"t_wait = {wait_time:.2f} s  " if tensor_loss is None else "") +
                  f"t = {str(timedelta(seconds=self.elapsed_time[-1]))[:-5]}   ",
                   end="\r", flush=True)

    def num_batches(self):
        return len(self.epoch) - 1

    # number of epochs we have seen, inclusive of partial epochs
    def num_epochs(self):
        return self.epoch[-1]

    def max_batch(self):
        return len(self.loss) - 1


class TestingHistory(BaseHistory):

    # training_window_size: moving average for training_loss over this many minibatches
    def __init__(self, examples_per_epoch, testing_batches, relevant_elements=None, device='cuda',
                 failed=0, number_to_symbol=None, store_residuals=False, wandb_log=None,
                 file=None, hdf5=True, load=True):
        self.examples_per_epoch = examples_per_epoch
        self.testing_batches = testing_batches
        self.device = device
        self.failed = failed
        self.number_to_symbol = number_to_symbol if number_to_symbol else config.number_to_symbol
        self.wandb_log = wandb_log

        assert hdf5, "Non-hdf5 histories not working right now."
        if not load:
            self.relevant_elements = relevant_elements
            file.attrs['relevant_elements'] = relevant_elements
            self.store_residuals = store_residuals
            file.attrs['store_residuals'] = store_residuals

            # initialize the lists we will be accumulating
            self.batch_number = Array(file, 'batch_number', dtype=int)
            self.epoch = Array(file, 'epoch', dtype=int)
            self.batch_in_epoch = Array(file, 'batch_in_epoch', dtype=int)
            self.example_in_epoch = Array(file, 'example_in_epoch', dtype=int)
            self.example = Array(file, 'example', dtype=int)
            self.elapsed_time = Array(file, 'elapsed_time', dtype=float)
            self.loss = Array(file, 'loss', dtype=float)
            self.mean_error_by_element = Array(file, 'mean_error_by_element', (0,len(relevant_elements)), dtype=float)
            self.RMSE_by_element = Array(file, 'RMSE_by_element', (0,len(relevant_elements)), dtype=float)
        else:
            self.store_residuals = file.attrs['relevant_elements']
            self.store_residuals = file.attrs['store_residuals']

            self.batch_number = Array(file, 'batch_number')
            self.epoch = Array(file, 'epoch')
            self.batch_in_epoch = Array(file, 'batch_in_epoch')
            self.example_in_epoch = Array(file, 'example_in_epoch')
            self.example = Array(file, 'example')
            self.elapsed_time = Array(file, 'elapsed_time')
            self.loss = Array(file, 'loss')
            self.mean_error_by_element = Array(file, 'mean_error_by_element')
            self.RMSE_by_element = Array(file, 'RMSE_by_element')
            


        # for each relevant element, gives you a list of atom indices in the testing set
        atom_indices = {e:[] for e in self.relevant_elements}
        atom_index = 0
        for batch in testing_batches:
            atomic_numbers = Molecule.get_atomic_numbers(batch.x)
            for e in atomic_numbers:
                e = e.item()
                if e in self.relevant_elements:
                    atom_indices[e].append(atom_index)
                atom_index += 1
        self.atom_indices = {e:np.array(ai) for e,ai in atom_indices.items()}

        # precompute weight per testing batch and total testing weight
        self.batch_weights = torch.tensor([torch.sum(batch.weights) for batch in testing_batches])
        self.total_weight = sum(self.batch_weights)

    def max_batch(self):
        return self.batch_number[-1]
        
    def plot(self, figure=None, x_axis='example'): # x_axis = 'time' is also allowed
        x_axis = self.example if x_axis=='example' else self.elapsed_time
        if figure is None:
            plt.figure(figsize=(12,8))
        plt.plot(np.array(x_axis), np.array(self.loss), "bo-", label="test")
        if figure is None:
            plt.legend(loc="best")

    def run_test(self, model, batch_number, epoch, batch_in_epoch, example_in_epoch, example, elapsed_time, verbose=True, log=True):
        if verbose: print("")
        use_tensor_constraint = (self.testing_batches[0].y.shape[-1] == 10)

        time0 = time.time()
        losses = []
        residual_chunks = []
        model.eval() # don't compute running means
        with torch.no_grad(): # don't compute gradients
            for i, batch in enumerate(self.testing_batches):
                if verbose: print(f"Testing batches...  {i:3} / {len(self.testing_batches)}   ", end="\r", flush=True)
                batch.to(self.device)
                scalar_loss, *_, chunk = loss_function(model(batch.x, batch.edge_index, batch.edge_attr), batch, use_tensor_constraint=use_tensor_constraint)
                if use_tensor_constraint:
                    chunk = chunk[...,0] # Keep only scalar part of residuals
                losses.append(scalar_loss.detach())
                residual_chunks.append(chunk.detach())

        if verbose: print("Collating batch results...", end="\r", flush=True)
        # Using average loss over batches, rather than weighted average, to better mirror
        # running average of testing loss:
        #loss = (torch.dot(torch.tensor(losses), self.batch_weights) / self.total_weight).sqrt()
        loss = torch.tensor(losses).sqrt().mean()
        residuals = torch.cat(residual_chunks)

        if verbose: print("Calculating stats by element...", end="\r", flush=True)
        residuals_by_element = {e:residuals[self.atom_indices[e]] for e in self.relevant_elements}

        # compute mean errors and RMSEs
        mean_error_by_element = [residuals_by_element[e].mean() for e in self.relevant_elements]
        RMSE_by_element = [residuals_by_element[e].square().mean().sqrt() for e in self.relevant_elements]

        time1 = time.time()
        test_time = time1 - time0

        if verbose:
            print(f"  Test loss = {loss:6.3f}   Test time = {test_time:.2f}")
            print(f"  Element   Mean Error    RMSE")
            #print(f"<4> Ee  <7>  012.345 <5> 012.345")
            for i, e in self.relevant_elements:
                print(f"    {self.number_to_symbol[e].rjust(2)}       {mean_error_by_element[i]:3.3f}     {RMSE_by_element[i]:3.3f}")

        if log:
            if example is None:
                example = (epoch-1) * self.examples_per_epoch + example_in_epoch
            elif example_in_epoch is None:
                example_in_epoch = example % self.examples_per_epoch
            self.log_test(batch_number, epoch, batch_in_epoch, example_in_epoch, example, elapsed_time,
                    loss, mean_error_by_element, RMSE_by_element, residuals_by_element)

    def log_test(self, batch_number, epoch, batch_in_epoch, example_in_epoch, example, elapsed_time,
            loss, mean_error_by_element, RMSE_by_element, residuals_by_element=None):
        self.batch_number.append(batch_number)
        self.epoch.append(epoch)
        self.batch_in_epoch.append(batch_in_epoch)
        self.example_in_epoch.append(example_in_epoch)
        self.example.append(example)
        self.elapsed_time.append(elapsed_time)
        self.loss.append(loss)
        self.mean_error_by_element.append(mean_error_by_element)
        self.RMSE_by_element.append(RMSE_by_element)
        if self.store_residuals:
            self.residuals_by_element = residuals_by_element
            
        if self.wandb_log is not None:
            self.wandb_log({
                'batch_number':batch_number,
                'epoch':epoch,
                'batch_in_epoch':batch_in_epoch,
                'example_in_epoch':example_in_epoch,
                'example':example,
                'elapsed_time':elapsed_time,
                'test_loss':loss,
                'mean_error_by_element':mean_error_by_element,
                'RMSE_by_element':RMSE_by_element})
                
    def smoothed_loss(self, i, window=5):
        i = len(self.loss) - i if i < 0 else i
        window = min(window, i)
        return sum(self.loss[i-window:i+1]) / window
        
    def coord(self, coord):
        if coord == 'time':
            return self.elapsed_time
        elif coord == 'example':
            return self.example
        else:
            raise ValueError("Argument 'coord' should be 'time' or 'example'")
                
    def loss_interpolate(self, x, coord='time', window=5):
        x_array = self.coord(coord)
        i = bisect.bisect_right(x_array, x) 
        if i > 0 and x_array[i-1] == x:
            return self.smoothed_loss(i, window)
        if i==0 or i==len(x_array):
            raise ValueError(coord.capitalize() + " coordinate is out of bounds.")
        x1 = x_array[i-1]
        x2 = x_array[i]
        s = (x2 - x) / (x2 - x1)
        return s * self.smoothed_loss(i-1, window) + (1-s) * self.smoothed_loss(i, window)

    def loss_extrapolate(self, x, histories, coord='time', window=5):
        if self.failed or len(self.loss) == 0:
            return float("nan")
        last_x = self.coord(coord)[-1]
        assert x > last_x, f"Tried to extrapolate, but given {coord} is within bounds."
        last_loss = self.smoothed_loss(-1, window)
        losses = []
        for h in histories:
            if h.coord(coord) >= x and h is not self:
                losses.append(h.loss_interpolate(x, coord, window) * last_loss / h.loss_interpolate(last_x, coord, window))
        assert losses, "Extrapolating beyond farthest history."
        return sum(losses) / len(losses)

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['testing_batches']
        del d['wandb_log']
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

