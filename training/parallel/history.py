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
#import training_config as config
from pipeline import Molecule
from resizable import H5Array as Array
import os
from shutil import copyfile
import h5py
from scipy.optimize import curve_fit

n_to_s = {
1:'H',
6:'C',
7:'N',
8:'O',
16:'S',
9:'F',
17:'Cl'
}


class TrainTestHistory:
    def __init__(self, testing_batches=[], device='cuda', examples_per_epoch=None, relevant_elements=None, 
                 run_name=None, number_to_symbol=None, smoothing_window=10, failed=0,
                 use_tensor_constraint=False, store_residuals=False, wandb_log=None, wandb_interval=1,
                 file=None, save_prefix=None, hdf5=True, use_backup=True, load=True):
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
                
            self.train = TrainingHistory(
                examples_per_epoch, smoothing_window, failed, use_tensor_constraint,
                wandb_log=wandb_log, wandb_interval=wandb_interval,
                file=self.file.create_group('train'), hdf5=hdf5, load=False)
            self.test = TestingHistory(
                examples_per_epoch, testing_batches, relevant_elements, device, failed,
                number_to_symbol=number_to_symbol, store_residuals=store_residuals, wandb_log=wandb_log,
                file=self.file.create_group('test'), hdf5=True, load=False)
            self.failed = failed
            self.file.attrs['failed'] = failed
            self.name = run_name
            self.file.attrs['name'] = run_name
            self.file.attrs['examples_per_epoch'] = examples_per_epoch
            self.file.attrs['use_tensor_constraint'] = use_tensor_constraint

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
            self.save(verbose=False)
                    
            self.failed = self.file.attrs['failed']
            self.name = self.file.attrs['name']
            if 'examples_per_epoch' in self.file.attrs:
                examples_per_epoch = self.file.attrs['examples_per_epoch']
            else:
                self.file.attrs['examples_per_epoch'] = examples_per_epoch
            self.file.attrs['use_tensor_constraint'] = use_tensor_constraint
            self.train = TrainingHistory(
                examples_per_epoch, failed=self.failed,
                use_tensor_constraint=self.file.attrs['use_tensor_constraint'],
                wandb_log=wandb_log, wandb_interval=wandb_interval, 
                file=self.file['train'], hdf5=True, load=True)
            self.test = TestingHistory(
                examples_per_epoch, testing_batches=testing_batches,
                relevant_elements=relevant_elements, device=device,
                failed=self.failed, wandb_log=wandb_log,
                file=self.file['test'], hdf5=True, load=True)
            
            if os.path.isfile(filename + '.bak'):
                os.remove(filename + '.bak')
    
    def save(self, verbose = True):
        assert self.hdf5, "Non-hdf5 histories not currently working."
        if verbose: print("Saving " + ("and backing up " if self.use_backup else "") + "history...")
        self.file.flush()
        if self.use_backup:
            copyfile(self.file.filename, self.file.filename + ".bak")
        
    def close(self, verbose = True):
        if self.file is None:
            return
        self.save()
        filename = self.file.filename
        self.file.close()
        if self.use_backup:
            backup = filename + '.bak'
            os.remove(backup)
        self.file = None
                
            

    def plot(self, figure=None, x_axis='batch_number', y_axis='smoothed_loss', show=True):
        # x_axis = 'time' is also allowed
        if figure is None:
            plt.figure(figsize=(12,8))
        self.train.plot(figure, x_axis, y_axis)
        self.test.plot(figure, x_axis)
        if figure is None:
            plt.legend(loc="best")
            if show:
                plt.show()

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
                 wandb_log=None, wandb_interval=1, file=None, hdf5=True, load=True):
        self.examples_per_epoch = examples_per_epoch
        self.failed = failed
        self.wandb_log = wandb_log
        self.wandb_interval = wandb_interval
        self.file = file
        
        assert hdf5, "Non-hdf5 histories not implemented."
        if not load:
            self.smoothing_window = smoothing_window
            file.attrs['smoothing_window'] = smoothing_window
            
            self.last_wandb = 0
            file.attrs['last_wandb'] = 0

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
            self.last_wandb = file.attrs['last_wandb']
            
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
        
    def log_batch(
            self, batch_time, wait_time, examples_in_batch, atoms_in_batch, elapsed_time, example_in_epoch,
            example, scalar_loss, tensor_loss=None, epoch=None, batch_in_epoch=None, verbose=True):
        self.elapsed_time.append(elapsed_time)

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

        self.examples_in_batch.append(examples_in_batch)
        self.example_in_epoch.append(example_in_epoch)
        self.example.append(example)
        self.atoms_in_batch.append(atoms_in_batch)
        batch_in_epoch = math.ceil(example_in_epoch / self.examples_per_batch)
        self.batch_in_epoch.append(batch_in_epoch)
        
        self.loss.append(scalar_loss)
        window = min(len(self.loss)-1, self.smoothing_window)
        self.smoothed_loss.append(sum(self.loss[-window:]) / window)
        
        if tensor_loss is not None:
            self.tensor_loss.append(tensor_loss)
            window = min(len(self.tensor_loss)-1, self.smoothing_window)
            self.smoothed_tensor_loss.append(sum(self.tensor_loss[-window:]) / window)            
        
        if self.wandb_log is not None and len(self.loss) - 1 - self.last_wandb >= self.wandb_interval:
            self.last_wandb = len(self.loss) - 1
            self.file.attrs['last_wandb'] = self.last_wandb
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
        return max(self.examples_in_batch[-2:])

    def batches_remaining_in_epoch(self, batch=-1):
        return math.ceil((self.examples_per_epoch - self.example_in_epoch[batch]) / self.examples_per_batch)

    def max_batch(self):
        return len(self.loss) - 1
        
    # x_axis = 'time' and y_axis = 'loss' are also allowed
    def plot(self, figure=None, x_axis='example', y_axis='smoothed_loss', show=True):
        x_axis = self.example[1:] if x_axis=='example' else self.elapsed_time[1:]
        y_axis = self.smoothed_loss[1:] if y_axis=='smoothed_loss' else self.loss[1:]
        
        if figure is None:
            plt.figure(figsize=(12,8))
        plt.plot(np.array(x_axis), np.array(y_axis), "ro-", label="train")
        if figure is None:
            plt.legend(loc="best")
            if show:
                plt.show()
            
    def __getstate__(self):
        d = self.__dict__.copy()
        del d['wandb_log']
        return d        


class TestingHistory(BaseHistory):

    # training_window_size: moving average for training_loss over this many minibatches
    def __init__(self, examples_per_epoch, testing_batches, relevant_elements=None, device='cuda',
                 failed=0, number_to_symbol=None, store_residuals=False, wandb_log=None,
                 file=None, hdf5=True, load=True):
        self.examples_per_epoch = examples_per_epoch
        self.testing_batches = testing_batches
        self.device = device
        self.failed = failed
        self.number_to_symbol = number_to_symbol if number_to_symbol else n_to_s
        self.wandb_log = wandb_log
        self.__curve_fit = None

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
            self.mean_error_by_element = Array(file, 'mean_error_by_element', (0,len(relevant_elements)), dtype=float, resizable_cross=True)
            self.RMSE_by_element = Array(file, 'RMSE_by_element', (0,len(relevant_elements)), dtype=float, resizable_cross=True)
        else:
            self.relevant_elements = file.attrs['relevant_elements']
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

            # legacy code, shouldn't be used in newer history files!
            if (self.mean_error_by_element.maxshape[1] is not None) or (self.RMSE_by_element.maxshape[1] is not None):
                mean_error_by_element = np.array(self.mean_error_by_element).copy()
                RMSE_by_element = np.array(self.RMSE_by_element).copy()
                del file['mean_error_by_element']
                del file['RMSE_by_element']
                self.mean_error_by_element = Array(file, 'mean_error_by_element', mean_error_by_element.shape, data=mean_error_by_element, dtype=float, resizable_cross=True)
                self.RMSE_by_element = Array(file, 'RMSE_by_element', RMSE_by_element.shape, data=RMSE_by_element, dtype=float, resizable_cross=True)
                #self.mean_error_by_element[:] = mean_error_by_element
                #self.RMSE_by_element[:] = RMSE_by_element

            # allows the resizing of by-element data arrays if relevant elements change
            if relevant_elements is not None and len(self.relevant_elements) != len(relevant_elements):
                print("relevant_elements has changed!")
                if len(self.relevant_elements) < len(relevant_elements):
                    self.mean_error_by_element.resize_cross(len(relevant_elements))
                    self.RMSE_by_element.resize_cross(len(relevant_elements))
                self.relevant_elements = relevant_elements

        # if we have no testing batches, we won't be running tests, so no need to prep
        if not testing_batches:
            return

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
        mean_error_by_element = torch.tensor([residuals_by_element[e].mean() for e in self.relevant_elements])
        RMSE_by_element = torch.tensor([residuals_by_element[e].square().mean().sqrt() for e in self.relevant_elements])

        time1 = time.time()
        test_time = time1 - time0

        if verbose:
            print(f"  Test loss = {loss:6.3f}   Test time = {test_time:.2f}")
            print(f"  Element   Mean Error    RMSE")
            #print(f"<4> Ee  <7>  012.345 <5> 012.345")
            for i, e in enumerate(self.relevant_elements):
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
                'mean_error_by_element':{
                    int(e):mean_error_by_element[i].item() for i, e in enumerate(self.relevant_elements)},
                'RMSE_by_element':{
                    int(e):RMSE_by_element[i].item() for i, e in enumerate(self.relevant_elements)}
                })
        self.__curve_fit = None
                
    def smoothed_loss(self, i = -1, window=5):
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
        
    def relative_loss(self, other, x, coord='time', window=5):
        if self.failed or len(self.loss) == 0:
            return float("nan")
        last_x = min(self.coord(coord)[-1], x)
        last_i = 0
        weights = 0
    
    def plot(self, figure=None, x_axis='example',
             curve_fit=False, asymptote=False,
             color='b', fit_color='g', asymptote_color='r', show=True, **kwargs): # x_axis = 'time' is also allowed
        #x_axis = self.example if x_axis=='example' else self.elapsed_time
        x_axis = self.coord(x_axis)
        if figure is None:
            plt.figure(figsize=(12,8))
        plt.plot(np.array(x_axis), np.array(self.loss), f'{color}o-', label="test", **kwargs)
        if curve_fit:
            self.plot_curve_fit(color=fit_color, **kwargs)
        if asymptote:
            self.plot_asymptote(color=asymptote_color, **kwargs)
        if figure is None:
            plt.legend(loc="best")
            if show:
                plt.show()
                
    def plot_curve_fit(self, coord='example', color='g', **kwargs):
        popt, _ = self.curve_fit(coord=coord)
        fn = lambda x : exp_fn(x, *popt)
        x_axis = self.coord(coord)
        
        plt.plot(x_axis, fn(x_axis), f'{color}:', **kwargs)

    def plot_asymptote(self, color='r', coord='example', show_error=True, n_sigma=3, **kwargs):
        popt, pcov = self.curve_fit(coord=coord)
        perr = np.sqrt(np.diag(pcov))
        x_axis = self.coord(coord)
        
        plt.plot(x_axis, np.full(len(x_axis), popt[2]), f'{color}', **kwargs)
        if show_error:
            plt.plot(x_axis, np.full(len(x_axis), popt[2] + n_sigma * perr[2]), f'{color}:', **kwargs)
            plt.plot(x_axis, np.full(len(x_axis), popt[2] - n_sigma * perr[2]), f'{color}:', **kwargs)
        
    def curve_fit(self, coord='example', subset=None, **kwargs):
        if self.__curve_fit is None or subset is not None:
            x = self.coord(coord)
            if subset is not None:
                x = x[subset]    
            limit0 = self.smoothed_loss()
            x2 = x[-1]
            x1 = x2 / 2
            y2 = self.loss_interpolate(x2, coord) - limit0
            y1 = self.loss_interpolate(x1, coord) - limit0
            speed0 = 2 * ( np.log(y2) - np.log(y1) ) / x2
            a0 = (y2 - y1) / (np.exp(speed0 * x2) - np.exp(speed0 * x1))
            
            if subset is not None:
                return curve_fit(exp_fn, x, self.loss[subset], (a0, speed0, limit0), **kwargs)
            self.__curve_fit = curve_fit(exp_fn, x, self.loss, (a0, speed0, limit0), **kwargs)
        return self.__curve_fit
        
    def asymptote(self, coord='example'):
        return self.curve_fit()[0][2], np.sqrt(self.curve_fit()[1][2,2])
        
    def show_fit_evolution(self, coord='example', pause = 4., n_sigma=3):
        x_axis = self.coord(coord)
        plt.figure(figsize=(12,8))
        plt.plot(np.array(x_axis), np.array(self.loss), 'k-', alpha=.5)
        #limit = (np.array([.0625,.125,.25,.5,1]) * len(x_axis)).astype(int)
        limit = (np.array([1]) * len(x_axis)).astype(int)
        plt.show(block = False)
        ax = plt.gca()
        
        for l in limit:
            print(f"Fitting range(0,{x_axis[l-1]})...")
            popt, pcov = self.curve_fit(coord=coord, subset=slice(l))#, method='dogbox')
            perr = np.sqrt(np.diag(pcov))
            print(popt)
            print(perr)
            
            fn = lambda x : exp_fn(x, *popt)        
            curve = plt.plot(x_axis, fn(x_axis), 'g--')
            
            bd = plt.axvline(x=x_axis[l-1], c='g', ls=':')
            
            #ac = plt.axhline(y=popt[2], c='r', ls='--')
            #a0 = plt.axhline(y=popt[2] - n_sigma * perr[2], c='r', ls=':')
            #a1 = plt.axhline(y=popt[2] + n_sigma * perr[2], c='r', ls=':')
            
            ax.relim()
            ax.autoscale_view()
            
            plt.show(block = False)
            plt.pause(pause)
            
            for c in curve:
                c.remove()
            bd.remove()
            #ac.remove()
            #a0.remove()
            #a1.remove()
            
        

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['testing_batches']
        del d['wandb_log']
        return d
        

#def exp_fn(x, a, speed, limit):
#    return a * np.exp(speed * x) + limit
def exp_fn(x, a, speed, limit):
    return np.exp(a * np.exp(speed * x) + limit)
#def exp_fn(x, a, speed, limit, k, w):
#    return a * np.exp(speed * x) + limit + k / (x + w)
#def exp_fn(x, k, w, limit):
#    return limit + k / (x + w)
        

# if function is called
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        exit()
    input("Exploring the evolution of exp fit over time...")
    history = TrainTestHistory(file=sys.argv[1])
    history.test.show_fit_evolution()
    input("Press Enter to continue...")


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

