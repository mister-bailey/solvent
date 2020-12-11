from collections import deque
from functools import partial
import gc

import numpy as np
import torch
import os

from variable_networks import VariableParityNetwork
from sparse_kernel_conv import SparseKernelConv, DummyConvolution
from training_utils import train_batch
from pipeline import Pipeline
from training_config import Config
from exploration import random_parameters_and_seed, generate_parameters



class TrainableSampleGenerator:
    
    def __init__(self, filename=None, configs=['exploration.ini'], num_batches=2, stub=False):
        self.filename=filename
        self.config_files=configs
        self.__config = None
        self.config = Config(*self.config_files, track_sources=True, use_command_args=False)
        
        self.seed_length = len(random_parameters_and_seed()[1])
        self.seeds = np.zeros((0,self.seed_length))
        self.passed = np.zeros(0, dtype=bool)
        self.failure_model = None
        self.num_batches = num_batches

        self.stub=stub
        if stub:
            return
        
        num_examples = self.config.training.batch_size * num_batches
        print("Initializing data pipeline... ", end='')
        pipeline = Pipeline(self.config)
        pipeline.set_indices(torch.load(self.config.data.test_train_shuffle)[:num_examples])
        pipeline.start_reading(num_examples)
        print("Done.")
        print("Reading test batches... ", end='')
        self.batches = []
        for _ in range(num_batches):
            self.batches.append(pipeline.get_batch())
        print("Done.")
        print("Closing pipeline... ", end='')
        pipeline.close()
        del pipeline
        print("Done.")
                
    def sample(self, num_samples=None, num_passes=None, increment=100, verbose=True, verbose_test=False, save=True):
        current_samples = len(self.seeds)
        current_passes = self.num_passes
        if increment and num_samples is not None:
            for n in list(range(current_samples, num_samples, increment))[1:] + [num_samples]:
                self.sample(num_samples=n, verbose_test=verbose_test, save=save)
            return
        elif increment and num_passes is not None:
            for n in list(range(current_passes, num_passes, increment))[1:] + [num_passes]:
                self.sample(num_passes=n, verbose_test=verbose_test, save=save)
            return
        
        if num_samples is None:
            size = current_samples + num_passes - current_passes
        else:
            size = num_samples
        self.seeds.resize(size, self.seed_length)
        self.passed.resize(size)
        
        distribution = partial(np.random.random_sample, self.seed_length)
        
        while True:
            for i in range(current_samples, size):
                seed = distribution()
                self.seeds[i] = seed
                failed = self.test_sample(seed, verbose=verbose_test)
                self.passed[i] = not failed
                if not failed:
                    current_passes += 1
                    if verbose: print(f"*********** {current_passes:5d}  ***********")
                    if num_passes is not None and current_passes >= num_passes:
                        if save: self.save()
                        return
                else:
                    if verbose: print(f"· · · · · ·        · · · · · ·")
                current_samples += 1
                if num_samples is not None and current_samples >= num_samples:
                    if save: self.save()
                    return
            size = current_samples + num_passes - current_passes
            self.seeds.resize(size, self.seed_length)
            self.passed.resize(size)
    
    @property         
    def passes(self):
        return self.seeds[self.passed]
    
    @property
    def num_passes(self):
        return np.sum(self.passed)
    
    @staticmethod
    def load(file):
        if os.path.isfile(file):
            return torch.load(file)
        else:
            return None
    
    def save(self, filename=None, verbose=True):
        if filename is None:
            filename = self.filename
        if filename is None:
            return
        if verbose: print("Saving samples... ", end='')
        torch.save(self, filename)
        if verbose: print("Done.")
    
    def test_sample(self, seed, verbose=True):
        failed=0
        settings = generate_parameters(seed)
        if self.stub:
            return self.test_stub(settings)
        if verbose: print(f"Testing sample: {settings['model']}")
        
        if verbose: print("Building model... ", end='')
        model_kwargs = {
        'kernel': SparseKernelConv,
        'convolution': DummyConvolution,
        'batch_norm': True,
        }
        model_kwargs.update(self.config.model.kwargs)
        model_kwargs.update(settings['model'])
        try:
            model = VariableParityNetwork(**model_kwargs)
            model.to(self.config.device)
        except ValueError as e:
            if verbose: print(f"\n!!!! Failed: {e}")
            return 4
        
        if verbose: print("Building optimizer... ", end='')
        optimizer = torch.optim.Adam(model.parameters(), self.config.training.learning_rate)
        
        b=0
        data_queue = deque(maxlen=self.config.data.batch_preload)
        losses = None
        
        if verbose: print("Training Batch", end='')
        try:
            for i in range(self.num_batches):
                while len(data_queue) < self.config.data.batch_preload and b < len(self.batches):
                    data_queue.appendleft(self.batches[b].to(self.config.device))
                    b += 1
                
                if verbose: print(f" {i+1}", end='')
                losses = train_batch(data_queue, model, optimizer)
                                    
        except RuntimeError as e:
            if 'CUDA' in e.args[0]:
                failed = 3
            else:
                raise
        
        del losses
        del model
        del optimizer
        del data_queue
        gc.collect()
        #torch.cuda.empty_cache()
        
        if failed:
            if verbose: print("\n!!!! Failed !!!!")
        else:
            if verbose: print("\n**** Passed ****")
        return failed
    
    def test_stub(self, settings):
        ls = [list(range(lmax + 1)) for lmax in settings['model']['lmaxes']]
        muls = settings['model']['muls']
        lmuls = [list(zip(li, mi)) for li,mi in list(zip(ls,muls))]
        
        memory = sum(sum(l**2 * m for l,m in layer) for layer in lmuls)
        
        if memory > 100:
            return 3
        else:
            return 0
            
    def remove(self, seed):
        indices = np.argwhere(self.seeds==seed)
        for index in np.flip(indices, axis=0):
            self.seeds = np.delete(self.seeds, index, axis=0)
            self.passed = np.delete(self.passed, index, axis=0)
        

if __name__ == '__main__':
    sampler = TrainableSampleGenerator(stub=True)
    sampler.sample(num_passes=1000, increment=100, verbose_test=False)
        
        
            
