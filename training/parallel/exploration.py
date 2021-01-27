from training_config import Config, secs_to_str, update, immut
from configparser import ConfigParser
import os
import sys
import subprocess
import math
import random
from history import TrainTestHistory
import bayes_search as bayes
from exploration_policy import random_parameters_and_seed, generate_parameters, proceed_with_training, next_training_limit, order_runs as order_runs_policy
from sample_hyperparameters import TrainableSampleGenerator
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

class TrainingRun:
    def __init__(self, parent_dir="runs", identifier=None, config_file=None, settings={},
                 create_file=True, create_identifier=True, parent_configs=['exploration.ini'], save_prefix=None):
        settings = settings.copy() # To avoid accumulating same settings from one run to another!

        if identifier is None:
            if settings != {} and create_identifier:
                identifier = "run-" + hex(hash(immut(settings)))[-8:]
                if os.path.isdir(os.path.join(parent_dir, identifier)):
                    raise Exception(f"Hash collision with existing training run {identifier}")
            else:
                raise Exception("No training run identifier provided!")
        self.identifier = identifier
        self.run_dir = os.path.join(parent_dir, identifier)
        os.makedirs(self.run_dir, exist_ok=True)
        
        if config_file is None:
            config_file = os.path.join(self.run_dir, "training.ini")
        self.config_file = config_file
        
        if os.path.isfile(config_file):
            parser = ConfigParser(allow_no_value=True)
            parser.read([config_file])
            parser.read_dict(settings)
            if "training" in parser:
                #if "run_name" in parser["training"]:
                #    identifier = parser["training"]["run_name"]
                if "save_prefix" in parser["training"]:
                    save_prefix = parser["training"]["save_prefix"]
            del parser
        self.parent_configs = parent_configs
        #self.settings = {k:str(v) for k,v in settings.items()} # may need to allow alternate string conversions

        if save_prefix is None:
            save_prefix = os.path.join(self.run_dir, identifier[:7])
        self.save_prefix = save_prefix

        settings = update(settings, {'training':{'save_prefix':self.save_prefix, 'run_name':self.identifier, 'resume':True}})

        self.set_config(settings)
        self.__history = None
        
    @property
    def history(self):
        if self.__history is None:
            if os.path.isfile(self.save_prefix + "-history.torch"):
                try:
                    self.__history = TrainTestHistory(save_prefix = self.save_prefix, load=True, use_backup=False)
                except:
                    self.__history = None
        return self.__history
        
    def close_history(self):
        if self.__history is not None:
            self.__history.close()
            self.__history = None

    @property
    def config(self):
        if self.__config is None:
            self.__config = Config(*self.parent_configs, self.config_file,
                                   track_sources=True, use_command_args=False)
        return self.__config
        
    @property
    def local_config(self):
        if self.__local_config is None:
            self.__local_config = self.config.file_settings(self.config_file)
        return self.__local_config
    
    def set_config(self, settings):
        parser = ConfigParser(allow_no_value=True)
        parser.read([self.config_file])
        parser.read_dict(settings)
        with open(self.config_file, 'w') as f:
            parser.write(f)
        self.__config = None
        self.__local_config = None
        
    @property
    def epochs(self):
        return self.history.train.current_epoch() - 1
    
    @property
    def time(self):
        return self.history.train.max_time()
        
    @property
    def examples(self):
        return self.history.train.example[-1]
        
            
    # saves the contents of self.__local_config to a config file
    #def save_config(self):
    #    sections = self.local_config.__dict__()
    #    settings = {sec_name:sec.__dict__() for sec_name,sec in sections.items()}
    #    parser = ConfigParser(allow_no_value=True)
    #    parser.read_dict(settings)
        
        
    def execute_run(self, configs=[], stub=False):
        self.close_history()
        print("\n==============================================")
        print(f"Run {self.identifier}:")
        print(f"      run dir: {self.run_dir}")
        print(f"  save_prefix: {self.save_prefix}")
        print(f"  config file: {self.config_file}")
        print("----------------------------------------------\n")
        if stub:
            self.execute_stub()
            return 0
        configs = self.parent_configs + [self.config_file] + configs
        self.last_execution = subprocess.run("python training.py " + " ".join(configs),
                              input=b"y\ny\ny\ny\n")
        return self.last_execution
        
    def execute_stub(self):
        # generates histories based on a simulated loss curve
        # logs simulated batches of 1000 examples
        # batch time is l^2 * muls summed over layers
        # loss is an exponential with decay speed l^2 * muls for max layer
        # and asymptote l * muls summed over layers
        
        ls = [list(range(lmax + 1)) for lmax in self.config.model.lmaxes]
        muls = self.config.model.muls
        lmuls = [list(zip(li, mi)) for li,mi in list(zip(ls,muls))]
        print(lmuls)
        
        batch_time = sum(sum(l**2 * m for l,m in layer) for layer in lmuls)
        asymptote = sum(sum((l+1) * m for l,m in layer) for layer in lmuls) / 100
        decay = max(sum((l+1)**2 * m for l,m in layer) for layer in lmuls) / 1000000
        print(f"Batch time = {batch_time:.2f}, asymptote = {asymptote:.3f}, decay = {decay}")
        
        if self.history is None:
            self.__history = TrainTestHistory(
                examples_per_epoch=1000000, relevant_elements=[6,1],
                save_prefix = self.save_prefix, run_name=self.identifier,
                load=False, use_backup=False)
        
        epoch = self.epochs
        time = self.time
        example = self.examples
        while True:
            example += 1000
            time += batch_time
            epoch = example // self.config.data.training_size + 1
            
            test_loss = (math.exp(-example * decay) + asymptote) * (.9 + .2 * random.random())
            train_loss = test_loss * (.9 + .2 * random.random())
            RMSE = np.array([1.5 * test_loss, .5 * test_loss])
            mean_error = RMSE * (np.random.random(2) - .5)
            
            self.history.train.log_batch(
                batch_time, 0, 1000, 18000,
                time, example % self.config.data.training_size, example,
                train_loss, epoch=epoch, verbose=False                
            )
            
            self.history.test.log_test(
                example // 1000, epoch,
                (example %  self.config.data.training_size) // 1000,
                example % self.config.data.training_size,
                example, time, test_loss,
                mean_error, RMSE  
            )
            
            if example >= self.config.training.example_limit:
                print(f"Finished training after {example} examples")
                break
            if time >= self.config.training.time_limit:
                print(f"Finished training after {secs_to_str(time)}")
                break
            if epoch > self.config.training.epoch_limit:
                print(f"Finished training after {epoch-1} epochs")
                
        self.history.save()        
        print("Displaying graph...")
        self.history.test.plot(curve_fit=True, asymptote=True)
        self.close_history()
            
        
        
    #def __del__(self):
    #    self.close_history()
        
class EnsembleOfRuns:

    def __init__(self, parent_dir="runs", use_existing=True, configs=['exploration.ini'], start_training=False, stub=False):
        self.parent_dir = parent_dir
        self.config_files = configs
        self.__config = None
        self.__test_samples = None
        self.seed_length = None
        if use_existing:
            self.runs = {d.name: TrainingRun(parent_dir=parent_dir, identifier=d.name, parent_configs=configs)
                         for d in os.scandir(parent_dir) if d.is_dir()}
            self.active_runs = {name:run for name,run in self.runs.items() if not run.config.exploration.inactive}
            for r in self.runs.values():
                if r.config.exploration.seed is not None:
                    self.seed_length = len(r.config.exploration.seed)
                    break
        else:
            self.runs = {}
        self.stub=stub
    
    @property
    def config(self):
        if self.__config is None:
            self.__config = Config(*self.config_files, track_sources=True, use_command_args=False)
        return self.__config
        
    def activate(self, run, active=True):
        if active:
            run.set_config({'exploration':{'inactive':False}})
            self.active_runs[run.identifier] = run
        else:
            self.inactivate(run)
    
    def inactivate(self, run):
        run.set_config({'exploration':{'inactive':True}})
        self.active_runs.pop(run.identifier, None)
             
    
    def generate_run(self, failure_cost=.1, active=True):
        if self.seed_length is None:
            settings, seed = random_parameters_and_seed()
            self.seed_length = len(seed)
        else:
            seeds = self.seeds
            names = list(seeds)
            losses = self.mixed_losses()
            failures = self.failures
            sample_X = [seeds[n] for n in names]
            sample_y = [losses[n] for n in names]
            failures = [failures[n] for n in names]
            if self.test_samples is not None:
                test_X = self.test_samples.passes
                X_bounds = None
            else:
                X_bounds = [(0,1)] * self.seed_length
                test_X = None

            seed = bayes.next_sample(
                sample_X = sample_X,
                sample_y = sample_y,
                X_bounds = X_bounds,
                test_X = test_X,
                failures = failures,
                failure_cost = failure_cost
            )[0]
            if self.test_samples is not None:
                self.test_samples.remove(seed)
                self.test_samples.save()
            settings = generate_parameters(seed)
            
        settings = update(settings, {'exploration':{'seed':seed, 'inactive':not active}})    
        run = TrainingRun(parent_dir=self.parent_dir, settings=settings, parent_configs=self.config_files)
        self.runs[run.identifier] = run
        if active: self.active_runs[run.identifier] = run
        return run         
            
    @property
    def seeds(self):
        return {name:run.config.exploration.seed for name,run in self.runs.items()
                if run.config.exploration.seed is not None}
                
    @property
    def test_samples(self):
        if self.__test_samples is None:
            self.__test_samples = TrainableSampleGenerator.load(self.config.exploration.sample_file)
        return self.__test_samples
        
    def fill_test_samples(self):
        if self.test_samples is None:
            self.__test_samples = TrainableSampleGenerator.load(self.config.exploration.sample_file)
        if self.test_samples.num_passes < self.config.exploration.random_samples:
            self.test_samples.sample(num_passes=self.config.exploration.random_samples)
            self.test_samples.save()
    
    @property
    def failures(self):
        return {name:bool(run.config.exploration.failed) for name,run in self.runs.items()}

    def last_x(self, coord='time'):
        return max(max_value(run.history.test.coord(coord)) for run in self.runs.values())
    
    
    # train a run up to 'examples' or 'time' limit
    def train_run(self, run, examples=None, time=None, retries=1):
        if examples is None: examples = self.config.exploration.max_example
        if time is None: time = self.config.exploration.max_time
        run.set_config({'training':{
            'example_limit':examples,
            'time_limit':secs_to_str(time)
        }})
        if run.examples < examples and run.time < time:
            run.execute_run()
            for i in range(retries):
                if run.examples < examples and run.time < time:
                    run.execute_run()
                else:
                    break
                    
    def set_config(self, settings):
        parser = ConfigParser()
        parser.read([self.config_files[-1]])
        parser.read_dict(settings)
        parser.write(self.config_files[-1])
        self.__config = None
        
    new_limit = next_training_limit
        
    # increases the current training limit, and brings all active runs
    # up to that level (or inactivates them if they suck)            
    def training_increment(self, runs, max_time=None, max_example=None):
        new_time, new_example = self.new_limit()
        if max_time is None: max_time = new_time
        if max_example is None: max_example = new_example
        self.set_config({'exploration':{
            'max_time':max_time,
            'max_epoch':max_example
        }})
        for run in runs:
            self.bring_up_run(run)
    
    proceed = proceed_with_training
    
    # trains a run in steps until it reaches the current
    # maximum time/example, or until it fails a preliminary
    # judgment
    def bring_up_run(self, run):
        if run not in self.active_runs:
            self.activate(run)
        config = self.config.exploration
        
        # create steps at 1/8th, 1/4th, etc of max extent
        steps = np.array([.125, .25, .5, 1])
        example_steps = (config.max_example * steps).astype(int)
        time_steps = (config.max_time * steps).astype(int)
        
        # don't train very short runs
        example_steps = example_steps[example_steps >= config.example_increment]
        time_steps = time_steps[time_steps >= config.time_increment]
        
        
        # Step through time/example steps until current run has
        # not surpassed a step. This is the step we will start
        # training toward.
        for i in range(len(steps)):
            if run.time < time_steps[i] or run.examples < example_steps[i]:
                break
        
        for i in range(i, len(steps)):
            self.train_run(run, time=time_steps[i], examples=time_steps[i])
            if not self.proceed(run):
                self.inactivate(run)
                break
    
    order_runs = order_runs_policy
    
    # activates the best n runs and inactivates the others
    def activate_best(self, n=4, runs=None):
        if runs is None:
            runs = self.runs.values()
        runs = self.order_runs(runs)
        
        for run in runs[:n]:
            if run not in self.active_runs:
                self.activate(run)
            
        for run in runs[n:]:
            if run in self.active_runs:
                self.inactivate(run)
            
    def plot_active_runs(self, run=None):
        plt.figure(figsize=(12,8))
        axes = plt.gca()
        runs = list(self.active_runs.values())
        if run is not None:
            if run in runs:
                runs.remove(run)
            alpha = .5
            color = cycle("bgcmy")
        else:
            alpha = 1.
            color = cycle("bgrcmyk")
        
        for r in runs:
            c = next(color)
            r.history.test.plot(axes, show=False, color=c, alpha = .5 * alpha, label = r.identifier)
            #r.history.test.plot_curve_fit(show=False, color=c, alpha = alpha)
            
        if run is not None:
            run.history.test.plot(axes, show=False, color='k', alpha = .5, label = run.identifier)
            #run.history.test.plot_curve_fit(show=False, color='k', alpha = 1.)
            #run.history.test.plot_asymptote(show=False, color='r')
        
        plt.legend(loc="best")
        plt.show()
        
    def plot_extrapolation(self, run, x, coord='example'):
        plt.figure(figsize=(12,8))
        axes = plt.gca()
        runs = list(self.active_runs.values())
        if run in runs:
            runs.remove(run)
        alpha = 1.
        color = cycle("bgcmy")
        
        others = [r.history.test for r in runs]
        for r in runs:
            c = next(color)
            r.history.test.plot(axes, show=False, color=c, alpha = .5 * alpha, label = r.identifier)
            #test.plot_curve_fit(show=False, color=c, alpha = alpha)
        
        test = run.history.test
        x_axis = test.coord(coord)
        interval = x_axis[-1] - x_axis[-2]
        new_x = np.arange(x_axis[-1] + interval, x, interval)
        new_loss = np.zeros_like(new_x).astype(float)
        
        log_avg, other_avgs = None, None
        for i in range(len(new_x)):
            log_ext, log_avg, other_avgs, others = test.log_loss_extrapolate(new_x[i], others, coord, log_avg=log_avg, other_avgs=other_avgs)        
            new_loss[i] = np.exp(log_ext)
            
        test.plot(axes, show=False, color='k', alpha = .5, label = run.identifier)
        plt.plot(new_x,new_loss,'r:', label='extrapolate')
        
        plt.legend(loc="best")
        plt.show()
     
        
        
            
        
    # Deprecated
    def extrapolated_losses(self, coord='time', smoothing=5, active_only=False):
        last_x = self.last_x(coord)
        long_runs = {}
        short_runs = {}
        long_histories = set()
        losses = {}
        runs = self.active_runs if active_only else self.runs
        for name, run in runs.values():
            if max_value(run.history.test.coord(coord)) == last_x:
                long_runs[name] = run
                losses[name] = run.history.test.smoothed_loss(-1, smoothing)
                long_histories.add(run.history.test)
            else:
                short_runs[name] = run
        for name, run in short_runs.values():
            losses[name] = run.history.test.loss_extrapolate(last_x, long_histories, coord, smoothing)
        return losses

    ######## Some deprecated methods #########    
    # Deprecated
    def mixed_losses(self, t_coeff=.5, transform='log', smoothing=5, active_only=False):
        if transform=='none' or transform is None:
            trn = lambda x : x
            inv = lambda x : x
        elif transform=='log':
            trn = math.log
            inv = math.exp
        t_losses = self.extrapolated_losses(coord='time', smoothing=smoothing, active_only=active_only)
        b_losses = self.extrapolated_losses(coord='example', smoothing=smoothing, active_only=active_only)
        return {name:inv(t_coeff * trn(t_losses[name]) + trn((1-t_coeff) * b_losses[name])) for name in t_losses}
    
    # Deprecated
    # asymptotes and exponential models don't work
    def asymptotes(self, coord='example', active_only=False):
        runs = self.active_runs if active_only else self.runs
        return {name:run.history.test.asymptote(coord=coord) for name,run in runs.values()}
        
            
        
        

                

def max_value(seq, min=0):
    if len(seq):
        return seq[-1]
    else:
        return min

def merge_dicts(*dicts):
    return {key:tuple(d[key] for d in dicts) for key in set(dicts[0]).intersection(*dicts[1:])}
            
    


if __name__ == '__main__':
    if len(sys.argv) > 1:
        parent_dir = sys.argv[1]
        ensemble = EnsembleOfRuns(parent_dir=parent_dir)
        if len(sys.argv) > 2:
            run = ensemble.runs[sys.argv[2]]
            ensemble.plot_extrapolation(run, 2500000)
        else:  
            ensemble.plot_active_runs()    
    else:
        settings, seed = random_parameters_and_seed()
        print("Simulating random run:")
        print(settings)
        settings = update(settings, {'exploration':{'seed':seed}})    
        run = TrainingRun(settings=settings)
        run.execute_stub()


    

            

