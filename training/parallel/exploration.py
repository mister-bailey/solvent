from training_config import Config
from configparser import ConfigParser
import os
import sys
import subprocess
import math
import random
from history import TrainTestHistory
import bayes_search as bayes

class TrainingRun:
    def __init__(self, parent_dir="runs/", identifier=None, config_file=None, settings={},
                 create_file=True, create_identifier=True, parent_configs=['exploration.ini']):
        self.parent_configs = parent_configs
        self.settings = {k:str(v) for k,v in settings.items()} # may need to allow alternate string conversions

        if identifier is None:
            if settings != {} and create_identifier:
                self.identifier = "run-" + hex(hash(settings))[-8:]
                if os.path.isdir(identifier):
                    raise Exception(f"Hash collision with existing training run {identifier}")
            else:
                raise Exception("No training run identifier provided!")
        self.identifier = identifier

        self.run_dir = parent_dir + identifier + "/"
        os.makedirs(self.run_dir, exist_ok=True)
        self.save_prefix = self.run_dir + identifier[:7]

        settings = {'training':{'save_prefix':self.save_prefix, 'resume':True}}

        if config_file is None:
            config_file = self.run_dir + "training.ini"
        self.config_file = config_file

        parser = ConfigParser(allow_no_value=True)
        if os.path.isfile(self.config_file):
            parser.read(self.config_file)
        if settings != {}:
            parser.read_dict(settings)
            parser.write(self.config_file)

        self.__history = None
        self.__local_config = None
        self.__config = None
        
    @property
    def history(self):
        if self.__history is None:
            if os.path.isfile(self.save_prefix + "-history.torch"):
                self.__history = TrainTestHistory.load(prefix = self.save_prefix)
        return self.__history

    # creates a Config by reading local config_file, then training.ini, then adding in self.settings
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
        parser = ConfigParser()
        parser.read([self.config_file])
        parser.read_dict(settings)
        parser.write(self.config_file)
        
            
    # saves the contents of self.__local_config to a config file
    #def save_config(self):
    #    sections = self.local_config.__dict__()
    #    settings = {sec_name:sec.__dict__() for sec_name,sec in sections.items()}
    #    parser = ConfigParser(allow_no_value=True)
    #    parser.read_dict(settings)
        
        
    def execute_run(self, configs=[]):
        configs.append(self.config_file)
        print("\n==============================================")
        print(f"Run {self.identifier}:")
        print(f"      run dir: {self.run_dir}")
        print(f"  save_prefix: {self.save_prefix}")
        print(f"  config file: {self.config_file}")
        print("----------------------------------------------\n")
        self.__history = None # a training run invalidates any loaded history
        self.last_execution = subprocess.run("python training.py " + " ".join(configs),
                              input=b"y\ny\ny\ny\n")
        
class EnsembleOfRuns:

    def __init__(self, parent_dir="runs/", use_existing=True, configs=['exploration.ini'], start_training=False):
        self.parent_dir = parent_dir
        self.config_files = configs
        self.__config = None
        self.seed_length = None
        if use_existing:
            self.runs = {d.name: TrainingRun(parent_dir=parent_dir, identifier=d.name, parent_configs=configs)
                         for d in os.scandir(parent_dir) if d.is_dir()}
            for r in self.runs.values():
                if r.config.exploration.seed is not None:
                    self.seed_length = len(r.config.exploration.seed)
                    break
        else:
            self.runs = {}
    
    @property
    def config(self):
        if self.__config is None:
            self.__config = Config(*self.config_files, track_sources=True, use_command_args=False)
        return self.__config

    # num training cycles through the ensemble
    # num < 0 means keep on going
    def training_cycle(self, num=-1):
        print(f"Training ensemble of runs for {num} cycles...")
        configs = [] if self.config is None else [self.config]
        n = 0
        while n != num:
            for run in self.runs.values():
                run.execute_run(configs)
    
    def generate_run(self, cost_of_failure=.1):
        if self.seed_length is None:
            settings, seed = self.random_parameters_and_seed()
            self.seed_length = len(seed)
        else:
            seeds = self.seeds()
            names = list(seeds)
            losses = self.mixed_losses()
            failures = self.failures()
            sample_X = [seeds[n] for n in names]
            sample_y = [losses[n] for n in names]
            X_bounds = [(0,1)] * len(names)
            failures = [failures[n] for n in names]
            seed = bayes.next_sample(
                sample_X = sample_X,
                sample_y = sample_y,
                X_bounds = X_bounds,
                failures = failures,
                failure_cost = .1
            )[0]
            settings = self.generate_parameters(seed)
        run = TrainingRun(parent_dir=self.parent_dir, settings=settings.update({'exploration':{'seed':seed}}),
                            parent_configs=self.config_files)
        self.runs[run.identifier] = run
        return run         
        
    def random_parameters_and_seed(self, distribution=random.random):
        gen = TrackingGenerator(distribution)
        settings = self.generate_parameters(gen)
        return settings, gen.seed
    
    def generate_parameters(self, var):
        """
        Defines a distribution of parameters
        Returns a settings dictionary
        var is an iterable of variables in the range [0,1) which
        we can make use of.
        """
        var = iter(var)
        model={}
        training={}
        settings = {'model':model, 'training':training}

        # radial_basis is exponential from 8 to 31
        model['radial_basis'] = int( 2 ** (3 + 2 * next(var)) )

        # radial_layers is from 1 to 15
        model['radial_layers'] = int(2 ** (4 * next(var)) )

        # radial_h from 10 to 79
        model['radial_h'] = int( 5 * 2 ** (1 + 3 * next(var)) )
        
        # numlayers from 1 to 9
        numlayers = int( 2 ** (3.32192 * next(var)))
        
        # lmax is a polynomial on [0,1], scaled to numlayers
        # lmax = l0 + l1 x + l2 x^2 + l3 x^3
        # l0 computed last to ensure positivity
        l3 = math.asin(2 * next(var) - 1) * 2
        l2 = math.asin(2 * next(var) - 1) * 2
        l1 = math.asin(2 * next(var) - 1) * 2
        
        ns = [l / numlayers for l in range(numlayers)]
        lmaxes = [int(round(l1 * n + l2 * n**2 + l3 * n**3)) for n in ns]
        bump = -min(lmaxes)
        lmin = int(next(var) * 4)
        lmaxes = [l + bump + lmin for l in lmaxes]
        
        # multiplicities are a fn of both layer n / numlayers and x = 10/(l+5)
        # m = m0 + m01 x + m10 n + m11 xn
        m11 = math.asin(2 * next(var) - 1) * 5
        m10 = math.asin(2 * next(var) - 1) * 5
        m01 = math.asin(2 * next(var) - 1) * 5
        
        xs = [[10 / (l + 5) for l in range(lmaxes[n]+1)] for n in range(numlayers)]
        muls = [[int(m01 * x + m10 * n + m11 * x * n) for x in xl] for n,xl in zip(ns,xs)]
        bump = -min([min(lmul) for lmul in muls])
        mulmin = int(next(var) * 4) + 1
        muls = [[m + bump + mulmin for m in lmul] for lmul in muls]

        return settings
         
    def seeds(self):
        return {name:run.config.exploration.seed for name,run in self.runs.items()
                if run.config.exploration.seed is not None}
    
    def failures(self):
        return {name:bool(run.config.exploration.failed) for name,run in self.runs.items()}

    def last_x(self, coord='time'):
        return max(max_value(run.history.test.coord(coord)) for run in self.runs.values())
    
    def extrapolated_losses(self, coord='time', smoothing=5):
        last_x = self.last_x(coord)
        long_runs = {}
        short_runs = {}
        long_histories = set()
        losses = {}
        for name, run in self.runs.values():
            if max_value(run.history.test.coord(coord)) == last_x:
                long_runs[name] = run
                losses[name] = run.history.test.smoothed_loss(-1, smoothing)
                long_histories.add(run.history.test)
            else:
                short_runs[name] = run
        for name, run in short_runs.values():
            losses[name] = run.history.test.loss_extrapolate(last_x, long_histories, coord, smoothing)
        return losses
        
    def mixed_losses(self, t_coeff=.5, transform='log', smoothing=5):
        if transform=='none' or transform is None:
            trn = lambda x : x
            inv = lambda x : x
        elif transform=='log':
            trn = math.log
            inv = math.exp
        t_losses = self.extrapolated_losses(coord='time', smoothing=smoothing)
        b_losses = self.extrapolated_losses(coord='example', smoothing=smoothing)
        return {inv(t_coeff * trn(t_losses[name]) + trn((1-t_coeff) * b_losses[name])) for name in self.runs}

def max_value(seq, min=0):
    if len(seq):
        return seq[-1]
    else:
        return min

def merge_dicts(*dicts):
    return {key:tuple(d[key] for d in dicts) for key in set(dicts[0]).intersection(*dicts[1:])}
            
    

class TrackingGenerator:
    def __init__(self, dist):
        assert dist == random.random, "Non-uniform distributions not yet supported!"
        self.dist = dist
        self.seed = []
        
    def __iter__(self):
        return self
    
    def __next__(self):
        var = self.dist()
        self.seed += [var]
        return var


if __name__ == '__main__':
    # do something here
    config = sys.argv[1] if len(sys.argv) > 1 else "exploration.ini"
    ensemble = EnsembleOfRuns(config = config)
    ensemble.training_cycle(10)

    

            

