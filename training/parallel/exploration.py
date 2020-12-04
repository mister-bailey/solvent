from training_config import Config, secs_to_str
from configparser import ConfigParser
import os
import sys
import subprocess
import math
import random
from history import TrainTestHistory
import bayes_search as bayes
from training_policy import proceed_with_training, next_training_limit

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
                self.__history = TrainTestHistory(save_prefix = self.save_prefix, load=True, use_backup=False)
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
        parser = ConfigParser()
        parser.read([self.config_file])
        parser.read_dict(settings)
        parser.write(self.config_file)
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
        
        
    def execute_run(self, configs=[]):
        self.close_history()
        configs = self.parent_configs + [self.config_file] + configs
        print("\n==============================================")
        print(f"Run {self.identifier}:")
        print(f"      run dir: {self.run_dir}")
        print(f"  save_prefix: {self.save_prefix}")
        print(f"  config file: {self.config_file}")
        print("----------------------------------------------\n")
        self.last_execution = subprocess.run("python training.py " + " ".join(configs),
                              input=b"y\ny\ny\ny\n")
        return self.last_execution
        
    def __del__(self):
        self.close_history()
        
class EnsembleOfRuns:

    def __init__(self, parent_dir="runs/", use_existing=True, configs=['exploration.ini'], start_training=False):
        self.parent_dir = parent_dir
        self.config_files = configs
        self.__config = None
        self.seed_length = None
        if use_existing:
            self.runs = {d.name: TrainingRun(parent_dir=parent_dir, identifier=d.name, parent_configs=configs)
                         for d in os.scandir(parent_dir) if d.is_dir()}
            self.active_runs = {name:run for name,run in self.runs.itemse() if not run.config.exploration.inactive}
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
                test_X = self.test_samples
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
            settings = generate_parameters(seed)
            
        settings = settings.update({'exploration':{'seed':seed, 'inactive':not active}})    
        run = TrainingRun(parent_dir=self.parent_dir, settings=settings, parent_configs=self.config_files)
        self.runs[run.identifier] = run
        if active: self.active_runs[run.identifier] = run
        return run         
            
    @property
    def seeds(self):
        return {name:run.config.exploration.seed for name,run in self.runs.items()
                if run.config.exploration.seed is not None}
    
    @property
    def failures(self):
        return {name:bool(run.config.exploration.failed) for name,run in self.runs.items()}

    def last_x(self, coord='time'):
        return max(max_value(run.history.test.coord(coord)) for run in self.runs.values())
    
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
    
    def bring_up_run(self, run):
        run.set_config({'exploration':{'inactive':False}})
        max_example = self.config.exploration.max_example
        max_time = self.config.exploration.max_time
        steps = np.array([.125, .25, .5, 1])
        example_steps = (max_example * steps).astype(int)
        time_steps = (max_time * steps).astype(int)
        
        for i in range(len(steps)):
            if run.time < time_steps[i] or run.examples < time_steps[i]:
                break
        
        for i in range(i, len(steps)):
            self.train_run(run, time=time_steps[i], examples=time_steps[i])
            if not self.proceed(run):
                run.set_config({'exploration':{'inactive':True}})
                break
            
        
        

def random_parameters_and_seed(distribution=random.random):
    gen = TrackingGenerator(distribution)
    settings = generate_parameters(gen)
    return settings, gen.seed

def generate_parameters(var):
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

    # number_of_basis is exponential from 8 to 31
    model['number_of_basis'] = int( 2 ** (3 + 2 * next(var)) )

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
    model['lmaxes'] = lmaxes
    
    # multiplicities are a fn of both n = layer / numlayers and x = 10/(l+5)
    # m = m0 + m01 x + m10 n + m11 xn
    m11 = math.asin(2 * next(var) - 1) * 5
    m10 = math.asin(2 * next(var) - 1) * 5
    m01 = math.asin(2 * next(var) - 1) * 5
    
    xs = [[10 / (l + 5) for l in range(lmaxes[n]+1)] for n in range(numlayers)]
    muls = [[int(m01 * x + m10 * n + m11 * x * n) for x in xl] for n,xl in zip(ns,xs)]
    bump = -min([min(lmul) for lmul in muls])
    mulmin = int(next(var) * 4) + 1
    muls = [[m + bump + mulmin for m in lmul] for lmul in muls]
    model['muls'] = muls

    return settings
                

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
    configs = sys.argv[1:] if len(sys.argv) > 1 else ["exploration.ini"]
    ensemble = EnsembleOfRuns(configs = configs)
    ensemble.training_cycle(10)

    

            

