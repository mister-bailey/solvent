from training_config import Config
from configparser import ConfigParser
import os
import sys
import subprocess
import math
from history import TrainTestHistory

class TrainingRun:
    def __init__(self, parent_dir="runs/", identifier=None, config_file=None, settings={}, create_file=True, create_identifier=True):
        settings = {k:str(v) for k,v in settings.items()} # may need to allow alternate string conversions
        parser = ConfigParser()

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

        parser.add_section('training')
        parser['training']['save_prefix'] = self.save_prefix
        parser['training']['resume'] = "True"

        if config_file is None:
            config_file = self.run_dir + "training.ini"
        self.config_file = config_file

        print(f"Config file {self.config_file}")
        if os.path.isfile(self.config_file):
            print(f"Found config file for {self.identifier}!")
            parser.read(self.config_file)
        if settings != {}:
            parser.read_dict(settings)
            parser.write(self.config_file)

        self.settings = {sec : dict(parser[sec]) for sec in parser}
        self.__history = None

    def load_history(self):
        self.__history = TrainTestHistory.load(prefix = self.save_prefix)
        return self.__history

    @property
    def history(self):
        if self.history is None:
            self.load_history()
        return self.__history
        

    # creates a Config by reading from filenames then adding in self.settings
    # note that if *filenames is empty, we use training.ini + command line arguments
    def load_config(self, *filenames):
        self.config = Config(*filenames, self.settings)
        return self.config

    def execute_run(self, configs=[]):
        configs.append(self.config_file)
        print("\n==============================================")
        print(f"Run {self.identifier}:")
        print(f"      run dir: {self.run_dir}")
        print(f"  save_prefix: {self.save_prefix}")
        print(f"  config file: {self.config_file}")
        print("----------------------------------------------\n")
        self.history = None # a training run invalidates any loaded history
        self.last_execution = subprocess.run("python training.py " + " ".join(configs),
                              input=b"y\ny\ny\ny\n")
        
class EnsembleOfRuns:

    def __init__(self, parent_dir="runs/", use_existing=True, config=None, start_training=False):
        self.parent_dir = parent_dir
        self.config = config
        if use_existing:
            self.runs = {d.name: TrainingRun(parent_dir=parent_dir, identifier=d.name)
                         for d in os.scandir(parent_dir) if d.is_dir()}

    # num training cycles through the ensemble
    # num < 0 means keep on going
    def training_cycle(self, num=-1):
        print(f"Training ensemble of runs for {num} cycles...")
        configs = [] if self.config is None else [self.config]
        n = 0
        while n != num:
            for run in self.runs.values():
                run.execute_run(configs)

    
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

        # radial_layers is from 3 to 14
        model['radial_layers'] = int( -1 + 2 ** (2 + 2 * next(var)) )

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


if __name__ == '__main__':
    # do something here
    config = sys.argv[1] if len(sys.argv) > 1 else "exploration.ini"
    ensemble = EnsembleOfRuns(config = config)
    ensemble.training_cycle(10)

    

            

