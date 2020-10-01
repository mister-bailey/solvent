from training_config import Config
from configparser import ConfigParser
import os
import sys
import subprocess
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
        self.history = None

    def load_history(self):
        self.history = TrainTestHistory.load(prefix = self.save_prefix)
        return self.history

    # creates a Config by reading from filenames then adding in self.settings
    # note that if *filenames is empty, we use training.ini + command line arguments
    def load_config(self, *filenames):
        self.config = Config(*filenames, self.settings)
        return self.config

    def execute_run(self):
        print(f"Executing run {self.identifier}...")
        self.last_execution = subprocess.run("python training.py " + self.config_file,
                              input=b"y\ny\ny\ny\n")
        
class EnsembleOfRuns:

    def __init__(self, parent_dir="runs/", use_existing=True, start_training=False):
        self.parent_dir = parent_dir
        if use_existing:
            self.runs = {d.name: TrainingRun(parent_dir=parent_dir, identifier=d.name)
                         for d in os.scandir(parent_dir) if d.is_dir()}

    # num training cycles through the ensemble
    # num < 0 means keep on going
    def training_cycle(self, num=-1):
        print(f"Training ensemble of runs for {num} cycles...")
        n = 0
        while n != num:
            for run in self.runs.values():
                print("\n==============================================")
                print(f"Run {run.identifier}:")
                print(f"      run dir: {run.run_dir}")
                print(f"  save_prefix: {run.save_prefix}")
                print(f"  config file: {run.config_file}")
                print("----------------------------------------------\n")
                run.execute_run()

    
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

        return settings




if __name__ == '__main__':
    # do something here
    ensemble = EnsembleOfRuns()
    if len(sys.argv) > 1:
        num = int(sys.argv[1])
    else:
        num = 10
    ensemble.training_cycle(10)

    

            

