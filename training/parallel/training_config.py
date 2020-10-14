from glob import glob
import configparser
import sys
import os
from collections.abc import Mapping

# import code for evaluating radial models
from e3nn.radial import *
from laurent import *
from functools import partial


# parses a delimited string
def parse_list(s, separator=",", func=lambda x : x):
    return [ func(i.strip()) for i in s.split(separator) ]

title_case = lambda s : s.title()

def str_to_secs(s):
    t_strings = s.split(":")
    assert len(t_strings) <= 3, f"Time string '{s}' has too many terms!"
    return sum(eval(n) * 60 ** i for i,n in enumerate(reversed(t_strings)))

class NO_STORE:
    def __init__(self):
        return

# dummy section:
class SECTION:
    def __init__(self):
        self._mapping = {}

class ConfigSection:
    def __init__(self, mapping, load_all=True, eval_func=lambda x : x, eval_funcs={}, eval_error=True,
            key_func=lambda x : x, include_keys=None, exclude_keys={}, default=None, defaults={}):
        self._mapping = dict(mapping)
        self._eval_func = eval_func
        self._eval_funcs = eval_funcs
        self._eval_error = eval_error
        self._key_func = key_func
        self._include_keys = include_keys
        self._exclude_keys = set(exclude_keys)
        self._default = default
        self._defaults = defaults
        self._sub_ini = []

        if load_all:
            self.load_all()

    def load_all(self):
        if self._include_keys is None:
            keys = set(self._mapping.keys())
        else:
            keys = set(self._include_keys)
        keys.update(self._defaults.keys())
        for key in keys:
            if key not in self._exclude_keys:
                self.load(key)

    def load(self, key, eval_func=None, key_func=None, eval_error=None, **kwargs):
        if not eval_func:
            if key in self._eval_funcs:
                eval_func = self._eval_funcs[key]
            else:
                eval_func = self._eval_func
        if not key_func:
            key_func = self._key_func
        if eval_error is None:
            eval_error = self._eval_error
        
        if key not in self._mapping:
            if 'default' in kwargs:
                default = kwargs['default']
            elif key in self._defaults:
                default = self._defaults[key]
            else:
                default = self._default
            if default == NO_STORE:
                return
            value = default
        else:
            try:
                value = eval_func(self._mapping[key])
            except Exception:
                if eval_error:
                    raise
                value = self._mapping[key]
        self._mapping[key] = value

        key = key_func(key).replace(" ", "_")
        self.__dict__[key] = value
        return value

    # For pickling:
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        for key in self.__dict__.keys():
            if key.startswith('_'):
                del self_dict[key]
        return self_dict

    def items(self):
        return ((key, value) for (key, value) in self.__dict__.items()
            if (not key.startswith('_')) and key not in {'load_all', 'load', 'items'}) 

class Config:
    def atomic_number(self, e):
        if isinstance(e, str):
            if e.isnumeric(): return int(e)
            return self.symbol_to_number[e]
        else:
            return e

    def load_section(self, name, store_as=None, **kwargs):
        section = ConfigSection(self._parser[name], **kwargs)
        if not store_as:
            store_as = name
        store_as = store_as.replace(" ", "_")
        self.__dict__[store_as] = section
        return section

    def load_section_into_base(self, name, **kwargs):
        section = ConfigSection(self._parser[name], **kwargs)
        self.__dict__.update(section.items())
        return section.items()

    # For pickling:
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        for key in self.__dict__.keys():
            if key.startswith('_'):
                del self_dict[key]
        return self_dict

    def items(self):
        return ((key, value) for (key, value) in self.__dict__.items()
            if (not key.startswith('_')) and key not in
            {'atomic_number', 'load_section', 'load_section_into_base', 'items'})

    def get_sub_configs(self):
        sub_configs = []
        for _, section in self._parser.items():
            for key, value in section.items():
                if value is None and key.endswith('.ini'):
                    del section[key]
                    sub_configs.append(key)
        return sub_configs

    # parses config files
    # later filenames overwrite values from earlier filenames
    # (so you can have mini configs that change just a few settings)
    def __init__(self, *filenames, settings=None, _set_names=False):
        if _set_names: # Only here to satisfy pylint and the code highlighter
            self._set_names()
        self._parser = configparser.ConfigParser(allow_no_value=True)

        if len(filenames) == 0:
            filenames = ["training.ini"]
            if len(sys.argv) > 1:
                filenames += sys.argv[1:]

        #print(f"Loading config from files {filenames}...")
                
        files_worked = self._parser.read(filenames)
        print(f"Loaded config from files {files_worked}.")

        # load any sub-configs specified in the file:
        sub_files = self._parser.read(self.get_sub_configs())
        print(f"Loaded sub-config from files {sub_files}")
        

        # load in extra caller-provided settings:
        if isinstance(settings, Mapping):
            self._parser.read_dict(settings)

        # dictionaries between symbols and numbers
        self.load_section('symbols_numbers_dict', key_func=title_case, eval_func=int)
        self.symbol_to_number = {s:n for s,n in self.symbols_numbers_dict.items()}
        self.number_to_symbol = {n:s for s,n in self.symbols_numbers_dict.items()}
        #print(self.symbol_to_number)
        
        # device, all_elements, relevant_elements
        efunc = partial(parse_list, func=self.atomic_number)
        self.load_section_into_base('general', eval_funcs={
            'all_elements':efunc, 'relevant_elements':efunc, 'gpus':eval, 'parallel':eval},
            defaults={'device':'cuda', 'gpus':1, 'parallel':None})

        if self.device == 'cpu':
            self.parallel = False
        if self.parallel is None:
            self.parallel = (self.gpus != 1)
        if self.parallel:
            import torch.cuda
            count = torch.cuda.device_count()
            if self.gpus < 1:
                self.gpus = int(self.gpus * count)
            else:
                self.gpus = min(self.gpus, count)
        else:
            self.gpus = 1

        # all elements
        assert len(self.all_elements) == len(set(self.all_elements)), "duplicate element"
        self.n_elements = len(self.all_elements)

        # elements to predict NMR shieldings for
        for e in self.relevant_elements:
            assert e in self.all_elements, f"relevant element {e} not found in all_elements"
        assert len(self.relevant_elements) == len(set(self.relevant_elements)), "duplicate element"

        # affine correction to database
        if "affine_correction" in self._parser:
            self.load_section('affine_correction', eval_func=eval)
            if self.affine_correction.correct:
                self.affine_correction = {self.atomic_number(title_case(e)) : v
                            for e,v in self.affine_correction.items() if e != 'correct'}
                #for e in self.relevant_elements:
                #    if e not in self.affine_correction:
                #        self.affine_correction[e] = (1., 0.)
            else:
                self.affine_correction = None
        else:
            self.affine_correction = None
        
        # see reference training.ini for all the parameters in 'data'
        self.load_section('data', eval_func=eval, eval_funcs={'hdf5_filenames':glob},
                          defaults={'randomize':True, 'get_from_start':False,
                          'multi_jiggle_data':False, 'jiggles_per_molecule':1,
                          'test_train_shuffle':None, 'batch_preload':0}, eval_error=False)
        # where the raw data are stored
        if self.data.source.startswith('hdf5'):
            if self.data.source == 'hdf5_0':
                self.data.file_format = 0
            else:
                self.data.file_format = 1
            self.data.source = 'hdf5'
            assert len(self.data.hdf5_filenames) > 0, "no files found!"
        elif self.data.source == 'SQL':
            self.data.connect_params = self.load_section('connect_params')._mapping
            self.data.SQL_fetch_size = self.data.sql_fetch_size

        # wandb authentication
        self.load_section('wandb')

        # model parameters
        self.load_section('model', eval_func=eval, eval_error=False)
        self.model.kwargs = self.model._mapping
        self.model.Rs_in = [ (self.n_elements, 0, 1) ]  # n_features, rank 0 tensor, even parity
        self.model.Rs_out = [ (1,0,1) ]            # one output per atom, rank 0 tensor, even parity
        self.model.kwargs['Rs_in'] = self.model.Rs_in
        self.model.kwargs['Rs_out'] = self.model.Rs_out
        self.max_radius = self.model.max_radius

        # training parameters
        self.load_section('training', eval_func=eval, eval_funcs={'save_prefix':str, 'time_limit':str_to_secs},
                defaults={'save_prefix':None, 'resume':False, 'n_epochs':None, 'time_limit':None, 'use_wandb':False})


    # The only purpose of this section is to get rid of the red squiggly lines from
    # not "defining" parameters explicitly in this file. This function is not actually called.
    def _set_names(self):
        self.project = "Solvent"
        self.device = ""
        self.gpus = 1
        self.parallel = True

        self.all_elements = []
        self.relevant_elements = []

        self.symbols_numbers_dict = {}

        self.data = SECTION()
        self.data.source = ""
        self.data.hdf5_filenames = []

        self.data.testing_size = 0
        self.data.training_size = 0
        self.data.test_train_shuffle = ""
        self.data.randomize = True
        self.data.get_from_start = False

        self.data.multi_jiggle_data = False
        self.data.jiggles_per_molecule = 1

        self.data.n_molecule_processors = 0
        self.data.molecule_queue_cap = 0
        self.data.example_queue_cap = 0
        self.data.batch_queue_cap = 0

        self.data.sql_fetch_size = 0
        self.data.connect_params = {}

        self.data.batch_preload = 0

        self.connect_params = SECTION()
        self.connect_params.host = ""
        self.connect_params.user = ""
        self.connect_params.passwd = ""
        self.connect_params.db = ""
        
        self.wandb = SECTION()
        self.wandb.user = ''
        self.wandb.pswd = ''

        self.model = SECTION()
        self.model.kwargs = {}
        self.model.max_radius = 0
        self.model.Rs_in = [ (0,0,0) ]   # n_features, rank 0 tensor, even parity
        self.model.Rs_out = [ (0,0,0) ]  # one output per atom, rank 0 tensor, even parity

        self.training = SECTION()
        self.training.n_epochs = None             # number of epochs
        self.training.time_limit = None
        self.training.batch_size = 0           # minibatch sizes
        self.training.testing_interval = 0     # compute testing loss every n minibatches
        self.training.save_interval = 0  # save model every n minibatches
        self.training.save_prefix = ""   # save checkpoints to files starting with this
        self.training.learning_rate = 0        # learning rate
        self.training.resume = False
        self.training.num_checkpoints = 1
        self.training.use_wandb = False

        self.affine_correction = {}




