# Treat this much like a config file, editing the functions to set exploration policy
#import exploration
import statistics as stat
import math

# Answers whether or not it is worth continuing to train 'run'
def proceed_with_training(ensemble, run):
    return True
    # Some handy values accumulated here    
    asymptotes_dict = ensemble.asymptotes(active_only = True)
    asymptote = asymptotes_dict[run.identifier]
    del asymptotes_dict[run.identifier]
    other_asymptotes = asymptotes_dict.values()
    other_worst = max(other_asymptotes, key=lambda x: x[0])
    
    progress = max(run.time / ensemble.config.exploration.max_time,
                   run.example / ensemble.config.exploration.max_example)
    
    #### Decision code goes here: ####
    
    if asymptote[0] - 4 * asymptote[1] < other_worst[0] + 4 * other_worst[1]:
        return True
    else:
        return False

    ##################################

# Returns new values (max_time, max_example)
def next_training_limit(ensemble, step=None):
    config = ensemble.config.exploration

    if config.example_schedule is not None and step < len(config.example_schedule):
        new_example = config.example_schedule[step]
    else:
        new_example = config.max_example * 2
    
    if config.time_schedule is not None and step < len(config.time_schedule):
        new_time = config.time_schedule[step]
    else:
        new_time = config.max_time * 2
    
    return new_time, new_example
    

# puts runs in order of how well they're performing; 0th is best
#def order_runs(ensemble, runs):
#    return sorted(runs, key=lambda r: r.history.test.asymptote[0])

def order_runs(ensemble, runs):
    losses = ensemble.mixed_log_losses()
    return sorted(runs, key = lambda r : losses[r.identifier])
    

import random

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

   
    

    
    