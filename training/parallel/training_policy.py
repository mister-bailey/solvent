# Treat this much like a config file, editing the functions to set exploration policy
import exploration
import statistics as stat

# Answers whether or not it is worth continuing to train 'run'
def proceed_with_training(ensemble, run):
    # Some handy values accumulated here
    losses_dict = ensemble.mixed_losses(active_only = True)
    loss = losses_dict[run.identifier]
    del losses_dict[run.identifier]
    other_losses = losses_dict.values()
    progress = max(run.time / ensemble.config.exploration.max_time,
                   run.example / ensemble.config.exploration.max_example)
    
    #### Decision code goes here: ####

    mean_loss = stat.mean(other_losses)
    std_dev = stat.stdev(other_losses)
    
    if loss < mean_loss + std_dev / (progress + .00001):
        return True
    else:
        return False

    ##################################

# Returns new values (max_time, max_example)
def next_training_limit(ensemble):
    # Possibly useful values here
    config = ensemble.config.exploration
    max_time = config.max_time
    max_example = config.max_example
    time_increment = config.time_increment
    example_increment = config.example_increment
    
    if max_time is None:
        new_time = time_increment
    else:
        new_time = max_time * 2
        
    if max_example is None:
        new_example = example_increment
    else:
        new_example = max_example * 2
        
    return new_time, new_example
    

#
    
    
    

    
    