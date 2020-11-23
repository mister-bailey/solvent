from enum import Enum
import re
import math
import os
from glob import glob
import time
import numpy as np
import h5py
import torch
torch.set_default_dtype(torch.float64)
import torch_geometric as tg
import e3nn
import e3nn.point.data_helpers as dh
import training_config
import tensor_constraint
from molecule_pipeline import ExampleBatch

### Code to Generate Molecules ###

# so we can normalize training data for the nuclei to be predicted
#relevant_elements = training_config.relevant_elements

# cpu or gpu
#device = training_config.device

# other parameters
#training_size = training_config.training_size
#testing_size = training_config.testing_size
#batch_size = training_config.batch_size

### Functions for Training ###

# saves a model and optimizer to disk
def save_checkpoint(model_kwargs, model, filename, optimizer, all_elements):
    model_dict = {
        'state_dict' : {key.removeprefix("module."):value for key,value in model.state_dict().items()},
        'model_kwargs' : model_kwargs,
        'optimizer_state_dict' : optimizer.state_dict(),
        'all_elements' : all_elements,
    }
    print("                                                                                                      ", end="\r", flush=True)
    print(f"Saving model to {filename}... ", end='')
    torch.save(model_dict, filename)
    file_size = os.path.getsize(filename) / 1E6
    print(f"occupies {file_size:.2f} MB.")

# deletes all but the {number} newest checkpoints
def cull_checkpoints(save_prefix, number):
    for f in sorted(glob(save_prefix + "-*-checkpoint.torch"), key = os.path.getmtime)[:-number]:
        os.remove(f)

# mean-squared loss (not RMS!)
def loss_function(predictions, data, use_tensor_constraint=False):
    
    #rank = str(data.x.device)[-1]
    if use_tensor_constraint:
        #print(f"[{rank}]: Converting 3x3 label to e3 tensor...", flush=True)
        observations = tensor_constraint.convert(data.y)
    else:
        observations = data.y
    #print(f"[{rank}]: Computing loss...", flush=True)
    residuals = predictions - observations
    weights = data.weights
    normalization = weights.sum()
    loss = (weights.t() @ residuals.square()) / normalization
    
    if use_tensor_constraint:
        #print(f"[{rank}]: Building loss return values...", flush=True)
        return loss[0], loss[1:9].sum(), residuals
    else:
        return loss, residuals


### Training Code ###


# train a single batch
def train_batch(data_queue, model, optimizer, use_tensor_constraint=False):
    # set model to training mode (for batchnorm)
    model.train()

    #data = data.to(device)
    data = data_queue.pop()
    
    rank = str(data.x.device)[-1]
    #print(f"[{rank}]: Running model...", flush=True)
    output = model(data.x, data.edge_index, data.edge_attr)
    #print(f"[{rank}]: Computing loss...", flush=True)
    scalar_loss, *tensor_losses, _ = loss_function(output, data, use_tensor_constraint=use_tensor_constraint)
    #print(f"[{rank}]: Portioning loss...", flush=True)
    if use_tensor_constraint:
        tensor_loss = tensor_losses[0]
        loss = scalar_loss + tensor_loss
    else:
        loss = scalar_loss

    # backward pass
    #print(f"[{rank}]: Zero grad...", flush=True)
    optimizer.zero_grad()
    #print(f"[{rank}]: Backward pass...", flush=True)
    loss.backward()
    #print(f"[{rank}]: Optimizer step...", flush=True)
    optimizer.step()
    #print(f"[{rank}]: Done update.", flush=True)

    # return RMSE
    if use_tensor_constraint:
        return np.sqrt(scalar_loss.item()), np.sqrt(tensor_loss.item())
    else:
        return (np.sqrt(loss.item()),)

# Collect list of examples into batches (slow, so only use for testing dataset)
# returns a list of batches, where the returned batches each have an extra field: example_list
def batch_examples(example_list, batch_size):
    batch_list = []
    for n in range(0,len(example_list),batch_size):
        sub_list = example_list[n:n+batch_size]
        pos = torch.cat([e.pos for e in sub_list])
        x = torch.cat([e.x for e in sub_list])
        y = torch.cat([e.y for e in sub_list])
        weights = torch.cat([e.weights for e in sub_list])
        atom_tally = 0
        sub_list_edges = []
        for e in sub_list:
            sub_list_edges.append(e.edge_index + atom_tally)
            atom_tally += e.pos.shape[0]
        edge_index = torch.cat(sub_list_edges, axis=1)
        edge_attr = torch.cat([e.edge_attr for e in sub_list])

        batch = ExampleBatch(pos, x, y, weights, edge_index, edge_attr, n_examples=len(sub_list))
        batch.example_list = sub_list
        batch_list.append(batch)
    return batch_list

#from training_config import Config
#config = Config()
#symbol_to_number = config.symbol_to_number
#number_to_symbol = config.number_to_symbol

def compare_models(model1, model2, data, tolerance=.01, copy_parameters=True):
    print("Comparing 2 models....")
    if copy_parameters:
        model2.load_state_dict(model1.state_dict())
    model1.eval()
    model2.eval()
    output1 = model1(data.x, data.edge_index, data.edge_attr)
    output2 = model2(data.x, data.edge_index, data.edge_attr)
    #print(torch.abs(output2 - output1) > tolerance)
    print(torch.cat((output1,output2),dim=1))
    print()


