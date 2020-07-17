import matplotlib.pyplot as plt
import torch
import numpy as np
import sys

def show_radial_parameters(model, max_radius=None):
    radial_parameters={}

    for name, param in model.named_parameters():
        if name.endswith("kernel.R.f.weights.0"):
            radial_parameters[name] = param

    radial_data = torch.cat([p.data for p in radial_parameters.values()], dim=0).cpu()
    radial_square_density = radial_data.pow(2).mean(dim=0)
    radial_rms = radial_square_density.pow(.5)

    plt.figure()
    if max_radius is not None:
        x_coords = range(len(radial_rms))
    else:
        x_coords = [max_radius * x / len(radial_rms) for x in range(len(radial_rms))]
    plt.bar(x_coords,radial_rms)
    plt.ylabel("RMS")
    plt.show()

def print_parameter_size(model):
    for name, param in model.named_parameters():
        print(name + ': ' + str(list(param.shape)))

def count_parameters(model):
    model_count_dict = {} # layer # -> n_params
    n_total_parameters = 0
    for name, param in model.named_parameters():
        n_params = np.prod(param.shape)
        n_total_parameters += n_params
        fields = name.split(".")
        layer = fields[1]
        if layer not in model_count_dict:
            model_count_dict[layer]=0
        model_count_dict[layer] += n_params
    for layer,n_params in model_count_dict.items():
        print(f"Layer {layer}: {n_params}")
    print(f"Total parameters: {n_total_parameters}")

# goshippo.com/blog/measure-real-size-any-python-object
def get_object_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_object_size(v, seen) for v in obj.values()])
        size += sum([get_object_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_object_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_object_size(i, seen) for i in obj])
    return size
