import matplotlib.pyplot as plt
import torch 

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