import torch
import e3nn
from e3nn.networks import GatedConvParityNetwork, S2ConvNetwork
from e3nn.point.message_passing import Convolution
from copy import deepcopy
import torch_geometric as tg
import math
from time import perf_counter


def loss_fn_mae(x, y):
    return (x - y).abs().mean()


def loss_fn_mse(x, y):
    return ((x - y)**2).mean()


def loglinspace(rate, step, end=None):
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step * (1 - math.exp(-t * rate / step)))


def evaluate(model, dataloader, loss_fns, device, n_norm):
    model.to(device)
    with torch.no_grad():
        losses = []
        for data in dataloader:
            data = tg.data.Batch.from_data_list(data)
            data.to(device)
            output = model(data.x, data.edge_index,
                           data.edge_attr, n_norm=n_norm)
            for loss_fn in loss_fns:
                assert loss_fn is not None, "loss_fn cannot be None"
                loss = [loss_fn(output, data.y) for loss_fn in loss_fns]
                loss = torch.stack(loss, dim=0)
            losses.append(loss)
        return torch.stack(losses, dim=0).mean(dim=0)


def model_from_kwargs(model_kwargs):
    d = deepcopy(model_kwargs)
    Network = eval(d.pop('network'))
    Conv = eval(d.pop('conv'))
    return Network(convolution=Conv, **d)


def load_model(filename, model_kwargs=None):
    model_dict = torch.load(filename)
    if model_kwargs is None:
        model_dict = model_dict['model_kwargs']
    model = model_from_kwargs(model_kwargs)
    try:
        model.load_state_dict(model_dict['state_dict'])
    except:
        model.load_state_dict(model_dict)
    return model


def save_model(model_kwargs, model, filename, optimizer=None):
    model_dict = {
        'state_dict': model.state_dict(),
        'model_kwargs': model_kwargs
    }
    if optimizer is not None:
        model_dict['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(model_dict, filename)


def train(model, optimizer, dataloader, test_dataloader,
          iter=100, device="cuda", n_norm=5, scale_loss=1.):
    model.to(device)

    checkpoint_generator = loglinspace(0.3, 5)
    checkpoint = next(checkpoint_generator)

    dynamics = []
    wall_start = perf_counter()

    print(scale_loss)

    for step in range(iter):
        for data in dataloader:
            data = tg.data.Batch.from_data_list(data)
            data.to(device)
            output = model(data.x, data.edge_index,
                           data.edge_attr, n_norm=n_norm)
            loss = loss_fn_mse(output, data.y) * scale_loss ** 2
            loss_mae = loss_fn_mae(output, data.y) * scale_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        wall = perf_counter() - wall_start

        if step == checkpoint:
            print(checkpoint)
            checkpoint = next(checkpoint_generator)
            assert checkpoint > step

            loss_fns = [loss_fn_mse, loss_fn_mae]

            test_avg_loss = evaluate(model, test_dataloader, loss_fns, device, n_norm)
            train_avg_loss = evaluate(model, dataloader, loss_fns, device, n_norm)

            print(step, loss.item(), loss_mae.item())

            dynamics.append({
                'step': step,
                'wall': wall,
                'batch': {
                    'loss': loss.item(),
                    'mean_abs': loss_mae.item(),
                },
                'test': {
                    'loss': test_avg_loss[0] * scale_loss ** 2,
                    'mean_abs': test_avg_loss[1] * scale_loss,
                },
                'train': {
                    'loss': train_avg_loss[0] * scale_loss ** 2,
                    'mean_abs': train_avg_loss[1] * scale_loss,
                },
            })

            yield {
                'dynamics': dynamics,
                'state': model.state_dict()
            }
