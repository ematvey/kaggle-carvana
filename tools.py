import numpy as np
import torch
from torch.nn.modules.module import _addindent


def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential,
            torch.nn.modules.container.ModuleList,
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr += ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr


def _unit_bit_size(tensor):
    t = tensor.type()
    if t.endswith('FloatTensor'):
        return 32
    elif t.endswith('LongTensor'):
        return 64
    else:
        raise ValueError(t)


def tensor_mem_size(tensor):
    return _unit_bit_size(tensor) * np.prod(tensor.size())


def grad_fn_mem_size(grad_fn):
    bits = 0
    if hasattr(grad_fn, 'saved_tensors'):
        for t in grad_fn.saved_tensors:
            bits += tensor_mem_size(t)
    if hasattr(grad_fn, 'next_functions'):
        for func, num in grad_fn.next_functions:
            if func is not None:
                bits += grad_fn_mem_size(func)
    return bits


def graph_mem_size(var):
    return grad_fn_mem_size(var.grad_fn)


def graph_mem_size_mb(var):
    return graph_mem_size(var) / 8 / 1024 / 1024


def params_mem_size(model):
    return np.sum([tensor_mem_size(p.data) for p in model.parameters()])


def params_mem_size_mb(model):
    return params_mem_size(model) / 8 / 1024 / 1024
