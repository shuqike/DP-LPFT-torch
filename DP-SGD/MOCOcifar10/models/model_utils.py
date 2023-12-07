import torch
import torch.nn

class ParamWrapperModule(torch.nn.Module):
    def __init__(self, param):
        super().__init__()
        self._param = param

def set_linear_layer(layer, coef, intercept):
    coef_tensor = torch.tensor(coef, dtype=layer.weight.dtype).cuda()
    bias_tensor = torch.tensor(intercept, dtype=layer.bias.dtype).cuda()
    coef_param = torch.nn.parameter.Parameter(coef_tensor)
    bias_param = torch.nn.parameter.Parameter(bias_tensor)
    layer.weight = coef_param
    layer.bias = bias_param

def set_requires_grad(component, val):
    for param in component.parameters():
        param.requires_grad = val
