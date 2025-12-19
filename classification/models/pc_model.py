import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
import copy

class PCNET(nn.Module):
    def __init__(self, model: nn.Module):
        super(PCNET, self).__init__()

        self.activation = F.leaky_relu

        self.module_dict = {}

        self.module_dict['generative'] = nn.ModuleList()
        for name, module in model.named_children():
            if isinstance(module, nn.ModuleList):
                for i, submodule in enumerate(module):
                    copied = copy.deepcopy(submodule)
                    self.module_dict['generative'].append(copied)
                    self.add_module(f"{name}_{i}", copied)
            else:
                copied = copy.deepcopy(module)
                self.module_dict['generative'].append(copied)
                self.add_module(name, copied)

        self.energy = torch.zeros(1)
        self.criterion = torch.nn.MSELoss(reduction='sum')

        for layer in self.module_dict['generative']:
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight)


    def to(self, device: torch.device):
        super().to(device)
        self.energy = self.energy.to(device)
        return self
    

    def update_energy(self, input_data, output, _):

        pred = self.activation(self.module_dict['generative'][-1](self.module_dict['variational'][-1]))

        energy = self.criterion(self.module_dict['variational'][0], self.activation(self.module_dict['generative'][0](input_data))) + sum(
            self.criterion(self.module_dict['variational'][k], self.activation(self.module_dict['generative'][k](self.module_dict['variational'][k - 1])))
            for k in range(1, len(self.module_dict['generative'])-1)
        ) + self.criterion(output, pred)

        return energy, pred


    def inference_phase(self, input: torch.Tensor, output: torch.Tensor, alpha: float = 0.01, 
                        inference_timestamps: int = 10, momentum: float = 0):
        
        self.module_dict['variational'] = nn.ParameterList([nn.Parameter(self.activation(self.module_dict['generative'][0](input)).detach().clone())])
        for j in range(1, len(self.module_dict['generative']) - 1):
            self.module_dict['variational'].append(nn.Parameter(self.activation(self.module_dict['generative'][j](self.module_dict['variational'][j-1])).detach().clone()))

        inference_optimizer = torch.optim.SGD(self.module_dict['variational'], lr=alpha, momentum=momentum)


        for _ in range(inference_timestamps):
            self.energy, pred = self.update_energy(input, output, _)

            self.energy.backward()
            inference_optimizer.step()
            inference_optimizer.zero_grad()

        return pred


    def forward(self, x: torch.Tensor):
        for layer in self.module_dict['generative']:
            x = self.activation(layer(x))
        return x



def train_model(model: nn.Module, input_data: torch.Tensor, output_data: torch.Tensor, 
                generative_optimizer: torch.optim.Optimizer, criterion: nn.Module, 
                inference_lr: int, inference_time: int, momentum: float = 0.0, pc_mode: bool=True):
    if pc_mode:
        model_preds = model.inference_phase(input_data, output_data, alpha=inference_lr, inference_timestamps=inference_time, momentum=momentum)
    else:
        model_preds = model(input_data)

    loss = criterion(model_preds, output_data)
    if not pc_mode:
        loss.backward()

    generative_optimizer.step()
    generative_optimizer.zero_grad()
    
    predictions = torch.argmax(model_preds, dim=1)
    true_labels = torch.argmax(output_data, dim=1)
    accuracy = (predictions == true_labels).float().mean().item()
    
    return loss.item(), accuracy



def test_model(model: nn.Module, input_data: torch.Tensor, target_data: torch.Tensor, criterion: nn.Module):
    model_preds = model(input_data)
    loss = criterion(model_preds, target_data)
    predictions = torch.argmax(model_preds, dim=1)
    true_labels = torch.argmax(target_data, dim=1)
    accuracy = (predictions == true_labels).float().mean().item()
    
    return loss.item(), accuracy