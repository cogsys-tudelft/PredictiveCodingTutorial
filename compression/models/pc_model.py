import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class PCNET(nn.Module):
    def __init__(self, model: nn.Module, batch_size: int):
        super(PCNET, self).__init__()
        #The model is the decoder module

        self.activation = F.tanh
        self.out_activation = F.relu

        self.module_dict = {}

        self.module_dict['generative'] = nn.ModuleList()
        self.module_dict['variational'] = nn.ParameterList()

        for name, module in model.named_children():
            modules = module.named_children() if isinstance(module, nn.Sequential) else [(name, module)]
            
            for subname, submodule in modules:
                if isinstance(submodule, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.ConvTranspose2d)):
                    copied = copy.deepcopy(submodule)
                    self.module_dict['generative'].append(copied)
                    self.add_module(f"{name}_{subname}" if isinstance(module, nn.Sequential) else name, copied)
                elif isinstance(submodule, nn.Unflatten):
                    copied = copy.deepcopy(submodule)
                    prev = self.module_dict['generative'].pop(len(self.module_dict['generative'])-1)
                    combined = nn.Sequential(prev, copied)
                    self.module_dict['generative'].append(combined)
                    self.add_module(f"{name}_{subname}_seq" if isinstance(module, nn.Sequential) else f"{name}_seq", combined)

        if isinstance(self.module_dict['generative'][0], nn.Linear):
            first_layer_in_features = self.module_dict['generative'][0].in_features
        elif isinstance(self.module_dict['generative'][0], nn.Sequential):
            first_layer_in_features = self.module_dict['generative'][0][0].in_features
        else:
            raise ValueError("First layer must be either a Linear or a Sequential containing a Linear layer")

        with torch.no_grad():
            self.module_dict['variational'].append(nn.Parameter(torch.randn(size=(batch_size, first_layer_in_features)), requires_grad=True))
            dummy_input = torch.randn(batch_size, first_layer_in_features)

            for layer in self.module_dict['generative'][:-1]: 
                dummy_input = layer(dummy_input)
                self.module_dict['variational'].append(nn.Parameter(torch.randn(size=(dummy_input.shape)), requires_grad=True))


        self.energy = torch.zeros(1)
        self.criterion = torch.nn.MSELoss(reduction='sum')

        for layer in self.module_dict['generative']:
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight)


    def to(self, device: torch.device):
        super().to(device)
        self.energy = self.energy.to(device)
        
        for module in self.module_dict['generative']:
            module.to(device)
        
        for param in self.module_dict['variational']:
            param.data = param.data.to(device)
        
        return self
    

    def update_energy(self, stimuli):
        energy = sum(
                    self.criterion(self.module_dict['variational'][k+1], self.activation(self.module_dict['generative'][k](self.module_dict['variational'][k])))
                    for k in range(0, len(self.module_dict['generative'])-1)
                ) + self.criterion(stimuli, self.out_activation(self.module_dict['generative'][-1](self.module_dict['variational'][-1])))

        return energy


    def inference_phase(self, stimuli: torch.Tensor, alpha: float = 0.01, 
                        inference_timestamps: int = 10, momentum: float = 0):

        inference_optimizer = torch.optim.SGD(self.module_dict['variational'], lr=alpha, momentum=momentum)

        for _ in range(inference_timestamps):
            self.energy = self.update_energy(stimuli)

            self.energy.backward()
            inference_optimizer.step()
            inference_optimizer.zero_grad()

        return self.forward(self.module_dict['variational'][0])


    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            for layer in self.module_dict['generative'][:-1]:
                x = self.activation(layer(x))
            x = self.out_activation(self.module_dict['generative'][-1](x))
            return x



def train_model(model: nn.Module, stimuli: torch.Tensor, 
                generative_optimizer: torch.optim.Optimizer, criterion: nn.Module, 
                inference_lr: int, inference_time: int, momentum: float = 0.0, pc_mode: bool=True):
    if pc_mode:
        model_preds = model.inference_phase(stimuli, alpha=inference_lr, inference_timestamps=inference_time, momentum=momentum)
    else:
        model_preds = model(stimuli)

    loss = criterion(model_preds, stimuli)
    if not pc_mode:
        loss.backward()

    generative_optimizer.step()
    generative_optimizer.zero_grad()

    return loss.item()



def test_model(model: nn.Module, stimuli: torch.Tensor, criterion: nn.Module, inference_lr: int, inference_time: int, momentum: float = 0.0, pc_mode: bool=False):
    if pc_mode:
        model_preds = model.inference_phase(stimuli, alpha=inference_lr, inference_timestamps=inference_time, momentum=momentum)
    else:
        with torch.no_grad():
            model_preds = model(stimuli)
    loss = criterion(model_preds, stimuli)
    
    return loss.item()