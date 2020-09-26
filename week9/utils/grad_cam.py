import random
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class GradCam:
    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.hooks = []
        self.fmap_pool = dict()
        self.grad_pool = dict()

        def forward_hook(module, input, output):
            self.fmap_pool[module] = output.detach().cpu()

        def backward_hook(module, grad_in, grad_out):
            self.grad_pool[module] = grad_out[0].detach().cpu()

        for layer in layers:
            self.hooks.append(layer.register_forward_hook(forward_hook))
            self.hooks.append(layer.register_backward_hook(backward_hook))

    def close(self):
        for hook in self.hooks:
            hook.remove()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __call__(self, *args, **kwargs):
        self.model.zero_grad()
        return self.model(*args, **kwargs)

    def get(self, layer):
        assert layer in self.layers, f'{layer} not in {self.layers}'
        fmap_b = self.fmap_pool[layer]  # [N, C, fmpH, fmpW]
        grad_b = self.grad_pool[layer]  # [N, C, fmpH, fmpW]

        grad_b = F.adaptive_avg_pool2d(grad_b, (1, 1))  # [N, C, 1, 1]
        gcam_b = (fmap_b * grad_b).sum(dim=1, keepdim=True)  # [N, 1, fmpH, fmpW]
        gcam_b = F.relu(gcam_b)

        return gcam_b


class GuidedBackPropogation:
    def __init__(self, model):
        self.model = model
        self.hooks = []

        def backward_hook(module, grad_in, grad_out):
            if isinstance(module, nn.ReLU):
                return tuple(grad.clamp(min=0.0) for grad in grad_in)

        for name, module in self.model.named_modules():
            self.hooks.append(module.register_backward_hook(backward_hook))

    def close(self):
        for hook in self.hooks:
            hook.remove()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __call__(self, *args, **kwargs):
        self.model.zero_grad()
        return self.model(*args, **kwargs)

    def get(self, layer):
        return layer.grad.cpu()

def colorize(tensor, colormap=plt.cm.jet):
    '''Apply colormap to tensor
    Args:
        tensor: (FloatTensor), sized [N, 1, H, W]
        colormap: (plt.cm.*)
    Return:
        tensor: (FloatTensor), sized [N, 3, H, W]
    '''
    tensor = tensor.clamp(min=0.0)
    tensor = tensor.squeeze(dim=1).numpy() # [N, H, W]
    tensor = colormap(tensor)[..., :3] # [N, H, W, 3]
    tensor = torch.from_numpy(tensor).float()
    tensor = tensor.permute(0, 3, 1, 2) # [N, 3, H, W]
    return tensor

def normalize(tensor, eps=1e-8):
    '''Normalize each tensor in mini-batch like Min-Max Scaler
    Args:
        tensor: (FloatTensor), sized [N, C, H, W]
    Return:
        tensor: (FloatTensor) ranged [0, 1], sized [N, C, H, W]
    '''
    N = tensor.size(0)
    min_val = tensor.contiguous().view(N, -1).min(dim=1)[0]
    tensor = tensor - min_val.view(N, 1, 1, 1)
    max_val = tensor.contiguous().view(N, -1).max(dim=1)[0]
    tensor = tensor / (max_val + eps).view(N, 1, 1, 1)
    return tensor


def draw_gad_cam(model, device, dataloader):
    model.eval()
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    for img, lbl in zip(images, labels):
        inp_b = img.unsqueeze(dim=0)  # [N, 3, 224, 224]
        inp_b = inp_b.to(device)

        with GradCam(model, [model.layer4]) as gcam:
            out_b = gcam(inp_b)  # [N, C]
            out_b[:, lbl].backward()

            gcam_b = gcam.get(model.layer4)  # [N, 1, fmpH, fmpW]
            gcam_b = F.interpolate(gcam_b, [32, 32], mode='bilinear', align_corners=False)  # [N, 1, inpH, inpW]
            # save_image(normalize(gcam_b), './gcam.png')

        with GuidedBackPropogation(model) as gdbp:
            inp_b = inp_b.requires_grad_()  # Enable recording inp_b's gradient
            out_b = gdbp(inp_b)
            out_b[:, lbl].backward()

            grad_b = gdbp.get(inp_b)  # [N, 3, inpH, inpW]
            grad_b = grad_b.mean(dim=1, keepdim=True)  # [N, 1, inpH, inpW]
            # save_image(normalize(grad_b), './grad.png')

        mixed = gcam_b * grad_b
        heatmap = normalize(mixed)
        img = img / 2 + 0.49139968  # unnormalize
        npimg = img.numpy()

        ## plotting the image and gradmap
        f, axarr = plt.subplots(nrows=1, ncols=2)
        plt.sca(axarr[0]);
        plt.imshow(heatmap.squeeze());
        plt.title('image')
        plt.sca(axarr[1]);
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        # plt.imshow(img);
        plt.title('grad_cam')
        plt.show()

