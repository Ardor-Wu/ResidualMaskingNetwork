#!/usr/bin/env python
# coding=utf-8

'Visualize model'

__author__ = 'Geovanni Zhang'

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from os.path import isfile
import os
import numpy as np
import matplotlib.pyplot as plt


class Visualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """

    def __init__(self, model, batch_size=1, name='example', device='cuda', writer=None, inputs=None):
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self.writer = writer
        self.inputs = inputs

        parameters_path = './parameters/{name}.pt'.format(name=name)
        if isfile(parameters_path):
            model.load_state_dict(torch.load(parameters_path))
        self.model.eval()

    def hook_layer(self, selected_layer, selected_filter):
        def hook_function(module, grad_in, grad_out):
            self.conv_output = grad_out[0, selected_filter]

        if not isinstance(selected_layer, nn.Module):
            raise RuntimeError('Selected layer must be a instance of nn.Module!')
        selected_layer.register_forward_hook(hook_function)

    def visualise_layer(self, selected_layer, selected_filter):
        path = "./result/gradient_ascent"
        if not os.path.exists(path):
            os.makedirs(path)
        if self.writer is None:
            writer = SummaryWriter()
        else:
            writer = self.writer
        self.conv_output = 0

        if self.inputs is None:
            inputs = torch.empty(self.batch_size, 3, 224, 224)
            nn.init.uniform_(inputs, a=0, b=255)
        else:
            inputs = self.inputs

        grid_name = 'Origin Image'
        created_image = inputs.to(self.device).requires_grad_()
        temp = created_image.expand(-1, 3, -1, -1)
        # grid_data = make_grid(temp, normalize=True, scale_each=True)
        grid_data = make_grid(temp)
        writer.add_image(grid_name, grid_data, 0)
        compare_image = created_image.clone()

        self.hook_layer(selected_layer, selected_filter)
        # optimizer = optim.SGD([created_image], lr=0.01, momentum=0.9)
        optimizer = optim.Adam([created_image], lr=1e-1, weight_decay=1e-6)

        for n_iter in range(200):
            optimizer.zero_grad()
            x = created_image
            x = self.model(x)

            loss = - torch.mean(self.conv_output)
            loss.backward()
            optimizer.step()

        grid_name = '{layer} => {filter}'.format(layer=selected_layer, filter=selected_filter)
        print('Logging: {grid_name}'.format(grid_name=grid_name))
        # grid_data = make_grid((created_image - compare_image).expand(-1, 3, -1, -1), normalize=True, scale_each=True)
        grid_data = make_grid((created_image - compare_image).expand(-1, 3, -1, -1))
        # writer.add_image(grid_name, grid_data, n_iter)
        img = grid_data[0].detach().cpu().numpy()  # FloatTensor转为ndarray，选第一个通道

        # 显示图片
        plt.imshow(img)
        plt.savefig("./result/gradient_ascent/" + str(selected_filter) + ".png")
        plt.show()

        # if self.writer is None:
        # writer.close()
