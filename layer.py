from __future__ import absolute_import, division # 导入__future__模块中的绝对导入和除法运算符，使代码与Python 3兼容

import torch # 导入PyTorch模块
import torch.nn as nn # 导入PyTorch中的神经网络模块

import numpy as np # 导入NumPy模块
from deform_conv import th_batch_map_offsets, th_generate_grid # 从deform_conv模块中导入th_batch_map_offsets和th_generate_grid函数

class ConvOffset2D(nn.Conv2d): # 继承自nn.Conv2d的新类ConvOffset2D

    def __init__(self, filters, init_normal_stddev=0.01, **kwargs): # 定义初始化函数
        self.filters = filters # 定义类变量filters
        self._grid_param = None # 定义类变量_grid_param，初始值为None
        super(ConvOffset2D, self).__init__(self.filters, self.filters*2, 3, padding=1, bias=False, **kwargs) # 调用父类构造函数，初始化一个Conv2d卷积层
        self.weight.data.copy_(self._init_weights(self.weight, init_normal_stddev)) # 使用_init_weights方法对卷积核进行初始化

    def forward(self, x): # 定义前向传播函数
        x_shape = x.size() # 获取输入的大小
        offsets = super(ConvOffset2D, self).forward(x) # 调用父类的forward方法，获取偏移量

        # offsets: (b*c, h, w, 2)
        offsets = self._to_bc_h_w_2(offsets, x_shape) # 调用_to_bc_h_w_2方法，将offsets的维度转换为(b*c, h, w, 2)

        # x: (b*c, h, w)
        x = self._to_bc_h_w(x, x_shape) # 调用_to_bc_h_w方法，将x的维度转换为(b*c, h, w)

        # X_offset: (b*c, h, w)
        x_offset = th_batch_map_offsets(x, offsets, grid=self._get_grid(self,x)) # 调用th_batch_map_offsets方法，利用偏移量对x进行采样得到偏移后的x_offset

        # x_offset: (b, h, w, c)
        x_offset = self._to_b_c_h_w(x_offset, x_shape) # 调用_to_b_c_h_w方法，将x_offset的维度转换为(b, h, w, c)

        return x_offset # 返回偏移后的x_offset

    @staticmethod
    def _get_grid(self, x): # 定义静态方法_get_grid，用于获取偏移量的网格
        batch_size, input_height, input_width = x.size(0), x.size(1), x.size(2) # 获取输入x的大小
        dtype, cuda = x.data.type(), x.data.is_cuda # 获取输入x的数据类型和是否在GPU上
        if self._grid_param == (batch_size, input_height, input_width, dtype, cuda): # 如果已经计算过网格，则直接返回之前计算
