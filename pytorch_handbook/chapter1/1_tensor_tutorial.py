# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:    1_tensor_tutorial
   Description:
   Author:       wangchun
   date:         2020/4/11
-------------------------------------------------
"""
from __future__ import print_function
import torch
import numpy as np

if __name__ == "__main__":
    x = torch.empty(5, 3)
    x = torch.rand(5, 3)
    x = torch.zeros(5, 3, dtype=torch.long)
    x = torch.tensor([5.5, 3])
    x = x.new_ones(5, 3, dtype=torch.double)  # new_* 方法来创建对象
    print(x)
    x = torch.randn_like(x, dtype=torch.float)  # 覆盖 dtype!
    print(x)  # 对象的size 是相同的，只是值和类型发生了变化
    print(x.size())
    y = torch.rand(5, 3)
    print(x + y)
    print(torch.add(x, y))
    result = torch.empty(5, 3)
    torch.add(x, y, out=result)
    print(result)
    y.add_(x)  # y = y+x
    print(y)

    print(x[:, 1])

    a = torch.ones(5)
    b = a.numpy()     # tensor to numpy
    print(b)

    a = np.ones(5)
    b = torch.from_numpy(a)   # numpy to tensor

    # is_available 函数判断是否有cuda可以使用
    # ``torch.device``将张量移动到指定的设备中
    if torch.cuda.is_available():
        device = torch.device("cuda")  # a CUDA 设备对象
        y = torch.ones_like(x, device=device)  # 直接从GPU创建张量
        x = x.to(device)  # 或者直接使用``.to("cuda")``将张量移动到cuda中
        z = x + y
        print(z)
        print(z.to("cpu", torch.double))  # ``.to`` 也会对变量的类型做更改