#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import time

#######################################
#            1) Torch部分
#######################################
class MyMLP(nn.Module):
    """
    一个简单的多层感知机：
    in=8 -> hidden1=4 -> hidden2=4 -> out=8
    中间层用 Tanh 激活，最后一层无激活
    """
    def __init__(self, in_dim=8, h1=4, h2=4, out_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.Tanh(),
            nn.Linear(h1, h2),
            nn.Tanh(),
            nn.Linear(h2, out_dim)
        )
    def forward(self, x):
        return self.net(x)

def torch_compute_jvp(mlp, x):
    """
    使用你自定义的 JVP:
      1) 通过 torch.autograd.functional.jacobian(mlp, x) 得到对 x 的完整Jacobian
      2) 按你的方式: sum->transpose->einsum
         jac: [B, out_dim, B, in_dim]
         => sum over dim=2 => [B, out_dim, in_dim]
         => transpose => [B, in_dim, out_dim]
         => einsum with mlp(x).unsqueeze(1)
    """
    def forward_fn(xx):
        return mlp(xx)

    # 计算 wrt x 的 Jacobian
    jac = torch.autograd.functional.jacobian(forward_fn, x)
    # [B, out_dim, B, in_dim]
    jac_summed = torch.sum(jac, dim=2)   # => [B, out_dim, in_dim]
    jac_t = jac_summed.transpose(2, 1)   # => [B, in_dim, out_dim]

    out = mlp(x)  # => [B, out_dim]
    # => (B,1,out_dim) x (B,in_dim,out_dim) => (B,in_dim)
    jvp = torch.einsum('bij,bjk->bik', out.unsqueeze(1), jac_t).squeeze(1)
    return jvp

#######################################
#            2) Taichi部分
#######################################
import taichi as ti

ti.init(arch=ti.cpu, debug=False)  # 如需GPU可改 arch=ti.gpu

@ti.kernel
def mlp_forward():
    """
    三层 MLP： (8->4->4->8)，中间层 tanh，最后不加激活
    """
    #y[b_] = ti.tanh(W[None] @ x[b_] + bias[None])
    for b in range(B):
        # layer1
        h1 =ti.tanh(W1[None] @ x[b] + b1[None])
        # layer2
        h2 = ti.tanh(W2[None] @ h1 + b2[None])
        # layer3
        out = W3[None] @ h2 + b3[None]
        y[b] = out

@ti.kernel
def clear_grad():
    for b in range(B):
        x.grad[b].fill(0.0)
        y.grad[b].fill(0.0)

@ti.kernel
def set_v_grad():
    for b in range(B):
        y.grad[b] = v[b]

@ti.kernel
def copy_jvp():
    for b in range(B):
        JVP_result[b] = x.grad[b]

def taichi_compute_jvp():
    """
    清空梯度 -> 设置 v -> 前向 -> 反向 -> 拷贝梯度
    """
    clear_grad()
    set_v_grad()
    mlp_forward()
    mlp_forward.grad()
    copy_jvp()
    return JVP_result.to_numpy()

#######################################
#       3) 主流程: 初始化 + 测试
#######################################
if __name__ == '__main__':
    # 超参数
    B       = 3  # batch_size
    in_dim  = 8
    h1dim   = 4
    h2dim   = 4
    out_dim = 8

    # 1) 先构建 Torch MLP
    net_torch = MyMLP(in_dim=in_dim, h1=h1dim, h2=h2dim, out_dim=out_dim)
    net_torch.train()

    # 随机初始化(或你也可给定 manual_seed 保持一致)
    # 这里以 net_torch 自带随机初始化为准
    # 读出网络参数 (权重,偏置)
    W1_np = net_torch.net[0].weight.detach().numpy()  # => shape=(4,8)
    b1_np = net_torch.net[0].bias.detach().numpy()    # => (4,)
    W2_np = net_torch.net[2].weight.detach().numpy()  # => (4,4)
    b2_np = net_torch.net[2].bias.detach().numpy()    # => (4,)
    W3_np = net_torch.net[4].weight.detach().numpy()  # => (8,4)
    b3_np = net_torch.net[4].bias.detach().numpy()    # => (8,)

    # 2) Taichi端创建同样的字段
    W1 = ti.Matrix.field(h1dim, in_dim,  ti.f32, shape=())
    b1 = ti.Vector.field(h1dim,         ti.f32, shape=())
    W2 = ti.Matrix.field(h2dim, h1dim,  ti.f32, shape=())
    b2 = ti.Vector.field(h2dim,         ti.f32, shape=())
    W3 = ti.Matrix.field(out_dim, h2dim, ti.f32, shape=())
    b3 = ti.Vector.field(out_dim,       ti.f32, shape=())

    # 把权重写入Taichi
    W1[None] = W1_np
    b1[None] = b1_np
    W2[None] = W2_np
    b2[None] = b2_np
    W3[None] = W3_np
    b3[None] = b3_np

    # 输入 x: (B,8), 输出 y: (B,8)
    x = ti.Vector.field(in_dim,  ti.f32, shape=(B,), needs_grad=True)
    y = ti.Vector.field(out_dim, ti.f32, shape=(B,), needs_grad=True)

    # JVP 的向量: v: (B,8), 以及结果: (B,8)
    v = ti.Vector.field(out_dim, ti.f32, shape=(B,))
    JVP_result = ti.Vector.field(in_dim,  ti.f32, shape=(B,))

    # 3) 构造输入, 以及 v
    torch.manual_seed(42)
    # 让 Torch + Taichi 的输入匹配
    imgs_np = np.random.randn(B, in_dim).astype(np.float32)

    # 把同样的 imgs_np 写入 torch
    x_torch = torch.tensor(imgs_np, dtype=torch.float32, requires_grad=True)
    # 再赋给 Taichi
    for b_ in range(B):
        x[b_] = imgs_np[b_]

    # 让向量 v 来自 net_torch(x_torch) 的输出
    out_t = net_torch(x_torch).detach().numpy()  # shape=[B,8]
    for b_ in range(B):
        v[b_] = out_t[b_]

    # 4) 先测试 Torch 的 JVP
    torch_start = time.time()
    torch_jvp = torch_compute_jvp(net_torch, x_torch)
    torch_time = time.time() - torch_start

    # 5) 测试 Taichi 的 JVP
    #  先预热一次 forward
    mlp_forward()
    taichi_start = time.time()
    taichi_jvp = taichi_compute_jvp()  # => shape=[B,8]
    taichi_time = time.time() - taichi_start

    # 6) 比对结果
    max_err = np.max(np.abs(taichi_jvp - torch_jvp.detach().numpy()))
    print(f'\n===== Compare JVP (batch={B}, in=8, out=8) =====')
    print('Taichi JVP shape:', taichi_jvp.shape, ', Torch JVP shape:', torch_jvp.shape)
    print(f'Max abs diff = {max_err:.3e}')
    print('Taichi JVP sample0[:5]:', taichi_jvp[0, :5])
    print('Torch  JVP sample0[:5]:', torch_jvp.detach().numpy()[0, :5])

    # 7) 对比时间
    print('\n===== Speed =====')
    print(f'Torch:  {torch_time*1000:.2f} ms')
    print(f'Taichi: {taichi_time*1000:.2f} ms')
    ratio = torch_time / taichi_time if taichi_time>1e-9 else 9999
    print(f'Speed ratio (Torch / Taichi) = {ratio:.2f}x')
