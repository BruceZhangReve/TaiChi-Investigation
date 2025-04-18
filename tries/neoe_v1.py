import taichi as ti
import numpy as np
import torch
import time

ti.init(arch=ti.cpu)  

# ------------------ Hyperparameter Setting ------------------
B = 10   # batch size
n = 32   # input dimension D
m = 16   # output dimension D'

# ------------------ Taichi object defined ------------------
W = ti.Matrix.field(m, n, dtype=ti.f32, shape=())
bias = ti.Vector.field(m, dtype=ti.f32, shape=())
x = ti.Vector.field(n, dtype=ti.f32, shape=(B,), needs_grad=True)
y = ti.Vector.field(m, dtype=ti.f32, shape=(B,), needs_grad=True)
v = ti.Vector.field(m, dtype=ti.f32, shape=(B,))
JVP_result = ti.Vector.field(n, dtype=ti.f32, shape=(B,))

# ------------------ Taichi Kernels ------------------
@ti.kernel
def compute_y_batch():
    for b_ in ti.ndrange(B):
        y[b_] = ti.tanh(W[None] @ x[b_] + bias[None])

@ti.kernel
def clear_gradients():
    for b_ in ti.ndrange(B):
        x.grad[b_].fill(0.0)
        y.grad[b_].fill(0.0)

@ti.kernel
def set_v_grad():
    for b_ in ti.ndrange(B):
        y.grad[b_] = v[b_]

@ti.kernel
def copy_grad_to_result():
    for b_ in ti.ndrange(B):
        JVP_result[b_] = x.grad[b_]

def taichi_compute_jvp():
    # set gradients to 0
    clear_gradients()
    
    # 设置梯度为v
    set_v_grad()
    
    # When using `kernel.grad()`, it is recommended that you always run forward kernel before backward, 
    # for example `kernel()`; `kernel.grad()`
    compute_y_batch()
    compute_y_batch.grad()
    
    # 拷贝结果
    copy_grad_to_result()
    return JVP_result.to_numpy()

# ------------------ PyTorch计算流程 ------------------
def pytorch_compute_jvp(x_torch, W_torch, b_torch, v_torch):
    def forward_fn(x_):
        return torch.tanh(x_ @ W_torch.T + b_torch)
    
    # 计算Jacobian
    jac = torch.autograd.functional.jacobian(forward_fn, x_torch)
    jac = jac.transpose(1, 2).diagonal(dim1=0, dim2=1).movedim(-1, 0)
    
    # 计算J^T v
    return torch.einsum('bmn,bm->bn', jac, v_torch).numpy()

if __name__ == "__main__":
    # ------------------ 初始化数据 ------------------
    torch.manual_seed(42)
    
    # 生成数据
    W_np = torch.randn(m, n, dtype=torch.float32).numpy()
    bias_np = torch.randn(m, dtype=torch.float32).numpy()
    x_np = torch.randn(B, n, dtype=torch.float32).numpy()
    v_np = torch.randn(B, m, dtype=torch.float32).numpy()

    # 加载数据到Taichi
    W[None] = W_np
    bias[None] = bias_np
    for b_ in range(B):
        x[b_] = x_np[b_]
        v[b_] = v_np[b_]
    
    # ------------------ 预热运行 ------------------
    _ = taichi_compute_jvp()  # 预热Taichi
    x_torch = torch.tensor(x_np, dtype=torch.float32)  # 预热PyTorch

    # ------------------ 正式测速 ------------------
    # Taichi计算
    taichi_start = time.time()
    taichi_res = taichi_compute_jvp()
    taichi_time = time.time() - taichi_start

    # PyTorch计算
    W_torch = torch.tensor(W_np, dtype=torch.float32)
    b_torch = torch.tensor(bias_np, dtype=torch.float32)
    v_torch = torch.tensor(v_np, dtype=torch.float32)
    
    torch_start = time.time()
    torch_res = pytorch_compute_jvp(x_torch, W_torch, b_torch, v_torch)
    torch_time = time.time() - torch_start

    # ------------------ 结果验证 ------------------
    max_error = np.max(np.abs(taichi_res - torch_res))
    print("\n================ 验证结果 ================")
    print(f"最大绝对误差: {max_error:.2e}")
    print(f"Taichi 结果样本0前5维: {taichi_res[0, :5]}")
    print(f"PyTorch结果样本0前5维: {torch_res[0, :5]}")

    # ------------------ 性能对比 ------------------
    print("\n================ 性能对比 ================")
    print(f"Taichi 计算时间: {taichi_time * 1000:.2f} ms")
    print(f"PyTorch 计算时间: {torch_time * 1000:.2f} ms")
    print(f"速度比率: {torch_time/taichi_time:.2f}x")