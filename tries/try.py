import taichi as ti
import numpy as np
import torch
import time

# ------------------ Hyperparameter Setting ------------------
B = 5   # batch size
n = 16  # input dimension D
m = 8   # output dimension D'

device = 'cpu'

# ------------------ Device Initialization ------------------
if device == 'gpu':
    ti.init(arch=ti.gpu)
    torch_device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
else:
    ti.init(arch=ti.cpu)
    torch_device = torch.device("cpu")

# -------------- Taichi Fields --------------
# Instead of: ti.Matrix.field(m, n, ...)
# we store W as a 2D field of shape (m, n).
W    = ti.field(dtype=ti.f32, shape=(m, n))
bias = ti.field(dtype=ti.f32, shape=(m,))

x = ti.Vector.field(n, dtype=ti.f32, shape=(B,), needs_grad=True)  # (B, n)
y = ti.Vector.field(m, dtype=ti.f32, shape=(B,), needs_grad=True)  # (B, m)
v = ti.Vector.field(m, dtype=ti.f32, shape=(B,))                   # (B, m)
JVP_result = ti.Vector.field(n, dtype=ti.f32, shape=(B,))          # (B, n)

# ------------ Taichi Functions & Kernels ------------
@ti.func
def matvec(W_field, x_vec):
    """Compute W*x for a single sample, 
       with W_field shape=(m, n) and x_vec shape=(n,)."""
    out = ti.Vector.zero(ti.f32, m)
    for i in range(m):
        accum = 0.0
        for j in range(n):
            accum += W_field[i, j] * x_vec[j]
        out[i] = accum
    return out

@ti.kernel
def compute_y_batch():
    # For each sample b in [0..B),
    # do out = tanh(W*x[b] + bias).
    for b_ in range(B):
        out = matvec(W, x[b_])
        for i in range(m):
            out[i] += bias[i]
        y[b_] = ti.tanh(out + bias)

@ti.kernel
def clear_gradients():
    # Reset gradient fields
    for b_ in range(B):
        x.grad[b_].fill(0.0)
        y.grad[b_].fill(0.0)

@ti.kernel
def set_v_grad():
    # Set y.grad = v
    for b_ in range(B):
        y.grad[b_] = v[b_]

@ti.kernel
def copy_grad_to_result():
    # Copy x.grad into JVP_result for final read
    for b_ in range(B):
        JVP_result[b_] = x.grad[b_]

def taichi_compute_jvp():
    clear_gradients()
    set_v_grad()
    compute_y_batch()
    compute_y_batch.grad()
    copy_grad_to_result()
    return JVP_result.to_numpy()

# ------------------ PyTorch Implementation ------------------
def pytorch_compute_jvp(x_torch, W_torch, b_torch, v_torch):
    """Compute JVP by building the full Jacobian, then J^T * v."""
    def forward_fn(x_):
        return torch.tanh(x_ @ W_torch.T + b_torch)

    # jac: shape = [B, m, B, n] after this call
    jac = torch.autograd.functional.jacobian(forward_fn, x_torch)
    # We need the diagonal blocks => shape [B, m, n]
    # Then compute (J^T) * v
    jac = jac.transpose(1, 2).diagonal(dim1=0, dim2=1).movedim(-1, 0)  
    return torch.einsum('bmn,bm->bn', jac, v_torch).cpu().numpy()

# ------------------ Main: Compare & Measure ------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    # 1) Create random parameter data
    W_np    = torch.randn(m, n, dtype=torch.float32)
    bias_np = torch.randn(m,     dtype=torch.float32)
    x_np    = torch.randn(B, n,  dtype=torch.float32)
    v_np    = torch.randn(B, m,  dtype=torch.float32)

    # 2) Transfer to Taichi
    for i in range(m):
        for j in range(n):
            W[i, j] = W_np[i, j]
    for i in range(m):
        bias[i] = bias_np[i]

    for b_ in range(B):
        x[b_] = x_np[b_].numpy()
        v[b_] = v_np[b_].numpy()

    # Pre-warm
    _ = taichi_compute_jvp()

    # 3) Taichi JVP
    taichi_start = time.time()
    taichi_res = taichi_compute_jvp()
    taichi_time = time.time() - taichi_start

    # 4) PyTorch side
    x_torch = x_np.clone().to(torch_device)
    W_torch = W_np.clone().to(torch_device)
    b_torch = bias_np.clone().to(torch_device)
    v_torch = v_np.clone().to(torch_device)

    torch_start = time.time()
    torch_res = pytorch_compute_jvp(x_torch, W_torch, b_torch, v_torch)
    torch_time = time.time() - torch_start

    # 5) Compare results
    max_error = np.max(np.abs(taichi_res - torch_res))
    print("\n================ Results Verification ================")
    print(f"Maximum Absolute Error: {max_error:.2e}")
    print(f"Taichi JVP Result, sample[0][:5]: {taichi_res[0, :5]}")
    print(f"PyTorch JVP Result, sample[0][:5]: {torch_res[0, :5]}")

    print("\n================ Speed Comparison ================")
    print(f"Taichi Computation Time: {taichi_time * 1000:.2f} ms")
    print(f"PyTorch Computation Time: {torch_time * 1000:.2f} ms")
    ratio = torch_time / taichi_time if taichi_time > 1e-9 else 9999
    print(f"Speed Ratio (PyTorch / Taichi) = {ratio:.2f}x")
