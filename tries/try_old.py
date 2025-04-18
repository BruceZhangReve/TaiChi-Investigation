import torch
import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

# ------------------ 1. Taichi Fields & Kernels ------------------
# 为简化，这里设置最大维度(比如 n<=1024, m<=1024)，以便初始化固定大小 field
n_max = 1024
m_max = 1024

# 注意：为了示例，我们把W/b都放到一个大的Field里; 实际使用时可以灵活设计
W_ti = ti.field(ti.f32, shape=(m_max, n_max), needs_grad=True)  
b_ti = ti.field(ti.f32, shape=(m_max,), needs_grad=True)

x_ti = ti.field(ti.f32, shape=(n_max,), needs_grad=True)
y_ti = ti.field(ti.f32, shape=(m_max,), needs_grad=True)

@ti.kernel
def forward_tanh(m: ti.i32, n: ti.i32):
    for i in range(m):
        tmp = 0.0
        for j in range(n):
            tmp += W_ti[i, j] * x_ti[j]
        tmp += b_ti[i]
        y_ti[i] = ti.tanh(tmp)

# --------------- 2. PyTorch 自定义 Function ---------------
class TaichiTanhLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_torch, W_torch, b_torch):
        """
        x_torch: (n,)
        W_torch: (m,n)
        b_torch: (m,)
        return:  (m,)
        """
        # 1) 将数据从 PyTorch -> NumPy -> Taichi
        x_np = x_torch.detach().cpu().numpy()
        W_np = W_torch.detach().cpu().numpy()
        b_np = b_torch.detach().cpu().numpy()

        n = x_np.shape[0]
        m = W_np.shape[0]

        # 把 x/W/b 写进 Taichi
        # 注意 x_np 的 shape: (n,)
        for i in range(n):
            x_ti[i] = x_np[i]
        # W_np shape: (m,n)
        for i in range(m):
            for j in range(n):
                W_ti[i, j] = W_np[i, j]
        # b_np shape: (m,)
        for i in range(m):
            b_ti[i] = b_np[i]

        # 2) 前向
        forward_tanh(m, n)
        
        # 3) 拷出 y
        y_np = np.zeros((m,), dtype=np.float32)
        for i in range(m):
            y_np[i] = y_ti[i]

        # 4) 在 ctx 中保存需要在 backward 时使用的尺寸和 CPU 数组大小
        ctx.m = m
        ctx.n = n

        # 注意：还可以把 x/W/b 也存到 ctx 里，但这里演示完备做法则需要额外内存。
        #      如果在 backward 里要重新从外部再拷一次，也可以(看需求)。
        # 这里演示比较“显式”的方案，把 x_np, W_np, b_np 也存到 ctx
        # 以便 backward 时做“二次拷贝” (或者完全可以再 forward 里不detach, 直接 ctx.save_for_backward(...) 也行)
        ctx.save_for_backward(x_torch, W_torch, b_torch)

        # 5) 返回一个 torch.Tensor
        y_torch = torch.from_numpy(y_np).to(x_torch.device)
        return y_torch

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: shape (m,)
        return (grad_x, grad_W, grad_b)
        """
        # 取回 forward 时保存的部分
        x_torch, W_torch, b_torch = ctx.saved_tensors
        m, n = ctx.m, ctx.n

        # 1) 将 grad_output 拷入 Taichi, 触发 y_ti.grad
        grad_out_np = grad_output.detach().cpu().numpy()
        # 先清空 Taichi 的 grad
        for i in range(n):
            x_ti.grad[i] = 0.0
        for i in range(m):
            y_ti.grad[i] = 0.0
            for j in range(n):
                W_ti.grad[i, j] = 0.0
            b_ti.grad[i] = 0.0

        # 赋值
        for i in range(m):
            y_ti.grad[i] = grad_out_np[i]

        # 2) 反向传播
        forward_tanh.grad(m, n)

        # 3) 从 Taichi 中拷回 grad_x, grad_W, grad_b
        grad_x_np = np.zeros((n,), dtype=np.float32)
        grad_W_np = np.zeros((m, n), dtype=np.float32)
        grad_b_np = np.zeros((m,), dtype=np.float32)

        for i in range(n):
            grad_x_np[i] = x_ti.grad[i]
        for i in range(m):
            grad_b_np[i] = b_ti.grad[i]
            for j in range(n):
                grad_W_np[i, j] = W_ti.grad[i, j]

        grad_x_torch = torch.from_numpy(grad_x_np).to(x_torch.device)
        grad_W_torch = torch.from_numpy(grad_W_np).to(W_torch.device)
        grad_b_torch = torch.from_numpy(grad_b_np).to(b_torch.device)

        # 4) 返回与 forward 参数对应的梯度
        return grad_x_torch, grad_W_torch, grad_b_torch

# ------------------ 3. 测试：用 PyTorch 调用这个自定义算子 ------------------

if __name__ == "__main__":
    # 准备一些随机数据
    n = 4
    m = 3
    x_pth = torch.randn(n, requires_grad=True)
    W_pth = torch.randn(m, n, requires_grad=True)
    b_pth = torch.randn(m, requires_grad=True)

    # 用我们自定义的Function
    y_pth = TaichiTanhLinear.apply(x_pth, W_pth, b_pth)  # shape (m,)

    # 定义个loss看看
    loss = y_pth.sum()
    loss.backward()

    # 查看梯度
    print("Forward y:", y_pth)
    print("x.grad:", x_pth.grad)
    print("W.grad:", W_pth.grad)
    print("b.grad:", b_pth.grad)

    # 也可对比一下 PyTorch 自己算的 tanh(Wx+b)
    # （这里只是对比，不一定要 exact 一样，因为我们自己写的 kernel 可能写错）
    # 不过如果确实是 tanh(Wx+b)，应该能一致
    with torch.no_grad():
        y_ref = torch.tanh(W_pth @ x_pth + b_pth)
    print("Compare with direct PyTorch tanh(Wx+b):", y_ref)
