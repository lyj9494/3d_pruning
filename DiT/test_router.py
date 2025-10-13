import torch
import torch.nn as nn

class Router(nn.Module):
    def __init__(self, num_choices):
        super().__init__()
        self.prob = nn.Parameter(torch.randn(num_choices), requires_grad=True)
        self.activation = nn.Sigmoid()

    def forward(self, x=None):
        return self.activation(self.prob)

class RouterNoActivation(nn.Module):
    def __init__(self, num_choices):
        super().__init__()
        self.prob = nn.Parameter(torch.randn(num_choices), requires_grad=True)

    def forward(self, x=None):
        return self.prob

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 测试 Router 类
router = Router(num_choices=1000).to(device)
torch.cuda.reset_peak_memory_stats()
print("Router with activation:")
output = router()
output.backward(torch.ones_like(output))
print(f"Memory allocated after forward and backward: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

# 测试 RouterNoActivation 类
router_no_activation = RouterNoActivation(num_choices=1000).to(device)
torch.cuda.reset_peak_memory_stats()
print("\nRouter without activation:")
output_no_activation = router_no_activation()
output_no_activation.backward(torch.ones_like(output_no_activation))
print(f"Memory allocated after forward and backward: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")