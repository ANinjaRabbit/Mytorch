import torch
import sys
sys.path.append("build/Release/")
import mytorch

avg_std_my = 0
avg_mean_my = 0
avg_std_torch = 0
avg_mean_torch = 0


for _ in range(1000):
    conv1 = mytorch.nn.Conv2d(2 , 2 , 3 )
    conv2 = torch.nn.Conv2d(2 , 2 , 3 )
    a = mytorch.numpy_from_tensor(conv1.parameters()[0])
    b = conv2.weight
    avg_std_my += a.std().item()
    avg_mean_my += a.mean().item()
    avg_std_torch += b.std().item()
    avg_mean_torch += b.mean().item()

print("MyTorch:")
print("Std: %f" % (avg_std_my / 1000))
print("Mean: %f" % (avg_mean_my / 1000))
print("Torch:")
print("Std: %f" % (avg_std_torch / 1000))
print("Mean: %f" % (avg_mean_torch / 1000))
