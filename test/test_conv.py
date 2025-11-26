import sys 
sys.path.append("../build/Release/")
import mytorch
import torch, numpy as np, unittest

class TestConv2d(unittest.TestCase):
    def test_forward(self):
        a = np.random.rand(2 , 6 , 8 , 8).astype(np.float32)
        mytorch_a = mytorch.tensor_from_numpy(a , mytorch.Cuda)
        torch_a = torch.from_numpy(a)
        mytorch_conv = mytorch.nn.Conv2d(6 , 6 , 5)
        torch_conv = torch.nn.Conv2d(6 , 6 , 5)
        mytorch_conv_out = mytorch_conv(mytorch_a)
        torch_conv_out = torch_conv(torch_a)
        self.assertTrue(np.allclose(mytorch.numpy_from_tensor(mytorch_conv_out), torch_conv_out.cpu().detach().numpy(),atol = 1e-5))