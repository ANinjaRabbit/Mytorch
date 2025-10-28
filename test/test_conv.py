import sys 
sys.path.append("../build/Release/")
import mytorch
import torch, numpy as np, unittest

class TestConv(unittest.TestCase):
    def test_conv_cpu(self):
        kernel = np.random.rand(1 , 2 , 3, 3)
        a = np.random.rand(1 , 2 , 4 ,4)
        mytorch_kernel = mytorch.tensor_from_numpy(kernel)
        mytorch_a = mytorch.tensor_from_numpy(a)



        conv = mytorch.nn.Conv(mytorch_kernel) 
        out = conv(mytorch_a)
        out_np = mytorch.numpy_from_tensor(out)

        ans_tensor = torch.nn.functional.conv2d(torch.from_numpy(a), torch.from_numpy(kernel) , padding=1)
        self.assertTrue(np.allclose(out_np, ans_tensor.numpy()))
    def test_conv_cuda(self):
        kernel = np.random.rand(1 , 2 , 3, 3)
        a = np.random.rand(1 , 2 , 4 ,4)
        mytorch_kernel = mytorch.tensor_from_numpy(kernel)
        mytorch_a = mytorch.tensor_from_numpy(a)
        mytorch_a.to(mytorch.Cuda)



        conv = mytorch.nn.Conv(mytorch_kernel) 
        out = conv(mytorch_a)
        out_np = mytorch.numpy_from_tensor(out)

        ans_tensor = torch.nn.functional.conv2d(torch.from_numpy(a), torch.from_numpy(kernel) , padding=1)
        self.assertTrue(np.allclose(out_np, ans_tensor.numpy()))
    


if __name__ == '__main__':
    unittest.main()
