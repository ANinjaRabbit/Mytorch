import sys 
sys.path.append("../build/Release/")
import mytorch
import torch, numpy as np, unittest

class TestPool2d(unittest.TestCase):
    def test_pool2d_cpu(self):
        kernel_shape = [2  , 2]
        a = np.random.rand(1 , 2 , 4 ,4)
        mytorch_a = mytorch.tensor_from_numpy(a)



        pool = mytorch.nn.Pool2d(kernel_shape) 
        out = pool(mytorch_a)
        out_np = mytorch.numpy_from_tensor(out)
        ans_tensor = torch.nn.functional.max_pool2d(torch.from_numpy(a), 2)
        self.assertTrue(np.allclose(out_np, ans_tensor.numpy()))

    def test_pool2d_cuda(self):
        kernel_shape = [2  , 2]
        a = np.random.rand(1 , 2 , 4 ,4)
        mytorch_a = mytorch.tensor_from_numpy(a)
        mytorch_a.to(mytorch.Cuda)



        pool = mytorch.nn.Pool2d(kernel_shape) 
        out = pool(mytorch_a)
        out_np = mytorch.numpy_from_tensor(out)
        ans_tensor = torch.nn.functional.max_pool2d(torch.from_numpy(a), 2)
        self.assertTrue(np.allclose(out_np, ans_tensor.numpy()))

if __name__ == '__main__':
    unittest.main()
