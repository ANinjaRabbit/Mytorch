import sys 
sys.path.append("../build/Release/")
import mytorch
import torch, numpy as np, unittest

class TestFlatten(unittest.TestCase):
    def test_flatten(self):
        a = np.random.rand(32 , 3 , 32 , 32).astype(np.float32)

        mytorch_a = mytorch.tensor_from_numpy(a)
        torch_a = torch.from_numpy(a)

        mytorch_flatten = mytorch.nn.Flatten(start_dim = 1)
        torch_flatten = torch.nn.Flatten(start_dim = 1)

        mytorch_b = mytorch_flatten(mytorch_a)
        torch_b = torch_flatten(torch_a)

        self.assertTrue(np.allclose( mytorch.numpy_from_tensor(mytorch_b) , torch_b.detach().numpy() , atol=1e-3))
    def test_flatten_backward(self):
        a = np.random.rand(32 , 3 , 32 , 32).astype(np.float32)

        mytorch_a = mytorch.tensor_from_numpy(a)
        torch_a = torch.from_numpy(a)
        mytorch_a.set_requires_grad(True)
        torch_a.requires_grad_(True)

        mytorch_flatten = mytorch.nn.Flatten(start_dim = 1)
        torch_flatten = torch.nn.Flatten(start_dim = 1)

        mytorch_b = mytorch_flatten(mytorch_a)
        torch_b = torch_flatten(torch_a)

        grad = np.random.rand(32 , 32 * 32 * 3).astype(np.float32)
        mytorch_grad = mytorch.tensor_from_numpy(grad)
        torch_grad = torch.from_numpy(grad)

        mytorch_b.backward(mytorch_grad)
        torch_b.backward(torch_grad)

        self.assertTrue(np.allclose( mytorch.numpy_from_tensor(mytorch_a.get_grad_tensor()) , torch_a.grad.numpy() , atol=1e-3))


if __name__ == '__main__':
    mytorch.set_default_device(mytorch.Cuda)

    unittest.main()