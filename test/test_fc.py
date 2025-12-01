import sys 
sys.path.append("../build/Release/")
import mytorch
import torch, numpy as np, unittest

class TestFC(unittest.TestCase):
    def test_fc(self):
        mytorch.set_default_device(mytorch.Cpu)
        a = np.random.randn(200 , 100).astype(np.float32)
        mytorch_a = mytorch.tensor_from_numpy(a)
        torch_a = torch.from_numpy(a)
        in_features = 100
        out_features = 50
        w = np.random.randn(out_features , in_features).astype(np.float32)
        b = np.random.randn(out_features).astype(np.float32)
        mytorch_fc = mytorch.nn.Linear(in_features , out_features)
        torch_fc = torch.nn.Linear(in_features , out_features)
        mytorch_fc.weight = mytorch.tensor_from_numpy(w)
        torch_fc.weight = torch.nn.Parameter(torch.from_numpy(w))
        mytorch_fc.bias = mytorch.tensor_from_numpy(b)
        torch_fc.bias = torch.nn.Parameter(torch.from_numpy(b))
        mytorch_out = mytorch_fc(mytorch_a)
        torch_out = torch_fc(torch_a)
        np.testing.assert_allclose(mytorch.numpy_from_tensor(mytorch_out) , torch_out.detach().numpy() , atol = 1e-4)
    def test_fc_backward(self):
        mytorch.set_default_device(mytorch.Cpu)
        a = np.random.randn(200 , 100).astype(np.float32)
        mytorch_a = mytorch.tensor_from_numpy(a)
        mytorch_a.set_requires_grad(True)
        torch_a = torch.from_numpy(a)
        torch_a.requires_grad_(True)

        in_features = 100
        out_features = 50
        w = np.random.randn(out_features , in_features).astype(np.float32)
        b = np.random.randn(out_features).astype(np.float32)
        mytorch_fc = mytorch.nn.Linear(in_features , out_features)
        torch_fc = torch.nn.Linear(in_features , out_features)
        mytorch_fc.weight = mytorch.tensor_from_numpy(w)
        torch_fc.weight = torch.nn.Parameter(torch.from_numpy(w))
        mytorch_fc.bias = mytorch.tensor_from_numpy(b)
        torch_fc.bias = torch.nn.Parameter(torch.from_numpy(b))
        mytorch_fc.zero_grad()
        mytorch_out = mytorch_fc(mytorch_a)
        torch_out = torch_fc(torch_a)

        grad = np.random.randn(200 , out_features).astype(np.float32)
        mytorch_grad = mytorch.tensor_from_numpy(grad)
        torch_grad = torch.from_numpy(grad)
        mytorch_out.backward(mytorch_grad)
        torch_out.backward(torch_grad)
        np.testing.assert_allclose(mytorch.numpy_from_tensor(mytorch_a.get_grad_tensor()) , torch_a.grad.numpy() , atol = 1e-4)
        np.testing.assert_allclose(mytorch.numpy_from_tensor(mytorch_fc.weight.get_grad_tensor()) , torch_fc.weight.grad.numpy() , atol = 1e-4)
        np.testing.assert_allclose(mytorch.numpy_from_tensor(mytorch_fc.bias.get_grad_tensor()) , torch_fc.bias.grad.numpy() , atol = 1e-4)
    def test_fc_cuda(self):
        mytorch.set_default_device(mytorch.Cuda)
        a = np.random.randn(200 , 100).astype(np.float32)
        mytorch_a = mytorch.tensor_from_numpy(a)
        torch_a = torch.from_numpy(a)
        in_features = 100
        out_features = 50
        w = np.random.randn(out_features , in_features).astype(np.float32)
        b = np.random.randn(out_features).astype(np.float32)
        mytorch_fc = mytorch.nn.Linear(in_features , out_features)
        torch_fc = torch.nn.Linear(in_features , out_features)
        mytorch_fc.weight = mytorch.tensor_from_numpy(w)
        torch_fc.weight = torch.nn.Parameter(torch.from_numpy(w))
        mytorch_fc.bias = mytorch.tensor_from_numpy(b)
        torch_fc.bias = torch.nn.Parameter(torch.from_numpy(b))
        mytorch_out = mytorch_fc(mytorch_a)
        torch_out = torch_fc(torch_a)
        np.testing.assert_allclose(mytorch.numpy_from_tensor(mytorch_out) , torch_out.detach().numpy() , atol = 1e-4)
    def test_fc_backward_cuda(self):
        mytorch.set_default_device(mytorch.Cuda)
        a = np.random.randn(200 , 100).astype(np.float32)
        mytorch_a = mytorch.tensor_from_numpy(a)
        mytorch_a.set_requires_grad(True)
        torch_a = torch.from_numpy(a)
        torch_a.requires_grad_(True)

        in_features = 100
        out_features = 50
        w = np.random.randn(out_features , in_features).astype(np.float32)
        b = np.random.randn(out_features).astype(np.float32)
        mytorch_fc = mytorch.nn.Linear(in_features , out_features)
        torch_fc = torch.nn.Linear(in_features , out_features)
        mytorch_fc.weight = mytorch.tensor_from_numpy(w)
        torch_fc.weight = torch.nn.Parameter(torch.from_numpy(w))
        mytorch_fc.bias = mytorch.tensor_from_numpy(b)
        torch_fc.bias = torch.nn.Parameter(torch.from_numpy(b))
        mytorch_fc.zero_grad()
        mytorch_out = mytorch_fc(mytorch_a)
        torch_out = torch_fc(torch_a)

        grad = np.random.randn(200 , out_features).astype(np.float32)
        mytorch_grad = mytorch.tensor_from_numpy(grad)
        torch_grad = torch.from_numpy(grad)
        mytorch_out.backward(mytorch_grad)
        torch_out.backward(torch_grad)
        np.testing.assert_allclose(mytorch.numpy_from_tensor(mytorch_a.get_grad_tensor()) , torch_a.grad.numpy() , atol = 1e-4)
        np.testing.assert_allclose(mytorch.numpy_from_tensor(mytorch_fc.weight.get_grad_tensor()) , torch_fc.weight.grad.numpy() , atol = 1e-4)
        np.testing.assert_allclose(mytorch.numpy_from_tensor(mytorch_fc.bias.get_grad_tensor()) , torch_fc.bias.grad.numpy() , atol = 1e-4)



if __name__ == '__main__':
    unittest.main()
