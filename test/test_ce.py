import sys 
sys.path.append("../build/Release/")
import mytorch
import torch, numpy as np, unittest

class TestCrossEntropy(unittest.TestCase):
    def test_ce(self):
        mytorch.set_default_device(mytorch.Cpu)
        a = np.random.rand(32 , 10).astype(np.float32)
        mytorch_a = mytorch.tensor_from_numpy(a)
        torch_a = torch.from_numpy(a)


        labels = np.random.randint(0, 10, (32,)).astype(np.int64)
        mytorch_labels = mytorch.tensor_from_numpy(labels.astype(np.float32))
        torch_labels = torch.from_numpy(labels)

        cem = mytorch.nn.CrossEntropy()
        torch_cem = torch.nn.CrossEntropyLoss()

        mytorch_loss = cem(mytorch_a, mytorch_labels)
        torch_loss = torch_cem(torch_a, torch_labels)
        self.assertAlmostEqual(mytorch_loss.item(), torch_loss.item(), places=5)
    def test_ce_grad(self):
        mytorch.set_default_device(mytorch.Cpu)
        a = np.random.rand(32 , 10).astype(np.float32)
        mytorch_a = mytorch.tensor_from_numpy(a)
        torch_a = torch.from_numpy(a)
        mytorch_a.set_requires_grad(True)
        torch_a.requires_grad_(True)

        labels = np.random.randint(0, 10, (32,)).astype(np.int64)
        mytorch_labels = mytorch.tensor_from_numpy(labels.astype(np.float32))
        torch_labels = torch.from_numpy(labels)

        cem = mytorch.nn.CrossEntropy()
        torch_cem = torch.nn.CrossEntropyLoss()

        mytorch_loss = cem(mytorch_a, mytorch_labels)
        torch_loss = torch_cem(torch_a, torch_labels)
        mytorch_loss.backward()
        torch_loss.backward()
        self.assertTrue(np.allclose(mytorch.numpy_from_tensor(mytorch_a.get_grad_tensor()), torch_a.grad, atol=1e-4))
    def test_ce_cuda(self):
        mytorch.set_default_device(mytorch.Cuda)
        a = np.random.rand(32 , 10).astype(np.float32)
        mytorch_a = mytorch.tensor_from_numpy(a)
        torch_a = torch.from_numpy(a)


        labels = np.random.randint(0, 10, (32,)).astype(np.int64)
        mytorch_labels = mytorch.tensor_from_numpy(labels.astype(np.float32))
        torch_labels = torch.from_numpy(labels)

        cem = mytorch.nn.CrossEntropy()
        torch_cem = torch.nn.CrossEntropyLoss()

        mytorch_loss = cem(mytorch_a, mytorch_labels)
        torch_loss = torch_cem(torch_a, torch_labels)
        self.assertAlmostEqual(mytorch_loss.item(), torch_loss.item(), places=5)
    def test_ce_grad_cuda(self):
        mytorch.set_default_device(mytorch.Cuda)
        a = np.random.rand(32 , 10).astype(np.float32)
        mytorch_a = mytorch.tensor_from_numpy(a)
        torch_a = torch.from_numpy(a)
        mytorch_a.set_requires_grad(True)
        torch_a.requires_grad_(True)

        labels = np.random.randint(0, 10, (32,)).astype(np.int64)
        mytorch_labels = mytorch.tensor_from_numpy(labels.astype(np.float32))
        torch_labels = torch.from_numpy(labels)

        cem = mytorch.nn.CrossEntropy()
        torch_cem = torch.nn.CrossEntropyLoss()

        mytorch_loss = cem(mytorch_a, mytorch_labels)
        torch_loss = torch_cem(torch_a, torch_labels)
        mytorch_loss.backward()
        torch_loss.backward()
        self.assertTrue(np.allclose(mytorch.numpy_from_tensor(mytorch_a.get_grad_tensor()), torch_a.grad, atol=1e-4))
    
if __name__ == '__main__':
    mytorch.set_default_device(mytorch.Cuda)
    unittest.main()
