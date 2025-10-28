import sys 
sys.path.append("../build/Release/")
import mytorch
import torch, numpy as np, unittest

class TestCE_with_softmax(unittest.TestCase):
    def test_softmax_cpu(self):
        a = np.random.rand(8 ,4)
        mytorch_a = mytorch.tensor_from_numpy(a)
        softmax = mytorch.nn.Softmax()
        out = softmax(mytorch_a)
        out_np = mytorch.numpy_from_tensor(out)
        ans_tensor = torch.nn.functional.softmax(torch.from_numpy(a), dim=1)
        self.assertTrue(np.allclose(out_np, ans_tensor.numpy()))

    def test_ce_cpu(self):
        a = np.random.rand(8 ,4)
        labels = np.random.randint(0, 4, size=(8,))
        mytorch_a = mytorch.tensor_from_numpy(a)
        ce = mytorch.nn.CrossEntropy(labels)
        out = ce(mytorch_a)
        out_np = mytorch.numpy_from_tensor(out)
        tensor_labels = torch.from_numpy(labels).to(int)
        ans_tensor = torch.nn.functional.cross_entropy(torch.from_numpy(a), tensor_labels)
        self.assertTrue(np.allclose(out_np, ans_tensor.numpy()))

    def test_softmax_cuda(self):
        a = np.random.rand(8 ,4)
        mytorch_a = mytorch.tensor_from_numpy(a)
        mytorch_a.to(mytorch.Cuda)

        softmax = mytorch.nn.Softmax()
        out = softmax(mytorch_a)
        out_np = mytorch.numpy_from_tensor(out)
        ans_tensor = torch.nn.functional.softmax(torch.from_numpy(a), dim=1)
        self.assertTrue(np.allclose(out_np, ans_tensor.numpy()))

    def test_ce_cuda(self):
        a = np.random.rand(8 ,4)
        labels = np.random.randint(0, 4, size=(8,))
        mytorch_a = mytorch.tensor_from_numpy(a)
        mytorch_a.to(mytorch.Cuda)
        ce = mytorch.nn.CrossEntropy(labels)
        out = ce(mytorch_a)
        out_np = mytorch.numpy_from_tensor(out)
        tensor_labels = torch.from_numpy(labels).to(int)
        ans_tensor = torch.nn.functional.cross_entropy(torch.from_numpy(a), tensor_labels)
        self.assertTrue(np.allclose(out_np, ans_tensor.numpy()))


if __name__ == '__main__':
    unittest.main()
