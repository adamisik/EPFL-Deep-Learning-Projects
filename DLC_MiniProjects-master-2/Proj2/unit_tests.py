import torch
import torch.nn as nn

import unittest

from framework.layers import Linear, Sequential
from framework.activations import ReLU, Sigmoid, Tanh, SELU
from framework.losses import LossMSE
from framework.optimizers import SGD


class DeepLearningFrameworkTests(unittest.TestCase):

    # Linear
    def test_linear_dimensions(self):
        input = torch.empty((50, 5)).normal_()
        layer = Linear(in_dim=5, out_dim=10)
        out = layer.forward(input)
        self.assertEqual(
            out.shape,
            (50, 10)
        )

    def test_params_of_linear(self):
        linear = Linear(in_dim=5, out_dim=10)
        self.assertEqual(
            len(linear.params()),
            2  # weights + bias
        )

    def test_linear_forward_backward(self):
        # For this test it's important
        torch.set_grad_enabled(True)

        linear = Linear(in_dim=5, out_dim=3, initialization="standard")
        torch_linear = nn.Linear(5, 3)

        torch_linear.weight = nn.Parameter(linear.weights, requires_grad=True)
        torch_linear.bias = nn.Parameter(linear.bias, requires_grad=True)

        t = torch.tensor([[5., 3., 6., 8., 3.],
                          [4., 6., 1., 11, 5.]], requires_grad=True)

        out = linear.forward(t)
        torch_out = torch_linear(t)

        self.assertTrue(torch.allclose(out, torch_out),
                        msg="Linear.forward() gives unexpected results.\n"
                            "Expected: " + str(torch_out) + ", and got " + str(out))

        loss = torch_out.sum()
        gradwrtoutput = torch.empty((t.shape[0], linear.out_dim)).fill_(1)  # grad (x+y+z) wrt (x, y, z) is (1, 1, 1)

        loss.backward()

        res = linear.backward(gradwrtoutput)

        self.assertTrue(torch.allclose(t.grad, res),
                        msg="Linear.backward() gives unexpected output.\n"
                            "Expected: " + str(t.grad) + ", and got " + str(res))

        self.assertTrue(torch.allclose(torch_linear.weight.grad, linear.weights_grad),
                        msg="Linear.backward() gives unexpected weight gradient.\n"
                        "Expected: " + str(torch_linear.weight.grad) + ", and got " + str(linear.weights_grad))

        self.assertTrue(torch.allclose(torch_linear.bias.grad, linear.bias_grad),
                        msg="Linear.backward() gives unexpected bias gradient.\n"
                        "Expected: " + str(torch_linear.bias.grad) + ", and got " + str(linear.bias_grad))

    # Sequential
    def test_params_of_sequential(self):
        linear1 = Linear(in_dim=5, out_dim=10)
        linear2 = Linear(in_dim=10, out_dim=25)
        linear3 = Linear(in_dim=25, out_dim=2)

        sequential1 = Sequential(linear1, linear2, linear3)
        sequential2 = Sequential(linear1, linear2)
        sequential3 = Sequential(sequential2, linear3)

        self.assertEqual(
            len(sequential1.params()),
            6
        )
        self.assertEqual(
            len(sequential3.params()),
            6
        )
        self.assertEqual(
            sequential1.params(),
            sequential3.params()
        )

    def test_sequential_dimension(self):
        sequential = Sequential(
            Linear(in_dim=5, out_dim=10),
            Linear(in_dim=10, out_dim=25),
            Linear(in_dim=25, out_dim=2)
        )

        t = torch.empty((20, 5)).normal_()
        out = sequential.forward(t)

        self.assertEqual(
            out.shape,
            (20, 2)
        )

    def test_sequential_forward_backward(self):
        # For this test it's important
        torch.set_grad_enabled(True)

        # Instantiate our version of sequential
        linear1 = Linear(in_dim=5, out_dim=10)
        linear2 = Linear(in_dim=10, out_dim=25)
        linear3 = Linear(in_dim=25, out_dim=2)

        sequential = Sequential(linear1, linear2, linear3)

        # Instantiate pytorch's sequential and copy our parameters
        torch_linear1 = torch.nn.Linear(5, 10)
        torch_linear2 = torch.nn.Linear(10, 25)
        torch_linear3 = torch.nn.Linear(25, 2)

        torch_linear1.weight = nn.Parameter(linear1.weights, requires_grad=True)
        torch_linear2.weight = nn.Parameter(linear2.weights, requires_grad=True)
        torch_linear3.weight = nn.Parameter(linear3.weights, requires_grad=True)

        torch_linear1.bias = nn.Parameter(linear1.bias, requires_grad=True)
        torch_linear2.bias = nn.Parameter(linear2.bias, requires_grad=True)
        torch_linear3.bias = nn.Parameter(linear3.bias, requires_grad=True)

        model = torch.nn.Sequential(
            torch_linear1,
            torch_linear2,
            torch_linear3
        )

        t = torch.tensor([[5., 3., 6., 8., 3.],
                          [4., 6., 1., 11, 5.]], requires_grad=True)

        torch_out = model(t)
        out = sequential.forward(t)

        self.assertTrue(torch.allclose(out, torch_out),
                        msg="Sequential.forward() gives unexpected output.\n"
                            "Expected: " + str(torch_out) + ", and got " + str(out))

        loss = torch_out.sum()
        loss.backward()

        gradwrtoutput = torch.empty((t.shape[0], linear3.out_dim)).fill_(1)  # the grad of (x+y) wrt (x, y) is (1, 1)
        res = sequential.backward(gradwrtoutput)

        self.assertTrue(torch.allclose(res, t.grad, atol=1e-4),
                        msg="Sequential.backward() gives unexpected output.\n"
                        "Expected: " + str(t.grad) + ", and got " + str(res))

    # ReLU
    def test_ReLU_forward(self):
        m = ReLU()
        t = torch.tensor([[1., -1.], [1., -1.]])
        expected = torch.tensor([[1., 0.], [1., 0.]])
        self.assertTrue(torch.equal(m.forward(t), expected))

    def test_ReLU_backward_succeed(self):
        m = ReLU()
        t = torch.tensor([[1., -1.], [1., -1.]])
        grad = torch.tensor([[5., -7.], [2., -3.]])
        expected = torch.tensor([[5., 0.], [2, 0]])
        m.forward(t)
        self.assertTrue(torch.allclose(m.backward(grad), expected))

    # Tanh
    def test_Tanh(self):
        m = Tanh()
        t = torch.tensor([[1., -1.], [1., -1.]])
        expected = torch.tensor([[0.7616, -0.7616], [0.7616, -0.7616]])
        self.assertTrue(torch.allclose(m.forward(t), expected))

    def test_Tanh_backward_succeed(self):
        m = Tanh()
        t = torch.tensor([[1., -1.], [1., -1.]])
        grad = torch.tensor([[5., -7.], [2., -3.]])
        expected = torch.tensor([[2.0999, -2.9398], [0.8399, -1.2599]])
        m.forward(t)
        self.assertTrue(torch.allclose(m.backward(grad), expected, atol=1e-4))

    # Sigmoid
    def test_Sigmoid(self):
        m = Sigmoid()
        t = torch.tensor([[1., -1.], [1., -1.]])
        expected = torch.sigmoid(t)
        self.assertTrue(torch.allclose(m.forward(t), expected))

    def test_Sigmoid_backward(self):
        m = Sigmoid()
        t = torch.tensor([[1., -1.], [1., -1.]])
        grad = torch.tensor([[5., -7.], [2., -3.]])
        expected = torch.tensor([[0.9831, -1.3763], [0.3932, -0.5898]])
        m.forward(t)
        self.assertTrue(torch.allclose(m.backward(grad), expected, atol=1e-4))

    def test_SELU_forward(self):
        m = SELU()
        m_torch = torch.nn.SELU()
        t = torch.tensor([[2., -1.], [1., -2.]])
        self.assertTrue(torch.allclose(m.forward(t), m_torch(t)))

    def test_SELU_backward_succeed(self):
        m = SELU()
        t = torch.tensor([[1., -1.], [1., -1.]])
        grad = torch.tensor([[5., -7.], [2., -3.]])
        expected = torch.tensor([[5.2535052, -4.5273805], [2.1014020, -1.9403059]])
        m.forward(t)
        self.assertTrue(torch.allclose(m.backward(grad), expected, atol=1e-4))

    # All activations
    def test_activation_backward_fail(self):
        for act in [ReLU, Tanh, Sigmoid, SELU]:
            m = act()
            grad = torch.tensor([[5., -7.], [2., -3.]])
            self.assertRaises(ValueError, m.backward, grad)

    # All layers
    def test_layers_backward_fail(self):
        for layer in [Linear(2, 4), Sequential(Linear(1, 3), Linear(3, 4))]:
            grad = torch.empty((2, 4)).normal_()
            self.assertRaises(ValueError, layer.backward, grad)

    # Losses
    def test_LossMSE_backward_fail(self):
        mse = LossMSE()
        self.assertRaises(ValueError, mse.backward)

    def test_LossMSE_loss(self):
        loss = LossMSE()
        input = torch.tensor([[-0.2507, 0.4160, -0.1165, 1.1063, 0.4817],
                              [-1.0744, -0.0711, -0.2475, -0.7132, -0.0477],
                              [1.8209, -0.1623, -0.9103, 1.2300, 0.0613]])
        target = torch.tensor([[-1.2346, 0.4344, 0.9291, -1.6947, -1.9404],
                               [-2.2208, 1.1297, -2.3058, -2.2362, 1.4502],
                               [-0.6941, -1.7844, -2.0554, 1.6713, -0.8687]])
        # input.requires_grad_(True)
        output = loss.forward(input, target)
        loss = nn.MSELoss()
        expected = loss(input, target)
        self.assertTrue(torch.allclose(expected, output))

    def test_SGD_with_LossMSE(self):
        model = Sequential(
            Linear(2, 5),
            ReLU(),
            Linear(5, 10),
            Tanh(),
            Linear(10, 1),
            Sigmoid()
        )

        optimizer = SGD(model)
        criterion = LossMSE()

        t = torch.tensor([[2, 2.], [4, 5], [3, 1]])
        target = torch.tensor([[1.], [0.], [1.]])

        out = model.forward(t)

        loss = criterion.forward(out, target)
        gradwrtoutput = criterion.backward()

        model.backward(gradwrtoutput)

        optimizer.step(lr=1e-3)


if __name__ == '__main__':
    unittest.main()
