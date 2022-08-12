#! /usr/bin/python3
import sys
import pennylane as qml
from pennylane import numpy as np

# DO NOT MODIFY any of these parameters
a = 0.7
b = -0.3
dev = qml.device("default.qubit", wires=3)
dev1 = qml.device("default.qubit", wires=3)


def natural_gradient(params):
    """Calculate the natural gradient of the qnode() cost function.

    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers.

    You should evaluate the metric tensor and the gradient of the QNode, and then combine these
    together using the natural gradient definition. The natural gradient should be returned as a
    NumPy array.

    The metric tensor should be evaluated using the equation provided in the problem text. Hint:
    you will need to define a new QNode that returns the quantum state before measurement.

    Args:
        params (np.ndarray): Input parameters, of dimension 6

    Returns:
        np.ndarray: The natural gradient evaluated at the input parameters, of dimension 6
    """

    natural_grad = np.zeros(6)

    gradient = np.zeros([natural_grad.shape[0]])
    fim = np.zeros([natural_grad.shape[0], natural_grad.shape[0]])

    eps = np.pi / 2

    for k in range(gradient.shape[0]):
        eps_plus = params.copy()
        eps_plus[k] += eps
        exp_value_plus = qnode(eps_plus)

        eps_minus = params.copy()
        eps_minus[k] -= eps
        exp_value_minus = qnode(eps_minus)

        gradient[k] = (exp_value_plus - exp_value_minus) / (2 * np.sin(eps))

    eps = np.pi / 2

    qnode(params)
    state = dev.state
    for k in range(natural_grad.shape[0]):
        for l in range(gradient.shape[0]):
            if l <= k:
                eps_pp = params.copy()
                eps_pp[k] += eps
                eps_pp[l] += eps

                eps_pm = params.copy()
                eps_pm[k] += eps
                eps_pm[l] -= eps

                eps_mp = params.copy()
                eps_mp[k] -= eps
                eps_mp[l] += eps

                eps_mm = params.copy()
                eps_mm[k] -= eps
                eps_mm[l] -= eps

                qnode(eps_pp)
                state_pp = dev.state
                measure_pp = np.abs(np.conjugate(state) @ state_pp)

                qnode(eps_pm)
                state_pm = dev.state
                measure_pm = np.abs(np.conjugate(state) @ state_pm)

                qnode(eps_mp)
                state_mp = dev.state
                measure_mp = np.abs(np.conjugate(state) @ state_mp)

                qnode(eps_mm)
                state_mm = dev.state
                measure_mm = np.abs(np.conjugate(state) @ state_mm)

                fim[k, l] = (-measure_pp ** 2 - measure_mm ** 2 + measure_mp ** 2 + measure_pm ** 2) / 8

    for k in range(natural_grad.shape[0]):
        for l in range(natural_grad.shape[0]):
            if l > k:
                fim[k, l] = fim[l, k]

    natural_grad = np.dot(np.linalg.inv(fim), gradient)

    return natural_grad


def non_parametrized_layer():
    """A layer of fixed quantum gates.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.ipynb.
    """
    qml.RX(a, wires=0)
    qml.RX(b, wires=1)
    qml.RX(a, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.RZ(a, wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(b, wires=1)
    qml.Hadamard(wires=0)


def variational_circuit(params):
    """A layered variational circuit composed of two parametrized layers of single qubit rotations
    interleaved with non-parameterized layers of fixed quantum gates specified by
    ``non_parametrized_layer``.

    The first parametrized layer uses the first three parameters of ``params``, while the second
    layer uses the final three parameters.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.ipynb.
    """
    non_parametrized_layer()
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)
    non_parametrized_layer()
    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)


@qml.qnode(dev)
def qnode(params):
    """A PennyLane QNode that pairs the variational_circuit with an expectation value
    measurement.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.ipynb.
    """
    variational_circuit(params)
    return qml.expval(qml.PauliX(1))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process inputs
    params = sys.stdin.read()
    params = params.split(",")
    params = np.array(params, float)

    updated_params = natural_gradient(params)

    print(*updated_params, sep=",")