#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def gradient_200(weights, dev):
    r"""This function must compute the gradient *and* the Hessian of the variational
    circuit using the parameter-shift rule, using exactly 51 device executions.
    The code you write for this challenge should be completely contained within
    this function between the # QHACK # comment markers.

    Args:
        weights (array): An array of floating-point numbers with size (5,).
        dev (Device): a PennyLane device for quantum circuit execution.

    Returns:
        tuple[array, array]: This function returns a tuple (gradient, hessian).

            * gradient is a real NumPy array of size (5,).

            * hessian is a real NumPy array of size (5, 5).
    """

    @qml.qnode(dev, interface=None)
    def circuit(w):
        for i in range(3):
            qml.RX(w[i], wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RY(w[3], wires=1)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RX(w[4], wires=2)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    gradient = np.zeros([5], dtype=np.float64)
    hessian = np.zeros([5, 5], dtype=np.float64)

    f = circuit(weights)

    eps = np.pi / 4

    exp_p = np.zeros([5], dtype=np.float64)
    exp_m = np.zeros([5], dtype=np.float64)

    for k in range(gradient.shape[0]):
        eps_plus = weights.copy()
        eps_plus[k] += eps
        exp_value_plus = circuit(eps_plus)
        exp_p[k] = exp_value_plus

        eps_minus = weights.copy()
        eps_minus[k] -= eps
        exp_value_minus = circuit(eps_minus)
        exp_m[k] = exp_value_minus

        gradient[k] = (exp_value_plus - exp_value_minus) / (2 * np.sin(eps))

    for k in range(gradient.shape[0]):
        hessian[k, k] = (exp_p[k] + exp_m[k] - 2 * f) / (4 * (np.sin(eps / 2) ** 2))
        for l in range(gradient.shape[0]):
            if l < k:
                eps_pp = weights.copy()
                eps_pp[k] += eps
                eps_pp[l] += eps

                eps_pm = weights.copy()
                eps_pm[k] += eps
                eps_pm[l] -= eps

                eps_mp = weights.copy()
                eps_mp[k] -= eps
                eps_mp[l] += eps

                eps_mm = weights.copy()
                eps_mm[k] -= eps
                eps_mm[l] -= eps

                measure_pp = circuit(eps_pp)
                measure_pm = circuit(eps_pm)
                measure_mp = circuit(eps_mp)
                measure_mm = circuit(eps_mm)

                hessian[k, l] = (measure_pp + measure_mm - measure_mp - measure_pm) / (4 * (np.sin(eps) ** 2))

    for k in range(gradient.shape[0]):
        for l in range(gradient.shape[0]):
            if l > k:
                hessian[k, l] = hessian[l, k]

    return gradient, hessian, circuit.diff_method


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    weights = sys.stdin.read()
    weights = weights.split(",")
    weights = np.array(weights, float)

    dev = qml.device("default.qubit", wires=3)
    gradient, hessian, diff_method = gradient_200(weights, dev)

    print(
        *np.round(gradient, 10),
        *np.round(hessian.flatten(), 10),
        dev.num_executions,
        sep=","
    )
