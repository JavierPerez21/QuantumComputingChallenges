#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def find_excited_states(H):
    """
    Fill in the missing parts between the # QHACK # markers below. Implement
    a variational method that can find the three lowest energies of the provided
    Hamiltonian.

    Args:
        H (qml.Hamiltonian): The input Hamiltonian

    Returns:
        The lowest three eigenenergies of the Hamiltonian as a comma-separated string,
        sorted from smallest to largest.
    """

    energies = np.zeros(3)

    # QHACK #

    from functools import partial
    from pennylane.templates.subroutines import UCCSD
    from pennylane import qchem

    qubits = len(H.wires)
    dev = qml.device("default.qubit", wires=qubits)
    state = np.zeros(2 ** qubits, dtype=np.complex)
    state[0] = 1

    def variational_ansatz(params, wires):
        n_qubits = len(wires)
        n_rotations = len(params)

        # qml.templates.state_preparations.MottonenStatePreparation(state,wires)

        if n_rotations > 1:
            n_layers = n_rotations // n_qubits
            n_extra_rots = n_rotations - n_layers * n_qubits

            # Alternating layers of unitary rotations on every qubit followed by a
            # ring cascade of CNOTs.
            for layer_idx in range(n_layers):
                layer_params = params[layer_idx * n_qubits: layer_idx * n_qubits + n_qubits, :]
                qml.broadcast(qml.Rot, wires, pattern="single", parameters=layer_params)
                qml.broadcast(qml.CNOT, wires, pattern="ring")

            # There may be "extra" parameter sets required for which it's not necessarily
            # to perform another full alternating cycle. Apply these to the qubits as needed.
            extra_params = params[-n_extra_rots:, :]
            extra_wires = wires[: n_qubits - 1 - n_extra_rots: -1]
            qml.broadcast(qml.Rot, extra_wires, pattern="single", parameters=extra_params)
        else:
            # For 1-qubit case, just a single rotation to the qubit
            qml.Rot(*params[0], wires=wires[0])

    dev2 = qml.device("default.qubit", wires=qubits)

    @qml.qnode(dev2)
    def variational_ansatz2(params):
        wires = range(len(H.wires))
        n_qubits = len(wires)
        n_rotations = len(params)

        if n_rotations > 1:
            n_layers = n_rotations // n_qubits
            n_extra_rots = n_rotations - n_layers * n_qubits

            # Alternating layers of unitary rotations on every qubit followed by a
            # ring cascade of CNOTs.
            for layer_idx in range(n_layers):
                layer_params = params[layer_idx * n_qubits: layer_idx * n_qubits + n_qubits, :]
                qml.broadcast(qml.Rot, wires, pattern="single", parameters=layer_params)
                qml.broadcast(qml.CNOT, wires, pattern="ring")

            # There may be "extra" parameter sets required for which it's not necessarily
            # to perform another full alternating cycle. Apply these to the qubits as needed.
            extra_params = params[-n_extra_rots:, :]
            extra_wires = wires[: n_qubits - 1 - n_extra_rots: -1]
            qml.broadcast(qml.Rot, extra_wires, pattern="single", parameters=extra_params)
        else:
            # For 1-qubit case, just a single rotation to the qubit
            qml.Rot(*params[0], wires=wires[0])

        qml.adjoint(qml.templates.state_preparations.MottonenStatePreparation(state0, wires))
        projector = np.zeros((2 ** qubits, 2 ** qubits))
        projector[0, 0] = 1
        return qml.expval(qml.Hermitian(projector, wires=range(qubits)))

    dev3 = qml.device("default.qubit", wires=qubits)

    @qml.qnode(dev3)
    def variational_ansatz3(params):
        wires = range(len(H.wires))
        n_qubits = len(wires)
        n_rotations = len(params)

        if n_rotations > 1:
            n_layers = n_rotations // n_qubits
            n_extra_rots = n_rotations - n_layers * n_qubits

            # Alternating layers of unitary rotations on every qubit followed by a
            # ring cascade of CNOTs.
            for layer_idx in range(n_layers):
                layer_params = params[layer_idx * n_qubits: layer_idx * n_qubits + n_qubits, :]
                qml.broadcast(qml.Rot, wires, pattern="single", parameters=layer_params)
                qml.broadcast(qml.CNOT, wires, pattern="ring")

            # There may be "extra" parameter sets required for which it's not necessarily
            # to perform another full alternating cycle. Apply these to the qubits as needed.
            extra_params = params[-n_extra_rots:, :]
            extra_wires = wires[: n_qubits - 1 - n_extra_rots: -1]
            qml.broadcast(qml.Rot, extra_wires, pattern="single", parameters=extra_params)
        else:
            # For 1-qubit case, just a single rotation to the qubit
            qml.Rot(*params[0], wires=wires[0])

        qml.adjoint(qml.templates.state_preparations.MottonenStatePreparation(state1, wires))
        projector = np.zeros((2 ** qubits, 2 ** qubits))
        projector[0, 0] = 1
        return qml.expval(qml.Hermitian(projector, wires=range(qubits)))

    cost_fn = qml.ExpvalCost(variational_ansatz, H, dev)

    opt = qml.AdamOptimizer(stepsize=0.4)
    opt2 = qml.AdamOptimizer(stepsize=0.4)
    opt3 = qml.AdamOptimizer(stepsize=1)
    num_param_sets = (2 ** qubits) - 1
    params = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=(num_param_sets, 3))
    params2 = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=(num_param_sets, 3))
    params3 = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=(num_param_sets, 3))

    max_iterations = 300
    conv_tol = 1e-07

    for n in range(max_iterations):
        params, prev_energy = opt.step_and_cost(cost_fn, params)
        energy = cost_fn(params)
        conv = np.abs(energy - prev_energy)
        # print(n,energy)
        if conv <= conv_tol:
            break

    state0 = dev.state
    energies[0] = energy

    def cost_fn_2(params):
        return cost_fn(params) + 100 * (variational_ansatz2(params)) ** 2  # 10

    for n in range(max_iterations):
        params2, prev_energy2 = opt2.step_and_cost(cost_fn_2, params2)
        energy2 = cost_fn_2(params2)
        conv = np.abs(energy2 - prev_energy2)
        if conv <= conv_tol:
            break

    energies[1] = cost_fn(params2)
    state1 = dev.state

    def cost_fn_3(params):
        return cost_fn(params) + 100 * (variational_ansatz2(params)) ** 2 + 160 * (variational_ansatz3(params)) ** 2

    max_iterations = 200
    for n in range(max_iterations):
        params3, prev_energy3 = opt3.step_and_cost(cost_fn_3, params3)
        energy3 = cost_fn_3(params3)
        conv = np.abs(energy3 - prev_energy3)
        '''
        if conv <= conv_tol:
            break
        '''
    energies[2] = cost_fn(params3)

    # QHACK #

    return ",".join([str(E) for E in energies])


def pauli_token_to_operator(token):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    Helper function to turn strings into qml operators.

    Args:
        token (str): A Pauli operator input in string form.

    Returns:
        A qml.Operator instance of the Pauli.
    """
    qubit_terms = []

    for term in token:
        # Special case of identity
        if term == "I":
            qubit_terms.append(qml.Identity(0))
        else:
            pauli, qubit_idx = term[0], term[1:]
            if pauli == "X":
                qubit_terms.append(qml.PauliX(int(qubit_idx)))
            elif pauli == "Y":
                qubit_terms.append(qml.PauliY(int(qubit_idx)))
            elif pauli == "Z":
                qubit_terms.append(qml.PauliZ(int(qubit_idx)))
            else:
                print("Invalid input.")

    full_term = qubit_terms[0]
    for term in qubit_terms[1:]:
        full_term = full_term @ term

    return full_term


def parse_hamiltonian_input(input_data):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    Turns the contents of the input file into a Hamiltonian.

    Args:
        filename(str): Name of the input file that contains the Hamiltonian.

    Returns:
        qml.Hamiltonian object of the Hamiltonian specified in the file.
    """
    # Get the input
    coeffs = []
    pauli_terms = []

    # Go through line by line and build up the Hamiltonian
    for line in input_data.split("S"):
        line = line.strip()
        tokens = line.split(" ")

        # Parse coefficients
        sign, value = tokens[0], tokens[1]

        coeff = float(value)
        if sign == "-":
            coeff *= -1
        coeffs.append(coeff)

        # Parse Pauli component
        pauli = tokens[2:]
        pauli_terms.append(pauli_token_to_operator(pauli))

    return qml.Hamiltonian(coeffs, pauli_terms)


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Turn input to Hamiltonian
    H = parse_hamiltonian_input(sys.stdin.read())

    # Send Hamiltonian through VQE routine and output the solution
    lowest_three_energies = find_excited_states(H)
    print(lowest_three_energies)