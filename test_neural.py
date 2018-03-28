import snake_neural
import numpy as np



def test_network():
    topology = [2,2,2]
    net = snake_neural.Network(topology)
    assert net.layers != []
    assert net.weights != []
    assert net.biases != []

    net.cross_over(net, net, 1)
    