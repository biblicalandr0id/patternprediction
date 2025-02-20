import numpy as np

class BiologicalNeuron:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.potential = 0
        self.refractory_period = 0
        self.is_firing = False
        self.connections = {
            'forward': [],
            'lateral': [],
            'backward': []
        }
        
    def connect(self, target_neuron, connection_type, weight):
        self.connections[connection_type].append((target_neuron, weight))
        
    def receive_signal(self, input_value):
        if self.refractory_period > 0:
            self.refractory_period -= 1
            return
            
        self.potential += input_value
        
        if self.potential >= self.threshold:
            self.fire()
            
    def fire(self):
        if not self.is_firing and self.refractory_period == 0:
            self.is_firing = True
            self.propagate_signal()
            self.potential = 0
            self.refractory_period = 3  # Refractory period of 3 time steps
            self.is_firing = False
            
    def propagate_signal(self):
        for connection_type in ['forward', 'lateral', 'backward']:
            for target_neuron, weight in self.connections[connection_type]:
                target_neuron.receive_signal(weight)

class BiologicalNeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = []
        self.create_layers(layer_sizes)
        self.connect_layers()
        
    def create_layers(self, layer_sizes):
        for size in layer_sizes:
            layer = [BiologicalNeuron() for _ in range(size)]
            self.layers.append(layer)
            
    def connect_layers(self):
        # Connect forward
        for i in range(len(self.layers) - 1):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]
            
            for neuron in current_layer:
                for target in next_layer:
                    weight = np.random.normal(0.5, 0.1)
                    neuron.connect(target, 'forward', weight)
                    
        # Connect lateral (within layer)
        for layer in self.layers:
            for i, neuron in enumerate(layer):
                for j, target in enumerate(layer):
                    if i != j:
                        weight = np.random.normal(0.3, 0.1)
                        neuron.connect(target, 'lateral', weight)
                        
        # Connect backward
        for i in range(len(self.layers) - 1, 0, -1):
            current_layer = self.layers[i]
            prev_layer = self.layers[i - 1]
            
            for neuron in current_layer:
                for target in prev_layer:
                    weight = np.random.normal(0.2, 0.1)
                    neuron.connect(target, 'backward', weight)
                    
    def process_input(self, input_values):
        # Activate input layer
        for neuron, value in zip(self.layers[0], input_values):
            neuron.receive_signal(value)
            
        # Let the signal propagate for several time steps
        for _ in range(10):  # Arbitrary number of time steps
            # Process each layer
            for layer in self.layers:
                for neuron in layer:
                    if neuron.potential >= neuron.threshold:
                        neuron.fire()

# Example usage
network = BiologicalNeuralNetwork([3, 4, 3])  # Create network with 3 layers
input_values = [1.0, 0.5, 0.8]  # Example input
network.process_input(input_values)
