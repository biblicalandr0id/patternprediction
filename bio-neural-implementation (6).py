import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class BiologicalNeuron:
    def __init__(self, position, threshold=0.5):
        self.position = position
        self.threshold = threshold
        self.potential = 0
        self.refractory_period = 0
        self.is_firing = False
        self.energy = 1.0
        self.connections = []
        self.activity_history = deque(maxlen=5)
        self.adaptation_level = 0  # Neural adaptation to sustained input
        
    def connect(self, target, weight, max_distance=2.0):
        distance = np.linalg.norm(self.position - target.position)
        if distance <= max_distance:
            # Stronger connections to closer neurons
            effective_weight = weight * (1 - distance/max_distance)
            self.connections.append((target, effective_weight))
    
    def update(self, local_field=0):
        if self.refractory_period > 0:
            self.refractory_period -= 1
            self.is_firing = False
            return False
            
        # Update energy and adaptation
        if self.is_firing:
            self.energy = max(0.1, self.energy - 0.2)
            self.adaptation_level = min(1.0, self.adaptation_level + 0.1)
        else:
            self.energy = min(1.0, self.energy + 0.05)
            self.adaptation_level = max(0.0, self.adaptation_level - 0.05)
        
        # Adjust input based on adaptation
        adjusted_input = (self.potential + local_field) * (1 - self.adaptation_level * 0.5)
        activation_prob = 1 / (1 + np.exp(-(adjusted_input - self.threshold)))
        
        should_fire = np.random.random() < activation_prob
        
        if should_fire and self.energy > 0.2:
            self.is_firing = True
            self.refractory_period = 2
            self.potential = 0
            self.activity_history.append(1)
            return True
        
        self.is_firing = False
        self.activity_history.append(0)
        return False

class PatternCompletionNetwork:
    def __init__(self, size=(5, 5)):
        self.size = size
        self.neurons = self._create_neuron_grid()
        self._connect_neurons()
        self.pattern_history = []
        
    def _create_neuron_grid(self):
        return [[BiologicalNeuron(np.array([i, j])) 
                for j in range(self.size[1])] 
                for i in range(self.size[0])]
    
    def _connect_neurons(self):
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                current = self.neurons[i][j]
                for di in range(-2, 3):
                    for dj in range(-2, 3):
                        ni, nj = i + di, j + dj
                        if (0 <= ni < self.size[0] and 0 <= nj < self.size[1] 
                            and (di != 0 or dj != 0)):
                            weight = np.random.normal(0.5, 0.1)
                            current.connect(self.neurons[ni][nj], weight)
    
    def present_pattern(self, pattern, removal_rate):
        """Present pattern with specified removal rate"""
        self.pattern_history = []
        known_positions = set()
        
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if np.random.random() > removal_rate:
                    known_positions.add((i, j))
                    if pattern[i][j] > 0:
                        self.neurons[i][j].potential = 1.0
                    else:
                        self.neurons[i][j].potential = -0.5
                else:
                    self.neurons[i][j].potential = 0
        
        return known_positions
    
    def update(self, steps=20):
        for step in range(steps):
            current_pattern = np.zeros(self.size)
            
            for i in range(self.size[0]):
                for j in range(self.size[1]):
                    neuron = self.neurons[i][j]
                    local_field = sum(w * t.is_firing 
                                    for t, w in neuron.connections)
                    fired = neuron.update(local_field)
                    current_pattern[i][j] = 1 if fired else 0
            
            self.pattern_history.append(current_pattern)
            
            if len(self.pattern_history) > 2:
                if np.array_equal(self.pattern_history[-1], 
                                self.pattern_history[-2]):
                    break
        
        return self.pattern_history[-1]

def test_completion_rates():
    # Create a more complex pattern (e.g., a simple face)
    size = (7, 7)
    original = np.zeros(size)
    
    # Eyes
    original[2, 2] = 1
    original[2, 4] = 1
    
    # Nose
    original[3, 3] = 1
    
    # Mouth
    original[4, 2:5] = 1
    
    # Test different removal rates
    removal_rates = [0.3, 0.5, 0.7, 0.9]
    results = []
    
    network = PatternCompletionNetwork(size)
    
    fig, axes = plt.subplots(len(removal_rates), 3, 
                            figsize=(15, 5*len(removal_rates)))
    
    for idx, rate in enumerate(removal_rates):
        # Present pattern with current removal rate
        known_positions = network.present_pattern(original, rate)
        
        # Create partial pattern for visualization
        partial = np.full(size, np.nan)
        for i in range(size[0]):
            for j in range(size[1]):
                if (i, j) in known_positions:
                    partial[i, j] = original[i, j]
        
        # Complete pattern
        completed = network.update(steps=20)
        
        # Visualize
        axes[idx, 0].imshow(original, cmap='binary')
        axes[idx, 0].set_title(f'Original Pattern')
        
        masked_partial = np.ma.masked_array(partial, mask=np.isnan(partial))
        axes[idx, 1].imshow(masked_partial, cmap='binary')
        axes[idx, 1].set_title(f'{int(rate*100)}% Removed')
        
        axes[idx, 2].imshow(completed, cmap='binary')
        axes[idx, 2].set_title('Completed Pattern')
        
        # Calculate completion accuracy
        accuracy = np.mean(completed == original)
        results.append((rate, accuracy))
        
    plt.tight_layout()
    plt.show()
    
    # Print accuracy results
    print("\nCompletion Accuracy:")
    for rate, accuracy in results:
        print(f"Removal Rate {int(rate*100)}%: {accuracy*100:.1f}% accurate")

test_completion_rates()
