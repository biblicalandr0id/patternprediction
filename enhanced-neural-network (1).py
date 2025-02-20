import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from typing import Tuple, List, Optional
import time

class EnhancedBiologicalNeuron:
    def __init__(self, position: np.ndarray, params: dict = None):
        self.position = position
        self.params = params or {
            'threshold': 0.3,           # Lower threshold for better sensitivity
            'adaptation_rate': 0.05,    # Slower adaptation for stability
            'energy_decay': 0.15,       # Reduced energy decay
            'energy_recovery': 0.08,    # Faster energy recovery
            'refractory_period': 2,     # Standard refractory period
            'connection_strength': 0.7,  # Stronger initial connections
            'memory_decay': 0.98        # Slow decay of pattern memory
        }
        
        self.potential = 0
        self.threshold = self.params['threshold']
        self.refractory_period = 0
        self.is_firing = False
        self.energy = 1.0
        self.connections = []
        self.activity_history = deque(maxlen=10)  # Increased history length
        self.adaptation_level = 0
        self.pattern_memory = 0  # New: memory of repeated patterns
        self.confidence = 0      # New: confidence in current state
        
    def connect(self, target: 'EnhancedBiologicalNeuron', weight: float, max_distance: float = 3.0):
        """Enhanced connection with dynamic weight adjustment"""
        distance = np.linalg.norm(self.position - target.position)
        if distance <= max_distance:
            # Modified distance-weight relationship
            effective_weight = weight * (1 - (distance/max_distance)**2) * self.params['connection_strength']
            self.connections.append((target, effective_weight))
            
    def update(self, local_field: float = 0, global_activity: float = 0) -> bool:
        """Enhanced update with global activity influence and memory"""
        if self.refractory_period > 0:
            self.refractory_period -= 1
            self.is_firing = False
            return False
            
        # Energy dynamics
        if self.is_firing:
            self.energy = max(0.1, self.energy - self.params['energy_decay'])
            self.adaptation_level = min(1.0, self.adaptation_level + self.params['adaptation_rate'])
        else:
            self.energy = min(1.0, self.energy + self.params['energy_recovery'])
            self.adaptation_level = max(0.0, self.adaptation_level - self.params['adaptation_rate']/2)
        
        # Pattern memory influence
        memory_influence = self.pattern_memory * 0.2
        
        # Dynamic threshold based on global activity
        dynamic_threshold = self.threshold * (1 + global_activity * 0.2)
        
        # Enhanced input processing
        adjusted_input = (self.potential + local_field + memory_influence) * (1 - self.adaptation_level * 0.5)
        activation_prob = 1 / (1 + np.exp(-(adjusted_input - dynamic_threshold)))
        
        should_fire = np.random.random() < activation_prob
        
        if should_fire and self.energy > 0.2:
            self.is_firing = True
            self.refractory_period = self.params['refractory_period']
            self.potential = 0
            self.activity_history.append(1)
            self.pattern_memory = min(1.0, self.pattern_memory + 0.1)  # Strengthen memory
            self.confidence = activation_prob  # Update confidence
            return True
        
        self.is_firing = False
        self.activity_history.append(0)
        self.pattern_memory *= self.params['memory_decay']  # Decay memory
        self.confidence = 1 - activation_prob  # Update confidence
        return False

class EnhancedPatternCompletionNetwork:
    def __init__(self, size: Tuple[int, int] = (7, 7), params: dict = None):
        self.size = size
        self.params = params or {
            'connection_density': 0.7,    # Higher connection density
            'weight_variance': 0.1,       # Connection weight variance
            'completion_steps': 30,       # More steps for refinement
            'stability_threshold': 0.95   # Required stability for completion
        }
        self.neurons = self._create_neuron_grid()
        self._connect_neurons()
        self.pattern_history = []
        self.confidence_map = np.zeros(size)
        
    def _create_neuron_grid(self) -> List[List[EnhancedBiologicalNeuron]]:
        return [[EnhancedBiologicalNeuron(np.array([i, j])) 
                for j in range(self.size[1])] 
                for i in range(self.size[0])]
    
    def _connect_neurons(self):
        """Enhanced connection scheme with variable density"""
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                current = self.neurons[i][j]
                # Increased connection radius
                for di in range(-3, 4):
                    for dj in range(-3, 4):
                        ni, nj = i + di, j + dj
                        if (0 <= ni < self.size[0] and 0 <= nj < self.size[1] 
                            and (di != 0 or dj != 0)):
                            if np.random.random() < self.params['connection_density']:
                                weight = np.random.normal(0.5, self.params['weight_variance'])
                                current.connect(self.neurons[ni][nj], weight)
    
    def present_pattern(self, pattern: np.ndarray, removal_rate: float) -> set:
        """Present pattern with enhanced initial state setup"""
        self.pattern_history = []
        self.confidence_map = np.zeros(self.size)
        known_positions = set()
        
        # Initialize with pattern and confidence
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if np.random.random() > removal_rate:
                    known_positions.add((i, j))
                    if pattern[i][j] > 0:
                        self.neurons[i][j].potential = 1.0
                        self.neurons[i][j].confidence = 0.9
                    else:
                        self.neurons[i][j].potential = -0.5
                        self.neurons[i][j].confidence = 0.7
                else:
                    self.neurons[i][j].potential = 0
                    self.neurons[i][j].confidence = 0.1
        
        return known_positions
    
    def update(self, steps: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Enhanced update with multi-phase completion"""
        if steps is None:
            steps = self.params['completion_steps']
            
        stability_count = 0
        prev_pattern = None
        
        for step in range(steps):
            current_pattern = np.zeros(self.size)
            confidence_map = np.zeros(self.size)
            
            # Calculate global activity
            global_activity = sum(
                neuron.is_firing 
                for row in self.neurons 
                for neuron in row
            ) / (self.size[0] * self.size[1])
            
            # Update phase
            for i in range(self.size[0]):
                for j in range(self.size[1]):
                    neuron = self.neurons[i][j]
                    local_field = sum(w * t.is_firing 
                                    for t, w in neuron.connections)
                    fired = neuron.update(local_field, global_activity)
                    current_pattern[i][j] = 1 if fired else 0
                    confidence_map[i][j] = neuron.confidence
            
            self.pattern_history.append(current_pattern)
            self.confidence_map = confidence_map
            
            # Check for stability
            if prev_pattern is not None:
                if np.array_equal(current_pattern, prev_pattern):
                    stability_count += 1
                    if stability_count >= 3:  # Require 3 stable steps
                        break
                else:
                    stability_count = 0
                    
            prev_pattern = current_pattern.copy()
            
        return self.pattern_history[-1], self.confidence_map

def create_enhanced_pattern(size: Tuple[int, int] = (7, 7), pattern_type: str = 'A') -> np.ndarray:
    """Create enhanced test patterns"""
    pattern = np.zeros(size)
    
    if pattern_type == 'A':
        # Enhanced 'A' pattern
        pattern[1:-1, 1] = 1      # Left line
        pattern[1:-1, -2] = 1     # Right line
        pattern[size[0]//2, 1:-1] = 1  # Middle line
        pattern[1, 1:-1] = 1      # Top line
        
    elif pattern_type == 'face':
        # Enhanced face pattern
        # Eyes
        pattern[2, 2:4] = 1
        pattern[2, 4:6] = 1
        # Nose
        pattern[3:5, 3] = 1
        # Mouth
        pattern[5, 2:6] = 1
        pattern[4, 2] = 1
        pattern[4, 5] = 1
        
    elif pattern_type == 'complex':
        # More complex pattern (spiral)
        center = (size[0]//2, size[1]//2)
        for i in range(size[0]):
            for j in range(size[1]):
                dist = np.sqrt((i-center[0])**2 + (j-center[1])**2)
                angle = np.arctan2(i-center[0], j-center[1])
                if abs(dist - angle) % 2 < 0.5:
                    pattern[i,j] = 1
                    
    return pattern

def test_enhanced_completion(pattern_type: str = 'A', 
                           removal_rates: List[float] = None,
                           show_confidence: bool = True):
    """Test pattern completion with enhanced visualization"""
    if removal_rates is None:
        removal_rates = [0.3, 0.5, 0.7, 0.9]
        
    size = (7, 7)
    original = create_enhanced_pattern(size, pattern_type)
    results = []
    
    # Create enhanced network
    network = EnhancedPatternCompletionNetwork(size)
    
    # Setup visualization
    n_rows = len(removal_rates)
    fig, axes = plt.subplots(n_rows, 4 if show_confidence else 3, 
                            figsize=(15 if show_confidence else 12, 5*n_rows))
    
    for idx, rate in enumerate(removal_rates):
        start_time = time.time()
        
        # Present pattern and complete
        known_positions = network.present_pattern(original, rate)
        
        partial = np.full(size, np.nan)
        for i in range(size[0]):
            for j in range(size[1]):
                if (i, j) in known_positions:
                    partial[i, j] = original[i, j]
        
        completed, confidence = network.update()
        
        # Calculate metrics
        accuracy = np.mean(completed == original)
        completion_time = time.time() - start_time
        
        # Visualize
        row = axes[idx] if n_rows > 1 else axes
        
        row[0].imshow(original, cmap='binary')
        row[0].set_title('Original Pattern')
        
        masked_partial = np.ma.masked_array(partial, mask=np.isnan(partial))
        row[1].imshow(masked_partial, cmap='binary')
        row[1].set_title(f'{int(rate*100)}% Removed')
        
        row[2].imshow(completed, cmap='binary')
        row[2].set_title(f'Completed (Acc: {accuracy*100:.1f}%)')
        
        if show_confidence:
            confidence_plot = row[3].imshow(confidence, cmap='plasma')
            row[3].set_title('Confidence Map')
            plt.colorbar(confidence_plot, ax=row[3])
        
        results.append({
            'removal_rate': rate,
            'accuracy': accuracy,
            'completion_time': completion_time,
            'avg_confidence': np.mean(confidence)
        })
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed results
    print("\nCompletion Results:")
    print("------------------")
    for result in results:
        print(f"\nRemoval Rate {int(result['removal_rate']*100)}%:")
        print(f"Accuracy: {result['accuracy']*100:.1f}%")
        print(f"Completion Time: {result['completion_time']*1000:.1f}ms")
        print(f"Average Confidence: {result['avg_confidence']*100:.1f}%")

if __name__ == "__main__":
    # Test with different patterns and visualize confidence
    test_enhanced_completion('A')
    test_enhanced_completion('face')
    test_enhanced_completion('complex')
    
    # Additional parameter exploration can be added here
    # For example:
    network = EnhancedPatternCompletionNetwork(
        size=(7, 7),
        params={
            'connection_density': 0.8,
            'weight_variance': 0.05,
            'completion_steps': 40,
            'stability_threshold': 0.98
        }
    )
