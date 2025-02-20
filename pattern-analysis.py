import numpy as np
import matplotlib.pyplot as plt
from enhanced_neural_network import EnhancedPatternCompletionNetwork, create_enhanced_pattern

def analyze_pattern_stability():
    """Analyze network stability with different patterns"""
    size = (7, 7)
    network = EnhancedPatternCompletionNetwork(size)
    
    # Test stability across different patterns
    patterns = ['A', 'face', 'complex']
    removal_rates = [0.3, 0.5, 0.7]
    results = {}
    
    for pattern_type in patterns:
        results[pattern_type] = []
        original = create_enhanced_pattern(size, pattern_type)
        
        for rate in removal_rates:
            # Multiple trials for statistical significance
            trial_accuracies = []
            trial_confidences = []
            
            for _ in range(10):  # 10 trials per configuration
                known_positions = network.present_pattern(original, rate)
                completed, confidence = network.update()
                
                accuracy = np.mean(completed == original)
                avg_confidence = np.mean(confidence)
                
                trial_accuracies.append(accuracy)
                trial_confidences.append(avg_confidence)
            
            results[pattern_type].append({
                'removal_rate': rate,
                'mean_accuracy': np.mean(trial_accuracies),
                'std_accuracy': np.std(trial_accuracies),
                'mean_confidence': np.mean(trial_confidences),
                'std_confidence': np.std(trial_confidences)
            })
    
    return results

def analyze_network_dynamics():
    """Analyze neural activation patterns and energy dynamics"""
    size = (7, 7)
    network = EnhancedPatternCompletionNetwork(size)
    pattern = create_enhanced_pattern(size, 'A')
    
    # Track network states over time
    known_positions = network.present_pattern(pattern, 0.5)
    states = []
    energies = []
    confidences = []
    
    for _ in range(30):  # Track 30 timesteps
        state, confidence = network.update(steps=1)
        
        # Collect network statistics
        energy_levels = [[network.neurons[i][j].energy 
                         for j in range(size[1])] 
                        for i in range(size[0])]
        
        states.append(state.copy())
        energies.append(np.mean(energy_levels))
        confidences.append(np.mean(confidence))
    
    return states, energies, confidences

def plot_analysis_results(stability_results, dynamics_results):
    """Visualize analysis results"""
    states, energies, confidences = dynamics_results
    
    # Plot 1: Pattern Completion Accuracy
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    for pattern in stability_results:
        rates = [r['removal_rate'] for r in stability_results[pattern]]
        accuracies = [r['mean_accuracy'] for r in stability_results[pattern]]
        errors = [r['std_accuracy'] for r in stability_results[pattern]]
        
        plt.errorbar(rates, accuracies, yerr=errors, label=pattern, marker='o')
    
    plt.xlabel('Removal Rate')
    plt.ylabel('Completion Accuracy')
    plt.title('Pattern Completion Performance')
    plt.legend()
    
    # Plot 2: Network Dynamics
    plt.subplot(132)
    plt.plot(energies, label='Mean Energy')
    plt.plot(confidences, label='Mean Confidence')
    plt.xlabel('Time Steps')
    plt.ylabel('Level')
    plt.title('Network Dynamics')
    plt.legend()
    
    # Plot 3: State Evolution
    plt.subplot(133)
    selected_states = [states[0], states[10], states[-1]]
    for idx, state in enumerate(selected_states):
        plt.subplot(133 + idx)
        plt.imshow(state, cmap='binary')
        plt.title(f'Step {idx*10}')
    
    plt.tight_layout()
    plt.show()

# Run analysis
stability_results = analyze_pattern_stability()
dynamics_results = analyze_network_dynamics()
plot_analysis_results(stability_results, dynamics_results)
