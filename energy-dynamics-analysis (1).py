import numpy as np
import matplotlib.pyplot as plt
from enhanced_neural_network import EnhancedPatternCompletionNetwork, create_enhanced_pattern

def analyze_energy_dynamics(param_sets):
    """Analyze network performance with different energy parameters"""
    size = (7, 7)
    results = {}
    
    # Create multiple test patterns for robust analysis
    patterns = {
        'A': create_enhanced_pattern(size, 'A'),
        'face': create_enhanced_pattern(size, 'face'),
        'complex': create_enhanced_pattern(size, 'complex')
    }
    
    for param_name, param_values in param_sets.items():
        results[param_name] = []
        
        for value in param_values:
            # Create network with modified parameters
            params = {
                'threshold': 0.3,
                'adaptation_rate': 0.05,
                'energy_decay': 0.15,
                'energy_recovery': 0.08,
                'refractory_period': 2,
                'connection_strength': 0.7,
                'memory_decay': 0.98
            }
            params[param_name] = value
            
            # Track metrics across different patterns
            pattern_results = []
            
            for pattern_type, pattern in patterns.items():
                network = EnhancedPatternCompletionNetwork(size, params)
                
                # Multiple trials with different removal rates
                removal_rates = [0.3, 0.5, 0.7]
                trial_results = []
                
                for rate in removal_rates:
                    energy_profiles = []
                    completion_accuracies = []
                    activation_counts = []
                    
                    for _ in range(5):  # 5 trials per configuration
                        known_positions = network.present_pattern(pattern, rate)
                        energy_levels = []
                        activations = []
                        
                        # Track energy over completion steps
                        for step in range(30):  # Extended time steps
                            completed, confidence = network.update(steps=1)
                            
                            # Calculate network metrics
                            avg_energy = np.mean([[network.neurons[i][j].energy 
                                                 for j in range(size[1])]
                                                for i in range(size[0])])
                            
                            # Count active neurons
                            active_count = np.sum([[network.neurons[i][j].is_firing 
                                                  for j in range(size[1])]
                                                 for i in range(size[0])])
                            
                            energy_levels.append(avg_energy)
                            activations.append(active_count)
                        
                        accuracy = np.mean(completed == pattern)
                        
                        energy_profiles.append(energy_levels)
                        completion_accuracies.append(accuracy)
                        activation_counts.append(activations)
                    
                    trial_results.append({
                        'removal_rate': rate,
                        'mean_energy_profile': np.mean(energy_profiles, axis=0),
                        'std_energy_profile': np.std(energy_profiles, axis=0),
                        'mean_accuracy': np.mean(completion_accuracies),
                        'std_accuracy': np.std(completion_accuracies),
                        'mean_activations': np.mean(activation_counts, axis=0),
                        'std_activations': np.std(activation_counts, axis=0)
                    })
                
                pattern_results.append({
                    'pattern_type': pattern_type,
                    'trials': trial_results
                })
            
            results[param_name].append({
                'value': value,
                'patterns': pattern_results
            })
    
    return results

def plot_comprehensive_analysis(results):
    """Create detailed visualization of energy dynamics analysis"""
    for param_name, param_results in results.items():
        # Create figure with subplots for different metrics
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Energy Profiles
        ax1 = plt.subplot(231)
        for result in param_results:
            # Average across patterns and removal rates
            mean_profile = np.mean([
                trial['mean_energy_profile']
                for pattern in result['patterns']
                for trial in pattern['trials']
            ], axis=0)
            
            ax1.plot(mean_profile, label=f'{param_name}={result["value"]:.3f}')
        
        ax1.set_title('Average Energy Profile')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Energy Level')
        ax1.legend()
        
        # 2. Accuracy vs Parameter Value
        ax2 = plt.subplot(232)
        param_values = [r['value'] for r in param_results]
        pattern_types = results[param_name][0]['patterns'][0]['pattern_type']
        
        for pattern_idx, pattern_type in enumerate(['A', 'face', 'complex']):
            accuracies = [
                np.mean([trial['mean_accuracy'] 
                        for trial in result['patterns'][pattern_idx]['trials']])
                for result in param_results
            ]
            ax2.plot(param_values, accuracies, 
                    label=f'Pattern: {pattern_type}', 
                    marker='o')
        
        ax2.set_title(f'Accuracy vs {param_name}')
        ax2.set_xlabel(param_name)
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        # 3. Energy Stability
        ax3 = plt.subplot(233)
        for pattern_idx, pattern_type in enumerate(['A', 'face', 'complex']):
            stability = [
                np.mean([np.std(trial['mean_energy_profile'])
                        for trial in result['patterns'][pattern_idx]['trials']])
                for result in param_results
            ]
            ax3.plot(param_values, stability,
                    label=f'Pattern: {pattern_type}',
                    marker='o')
        
        ax3.set_title('Energy Stability')
        ax3.set_xlabel(param_name)
        ax3.set_ylabel('Energy Variance')
        ax3.legend()
        
        # 4. Activation Patterns
        ax4 = plt.subplot(234)
        for result in param_results:
            mean_activations = np.mean([
                trial['mean_activations']
                for pattern in result['patterns']
                for trial in pattern['trials']
            ], axis=0)
            
            ax4.plot(mean_activations, 
                    label=f'{param_name}={result["value"]:.3f}')
        
        ax4.set_title('Neural Activation Patterns')
        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('Active Neurons')
        ax4.legend()
        
        plt.tight_layout()
        plt.show()

# Define expanded parameter ranges
param_sets = {
    'energy_decay': np.linspace(0.05, 0.35, 7),     # Broader decay range
    'energy_recovery': np.linspace(0.01, 0.15, 7),  # Broader recovery range
    'threshold': np.linspace(0.1, 0.6, 7),          # Broader threshold range
    'adaptation_rate': np.linspace(0.02, 0.12, 7)   # Added adaptation rate
}

# Run analysis and plot results
results = analyze_energy_dynamics(param_sets)
plot_comprehensive_analysis(results)

def find_optimal_parameters(results):
    """Find optimal parameters considering multiple metrics"""
    optimal_params = {}
    
    for param_name, param_results in results.items():
        scores = []
        values = [r['value'] for r in param_results]
        
        for result in param_results:
            # Calculate composite score across all patterns and trials
            accuracy_score = np.mean([
                np.mean([trial['mean_accuracy'] 
                        for trial in pattern['trials']])
                for pattern in result['patterns']
            ])
            
            stability_score = -np.mean([
                np.mean([np.std(trial['mean_energy_profile'])
                        for trial in pattern['trials']])
                for pattern in result['patterns']
            ])
            
            efficiency_score = np.mean([
                np.mean([np.mean(trial['mean_activations'])
                        for trial in pattern['trials']])
                for pattern in result['patterns']
            ])
            
            # Combine scores (weighted sum)
            composite_score = (0.5 * accuracy_score + 
                             0.3 * stability_score +
                             0.2 * efficiency_score)
            scores.append(composite_score)
        
        optimal_idx = np.argmax(scores)
        optimal_params[param_name] = values[optimal_idx]
    
    return optimal_params

# Find and print optimal parameters
optimal_params = find_optimal_parameters(results)
print("\nOptimal Parameters:")
for param, value in optimal_params.items():
    print(f"{param}: {value:.3f}")
