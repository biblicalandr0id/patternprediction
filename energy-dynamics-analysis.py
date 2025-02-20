import numpy as np
import matplotlib.pyplot as plt
from enhanced_neural_network import EnhancedPatternCompletionNetwork, create_enhanced_pattern

def analyze_energy_dynamics(param_sets):
    """Analyze network performance with different energy parameters"""
    size = (7, 7)
    results = {}
    
    # Create test pattern
    pattern = create_enhanced_pattern(size, 'A')
    
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
            
            network = EnhancedPatternCompletionNetwork(size, params)
            
            # Track energy metrics over multiple trials
            energy_profiles = []
            completion_accuracies = []
            
            for _ in range(5):  # 5 trials per configuration
                known_positions = network.present_pattern(pattern, 0.5)
                energy_levels = []
                
                # Track energy over completion steps
                for step in range(20):
                    completed, _ = network.update(steps=1)
                    
                    # Calculate average energy across network
                    avg_energy = np.mean([[network.neurons[i][j].energy 
                                         for j in range(size[1])]
                                        for i in range(size[0])])
                    energy_levels.append(avg_energy)
                
                # Calculate completion accuracy
                accuracy = np.mean(completed == pattern)
                
                energy_profiles.append(energy_levels)
                completion_accuracies.append(accuracy)
            
            results[param_name].append({
                'value': value,
                'mean_energy_profile': np.mean(energy_profiles, axis=0),
                'std_energy_profile': np.std(energy_profiles, axis=0),
                'mean_accuracy': np.mean(completion_accuracies),
                'std_accuracy': np.std(completion_accuracies)
            })
    
    return results

def plot_energy_analysis(results):
    """Visualize energy dynamics analysis results"""
    n_params = len(results)
    fig, axes = plt.subplots(n_params, 2, figsize=(15, 5*n_params))
    
    for idx, (param_name, param_results) in enumerate(results.items()):
        # Plot energy profiles
        ax1 = axes[idx][0]
        for result in param_results:
            profile = result['mean_energy_profile']
            std = result['std_energy_profile']
            steps = np.arange(len(profile))
            
            ax1.plot(steps, profile, label=f'{param_name}={result["value"]:.3f}')
            ax1.fill_between(steps, 
                           profile - std,
                           profile + std,
                           alpha=0.2)
        
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Average Energy')
        ax1.set_title(f'Energy Dynamics for Different {param_name}')
        ax1.legend()
        
        # Plot accuracy vs parameter value
        ax2 = axes[idx][1]
        values = [r['value'] for r in param_results]
        accuracies = [r['mean_accuracy'] for r in param_results]
        errors = [r['std_accuracy'] for r in param_results]
        
        ax2.errorbar(values, accuracies, yerr=errors, marker='o')
        ax2.set_xlabel(param_name)
        ax2.set_ylabel('Completion Accuracy')
        ax2.set_title(f'Accuracy vs {param_name}')
    
    plt.tight_layout()
    plt.show()

# Define parameter ranges to test
param_sets = {
    'energy_decay': np.linspace(0.05, 0.25, 5),
    'energy_recovery': np.linspace(0.02, 0.12, 5),
    'threshold': np.linspace(0.1, 0.5, 5)
}

# Run analysis
results = analyze_energy_dynamics(param_sets)
plot_energy_analysis(results)

def find_optimal_parameters(results):
    """Find parameter values that maximize accuracy while maintaining stable energy"""
    optimal_params = {}
    
    for param_name, param_results in results.items():
        # Find parameter value with highest accuracy
        accuracies = [r['mean_accuracy'] for r in param_results]
        values = [r['value'] for r in param_results]
        
        # Calculate energy stability (lower std dev is more stable)
        energy_stability = [np.mean(r['std_energy_profile']) for r in param_results]
        
        # Combine accuracy and stability metrics
        combined_score = [acc - 0.2 * stab for acc, stab in zip(accuracies, energy_stability)]
        
        optimal_idx = np.argmax(combined_score)
        optimal_params[param_name] = values[optimal_idx]
    
    return optimal_params

# Find optimal parameters
optimal_params = find_optimal_parameters(results)
print("\nOptimal Parameters:")
for param, value in optimal_params.items():
    print(f"{param}: {value:.3f}")
