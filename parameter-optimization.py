import numpy as np
import matplotlib.pyplot as plt
from enhanced_neural_network import EnhancedPatternCompletionNetwork, create_enhanced_pattern

def grid_search_optimization():
    """Perform grid search to find optimal parameters"""
    # Define parameter search space
    param_grid = {
        'threshold': np.linspace(0.2, 0.4, 5),         # Around original 0.3
        'adaptation_rate': np.linspace(0.03, 0.07, 5), # Around original 0.05
        'energy_decay': np.linspace(0.1, 0.2, 5),      # Around original 0.15
        'energy_recovery': np.linspace(0.06, 0.1, 5),  # Around original 0.08
        'connection_strength': np.linspace(0.6, 0.8, 5) # Around original 0.7
    }
    
    # Initialize tracking
    best_score = -np.inf
    best_params = None
    best_metrics = None
    results = []
    
    # Test patterns
    size = (7, 7)
    patterns = {
        'A': create_enhanced_pattern(size, 'A'),
        'face': create_enhanced_pattern(size, 'face'),
        'complex': create_enhanced_pattern(size, 'complex')
    }
    
    # Grid search
    for threshold in param_grid['threshold']:
        for adapt_rate in param_grid['adaptation_rate']:
            for decay in param_grid['energy_decay']:
                for recovery in param_grid['energy_recovery']:
                    for strength in param_grid['connection_strength']:
                        params = {
                            'threshold': threshold,
                            'adaptation_rate': adapt_rate,
                            'energy_decay': decay,
                            'energy_recovery': recovery,
                            'connection_strength': strength,
                            'refractory_period': 2,  # Fixed
                            'memory_decay': 0.98     # Fixed
                        }
                        
                        # Test configuration
                        network = EnhancedPatternCompletionNetwork(size, params)
                        pattern_scores = []
                        
                        for pattern_type, pattern in patterns.items():
                            accuracies = []
                            energies = []
                            completion_times = []
                            
                            # Multiple trials
                            for removal_rate in [0.3, 0.5, 0.7]:
                                for _ in range(3):  # 3 trials per configuration
                                    start_time = time.time()
                                    
                                    known_positions = network.present_pattern(pattern, removal_rate)
                                    completed, confidence = network.update()
                                    
                                    completion_time = time.time() - start_time
                                    accuracy = np.mean(completed == pattern)
                                    
                                    # Calculate average energy
                                    energy_levels = [[network.neurons[i][j].energy 
                                                    for j in range(size[1])]
                                                   for i in range(size[0])]
                                    avg_energy = np.mean(energy_levels)
                                    
                                    accuracies.append(accuracy)
                                    energies.append(avg_energy)
                                    completion_times.append(completion_time)
                            
                            pattern_scores.append({
                                'pattern': pattern_type,
                                'mean_accuracy': np.mean(accuracies),
                                'mean_energy': np.mean(energies),
                                'mean_completion_time': np.mean(completion_times)
                            })
                        
                        # Calculate composite score
                        avg_accuracy = np.mean([s['mean_accuracy'] for s in pattern_scores])
                        avg_energy_efficiency = np.mean([s['mean_energy'] for s in pattern_scores])
                        avg_completion_time = np.mean([s['mean_completion_time'] for s in pattern_scores])
                        
                        # Score weights
                        score = (0.6 * avg_accuracy + 
                                0.25 * (1 - avg_energy_efficiency) +  # Lower energy is better
                                0.15 * (1 / avg_completion_time))     # Faster is better
                        
                        results.append({
                            'params': params.copy(),
                            'score': score,
                            'metrics': {
                                'accuracy': avg_accuracy,
                                'energy_efficiency': avg_energy_efficiency,
                                'completion_time': avg_completion_time,
                                'pattern_scores': pattern_scores
                            }
                        })
                        
                        if score > best_score:
                            best_score = score
                            best_params = params.copy()
                            best_metrics = results[-1]['metrics']
    
    return results, best_params, best_metrics

def plot_optimization_results(results):
    """Visualize optimization results"""
    # Extract key metrics
    accuracies = [r['metrics']['accuracy'] for r in results]
    energies = [r['metrics']['energy_efficiency'] for r in results]
    times = [r['metrics']['completion_time'] for r in results]
    scores = [r['score'] for r in results]
    
    plt.figure(figsize=(15, 10))
    
    # Plot distributions
    plt.subplot(221)
    plt.hist(accuracies, bins=20)
    plt.title('Accuracy Distribution')
    plt.xlabel('Accuracy')
    plt.ylabel('Count')
    
    plt.subplot(222)
    plt.hist(energies, bins=20)
    plt.title('Energy Efficiency Distribution')
    plt.xlabel('Average Energy')
    plt.ylabel('Count')
    
    plt.subplot(223)
    plt.hist(times, bins=20)
    plt.title('Completion Time Distribution')
    plt.xlabel('Time (s)')
    plt.ylabel('Count')
    
    plt.subplot(224)
    plt.hist(scores, bins=20)
    plt.title('Overall Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.show()

# Run optimization
print("Starting parameter optimization...")
results, best_params, best_metrics = grid_search_optimization()

print("\nOptimal Parameters Found:")
for param, value in best_params.items():
    print(f"{param}: {value:.4f}")

print("\nPerformance Metrics:")
print(f"Average Accuracy: {best_metrics['accuracy']:.4f}")
print(f"Energy Efficiency: {best_metrics['energy_efficiency']:.4f}")
print(f"Average Completion Time: {best_metrics['completion_time']:.4f}")

print("\nPattern-Specific Performance:")
for pattern_score in best_metrics['pattern_scores']:
    print(f"\n{pattern_score['pattern']} Pattern:")
    print(f"  Accuracy: {pattern_score['mean_accuracy']:.4f}")
    print(f"  Energy Usage: {pattern_score['mean_energy']:.4f}")
    print(f"  Completion Time: {pattern_score['mean_completion_time']:.4f}")

# Plot results
plot_optimization_results(results)
