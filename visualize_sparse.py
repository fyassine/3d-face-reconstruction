import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
data = pd.read_csv('build/sparse_optimization_loss.csv')

# Create figure and axis
plt.figure(figsize=(12, 6))

# Initialize variables
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Different colors for each iteration

# Plot the cost values
plt.plot(data['Iteration'], data['Cost'], 
         color=colors[0], marker='o', 
         label='Optimization Progress',
         linewidth=2)

# Customize the plot
plt.title('Sparse Optimization Cost per Iteration', fontsize=14, pad=20)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Cost (log scale)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Use log scale for y-axis since sparse optimization values are very small
plt.yscale('log')

# Add text annotations for final cost value
final_cost = data['Cost'].iloc[-1]
plt.text(0.02, 0.98, f'Final Cost: {final_cost:.8f}', 
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
         fontsize=10)

# Add convergence metrics
final_gradient = data['GradientNorm'].iloc[-1]
plt.text(0.02, 0.90, f'Final Gradient Norm: {final_gradient:.8f}', 
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
         fontsize=10)

# Adjust layout and save
plt.tight_layout()
plt.savefig('sparse_optimization_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a second figure for other metrics
plt.figure(figsize=(12, 6))

# Plot other optimization metrics
metrics = ['GradientNorm', 'StepNorm', 'TrustRegionRadius']
for i, metric in enumerate(metrics):
    plt.plot(data['Iteration'], data[metric], 
            color=colors[i], marker='o', 
            label=metric,
            linewidth=2)

plt.title('Sparse Optimization Metrics', fontsize=14, pad=20)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Value (log scale)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.yscale('log')

# Adjust layout and save
plt.tight_layout()
plt.savefig('sparse_optimization_metrics.png', dpi=300, bbox_inches='tight')
plt.close()