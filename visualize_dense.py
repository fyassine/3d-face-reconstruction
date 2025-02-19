import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
data = pd.read_csv('build/dense_optimization_loss.csv')

# Create figure and axis
plt.figure(figsize=(12, 6))

# Initialize variables
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Different colors for each iteration
current_iteration = 0
start_idx = 0

# Find where new iterations start (where Iteration resets to 0)
iteration_starts = [0] + list(data[data['Iteration'] == 0].index)[1:]
iteration_ends = iteration_starts[1:] + [len(data)]

# Plot each iteration block with different colors
for i, (start, end) in enumerate(zip(iteration_starts, iteration_ends)):
    subset = data.iloc[start:end]
    color = colors[i % len(colors)]
    plt.plot(range(start, end), subset['Cost'], 
             color=color, marker='o', 
             label=f'Iteration Block {i+1}',
             linewidth=2)

# Customize the plot
plt.title('Optimization Cost per Step', fontsize=14, pad=20)
plt.xlabel('Step Number', fontsize=12)
plt.ylabel('Cost', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Add text annotations for final cost value
final_cost = data['Cost'].iloc[-1]
plt.text(0.02, 0.98, f'Final Cost: {final_cost:.2f}', 
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
         fontsize=10)

# Adjust layout and save
plt.tight_layout()
plt.savefig('optimization_cost_visualization.png', dpi=300, bbox_inches='tight')
plt.close()