import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_plot_loss(sparse_file='sparse_optimization_loss.csv', 
                      dense_file='dense_optimization_loss.csv'):
    # Set the style for better-looking plots
    sns.set_style("whitegrid")  # Changed from plt.style.use('seaborn')
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Try to load both files
    try:
        sparse_data = pd.read_csv(sparse_file)
        has_sparse = True
    except FileNotFoundError:
        print(f"Warning: {sparse_file} not found")
        has_sparse = False
        
    try:
        dense_data = pd.read_csv(dense_file)
        has_dense = True
    except FileNotFoundError:
        print(f"Warning: {dense_file} not found")
        has_dense = False
    
    if not (has_sparse or has_dense):
        print("Error: No data files found!")
        return
        
    # Plot Cost
    plt.subplot(2, 2, 1)
    if has_sparse:
        plt.plot(sparse_data['Iteration'], sparse_data['Cost'], 
                label='Sparse Optimization', marker='o')
    if has_dense:
        plt.plot(dense_data['Iteration'], dense_data['Cost'], 
                label='Dense Optimization', marker='o')
    plt.yscale('log')
    plt.title('Cost vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Cost (log scale)')
    plt.legend()
    plt.grid(True)
    
    # Plot Gradient Norm
    plt.subplot(2, 2, 2)
    if has_sparse:
        plt.plot(sparse_data['Iteration'], sparse_data['GradientNorm'], 
                label='Sparse Optimization', marker='o')
    if has_dense:
        plt.plot(dense_data['Iteration'], dense_data['GradientNorm'], 
                label='Dense Optimization', marker='o')
    plt.yscale('log')
    plt.title('Gradient Norm vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm (log scale)')
    plt.legend()
    plt.grid(True)
    
    # Plot Step Norm
    plt.subplot(2, 2, 3)
    if has_sparse:
        plt.plot(sparse_data['Iteration'], sparse_data['StepNorm'], 
                label='Sparse Optimization', marker='o')
    if has_dense:
        plt.plot(dense_data['Iteration'], dense_data['StepNorm'], 
                label='Dense Optimization', marker='o')
    plt.yscale('log')
    plt.title('Step Norm vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Step Norm (log scale)')
    plt.legend()
    plt.grid(True)
    
    # Plot Trust Region Radius
    plt.subplot(2, 2, 4)
    if has_sparse:
        plt.plot(sparse_data['Iteration'], sparse_data['TrustRegionRadius'], 
                label='Sparse Optimization', marker='o')
    if has_dense:
        plt.plot(dense_data['Iteration'], dense_data['TrustRegionRadius'], 
                label='Dense Optimization', marker='o')
    plt.yscale('log')
    plt.title('Trust Region Radius vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Trust Region Radius (log scale)')
    plt.legend()
    plt.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('optimization_loss_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Plots have been saved to 'optimization_loss_plots.png'")

if __name__ == "__main__":
    load_and_plot_loss()