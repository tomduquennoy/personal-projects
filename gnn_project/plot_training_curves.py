"""
Plot training curves from TensorBoard event files
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_tensorboard_data(log_dir):
    """Load data from TensorBoard event files"""
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    
    if not event_files:
        print(f"No event files found in {log_dir}")
        return None
    
    ea = EventAccumulator(event_files[0])
    ea.Reload()
    
    data = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = {'steps': steps, 'values': values}
    
    return data

def plot_training_curves(logs_base_dir):
    """Plot training curves for all runs"""
    # Find all run directories
    run_dirs = sorted([d for d in glob.glob(os.path.join(logs_base_dir, "*/RUN_*")) 
                      if os.path.isdir(d)])
    
    if not run_dirs:
        # Try parent directories
        run_dirs = sorted([d for d in glob.glob(os.path.join(logs_base_dir, "*")) 
                          if os.path.isdir(d) and not d.endswith('.git')])
    
    print(f"Found {len(run_dirs)} run(s)")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('GCN Training on ZINC Dataset', fontsize=16)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(run_dirs)))
    
    for idx, run_dir in enumerate(run_dirs):
        run_name = os.path.basename(run_dir)
        print(f"Loading {run_name}...")
        
        data = load_tensorboard_data(run_dir)
        if data is None:
            continue
        
        color = colors[idx]
        
        # Plot train loss
        if 'train/_loss' in data:
            axes[0, 0].plot(data['train/_loss']['steps'], 
                           data['train/_loss']['values'], 
                           label=run_name, color=color, linewidth=2)
        
        # Plot val loss
        if 'val/_loss' in data:
            axes[0, 1].plot(data['val/_loss']['steps'], 
                           data['val/_loss']['values'], 
                           label=run_name, color=color, linewidth=2)
        
        # Plot train MAE
        if 'train/_mae' in data:
            axes[1, 0].plot(data['train/_mae']['steps'], 
                           data['train/_mae']['values'], 
                           label=run_name, color=color, linewidth=2)
        
        # Plot test MAE
        if 'test/_mae' in data:
            axes[1, 1].plot(data['test/_mae']['steps'], 
                           data['test/_mae']['values'], 
                           label=run_name, color=color, linewidth=2)
    
    # Configure subplots
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].set_title('Training MAE')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].set_title('Test MAE')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'training_curves.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    
    plt.show()

if __name__ == "__main__":
    logs_dir = "out/molecules_graph_regression/logs"
    plot_training_curves(logs_dir)
