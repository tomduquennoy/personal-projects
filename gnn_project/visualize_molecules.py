"""
Visualize ZINC molecules as graphs and model predictions
"""
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Rectangle
import dgl

# Import data loading
from data.data import LoadData

# Atom type mapping for ZINC dataset
ATOM_TYPES = {
    0: 'C',   # Carbon
    1: 'N',   # Nitrogen  
    2: 'O',   # Oxygen
    3: 'F',   # Fluorine
    4: 'P',   # Phosphorus
    5: 'S',   # Sulfur
    6: 'Cl',  # Chlorine
    7: 'Br',  # Bromine
    8: 'I',   # Iodine
}

# Bond type mapping
BOND_TYPES = {
    0: 'none',
    1: 'single',
    2: 'double',
    3: 'triple',
}

# Atom colors
ATOM_COLORS = {
    'C': '#909090',   # Gray
    'N': '#3050F8',   # Blue
    'O': '#FF0D0D',   # Red
    'F': '#90E050',   # Light green
    'P': '#FF8000',   # Orange
    'S': '#FFFF30',   # Yellow
    'Cl': '#1FF01F',  # Green
    'Br': '#A62929',  # Dark red
    'I': '#940094',   # Purple
}

def dgl_to_networkx(dgl_graph):
    """Convert DGL graph to NetworkX for visualization"""
    G = nx.Graph()
    
    # Add nodes with features
    num_nodes = dgl_graph.number_of_nodes()
    node_features = dgl_graph.ndata['feat'].numpy()
    
    for i in range(num_nodes):
        atom_type = node_features[i]
        atom_symbol = ATOM_TYPES.get(atom_type, f'Atom{atom_type}')
        G.add_node(i, atom=atom_symbol, atom_type=int(atom_type))
    
    # Add edges with features
    src, dst = dgl_graph.edges()
    edge_features = dgl_graph.edata['feat'].numpy()
    
    for i, (s, d) in enumerate(zip(src.numpy(), dst.numpy())):
        if s < d:  # Avoid duplicate edges
            bond_type = edge_features[i]
            bond_name = BOND_TYPES.get(bond_type, 'unknown')
            G.add_edge(s, d, bond=bond_name, bond_type=int(bond_type))
    
    return G

def visualize_molecule(graph, label, idx, save_path=None):
    """Visualize a single molecule"""
    G = dgl_to_networkx(graph)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Layout
    pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    
    # Get node colors based on atom type
    node_colors = [ATOM_COLORS.get(G.nodes[node]['atom'], '#CCCCCC') for node in G.nodes()]
    node_labels = {node: G.nodes[node]['atom'] for node in G.nodes()}
    
    # Draw graph
    nx.draw_networkx_edges(G, pos, ax=ax, width=2, alpha=0.6, edge_color='#666666')
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                           node_size=800, edgecolors='black', linewidths=2)
    nx.draw_networkx_labels(G, pos, node_labels, ax=ax, font_size=12, font_weight='bold')
    
    # Add bond type labels
    edge_labels = {(u, v): G[u][v]['bond'] for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=9)
    
    # Convert label to float if it's a tensor
    label_val = float(label) if torch.is_tensor(label) else label
    ax.set_title(f'Molecule #{idx}\n{G.number_of_nodes()} atoms, {G.number_of_edges()} bonds\nTarget: {label_val:.4f}', 
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    return fig

def visualize_multiple_molecules(dataset, num_samples=6, split='train'):
    """Visualize multiple molecules in a grid"""
    if split == 'train':
        data = dataset.train
    elif split == 'val':
        data = dataset.val
    else:
        data = dataset.test
    
    # Sample random molecules
    indices = np.random.choice(len(data), min(num_samples, len(data)), replace=False)
    
    # Create grid
    rows = (num_samples + 2) // 3
    cols = min(3, num_samples)
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, idx in enumerate(indices):
        graph, label = data[idx]
        label_val = float(label) if torch.is_tensor(label) else label
        G = dgl_to_networkx(graph)
        
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        # Layout
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
        
        # Node colors
        node_colors = [ATOM_COLORS.get(G.nodes[node]['atom'], '#CCCCCC') for node in G.nodes()]
        node_labels = {node: G.nodes[node]['atom'] for node in G.nodes()}
        
        # Draw
        nx.draw_networkx_edges(G, pos, ax=ax, width=2, alpha=0.5, edge_color='#666666')
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                              node_size=600, edgecolors='black', linewidths=1.5)
        nx.draw_networkx_labels(G, pos, node_labels, ax=ax, font_size=10, font_weight='bold')
        
        ax.set_title(f'Molecule #{idx}\nAtoms: {G.number_of_nodes()}, Label: {label_val:.3f}', 
                    fontsize=11)
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(len(indices), rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'ZINC Dataset - {split.upper()} Split', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = f'molecules_visualization_{split}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {save_path}")
    
    plt.show()
    return fig

def print_dataset_statistics(dataset):
    """Print statistics about the dataset"""
    print("="*60)
    print("ZINC DATASET STATISTICS")
    print("="*60)
    
    for split_name, split_data in [('Train', dataset.train), 
                                     ('Validation', dataset.val), 
                                     ('Test', dataset.test)]:
        print(f"\n{split_name} Set:")
        print(f"  Number of molecules: {len(split_data)}")
        
        # Compute statistics
        num_nodes = [split_data[i][0].number_of_nodes() for i in range(len(split_data))]
        num_edges = [split_data[i][0].number_of_edges() for i in range(len(split_data))]
        labels = [float(split_data[i][1]) if torch.is_tensor(split_data[i][1]) else split_data[i][1] 
                  for i in range(len(split_data))]
        
        print(f"  Nodes (atoms) per molecule:")
        print(f"    Mean: {np.mean(num_nodes):.2f} ± {np.std(num_nodes):.2f}")
        print(f"    Range: [{min(num_nodes)}, {max(num_nodes)}]")
        
        print(f"  Edges (bonds) per molecule:")
        print(f"    Mean: {np.mean(num_edges):.2f} ± {np.std(num_edges):.2f}")
        print(f"    Range: [{min(num_edges)}, {max(num_edges)}]")
        
        print(f"  Target property (logP_SA_cycle_normalized):")
        print(f"    Mean: {np.mean(labels):.4f} ± {np.std(labels):.4f}")
        print(f"    Range: [{min(labels):.4f}, {max(labels):.4f}]")
    
    print(f"\nAtom types: {dataset.num_atom_type}")
    print(f"Bond types: {dataset.num_bond_type}")
    print("="*60)

def visualize_atom_distribution(dataset):
    """Visualize distribution of atom types in the dataset"""
    atom_counts = {i: 0 for i in range(dataset.num_atom_type)}
    
    # Count atoms in training set
    for i in range(len(dataset.train)):
        graph, _ = dataset.train[i]
        node_features = graph.ndata['feat'].numpy()
        for atom_type in node_features:
            atom_counts[atom_type] += 1
    
    # Create bar plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    atoms = [ATOM_TYPES.get(i, f'Type{i}') for i in range(dataset.num_atom_type)]
    counts = [atom_counts[i] for i in range(dataset.num_atom_type)]
    colors = [ATOM_COLORS.get(atom, '#CCCCCC') for atom in atoms]
    
    bars = ax.bar(atoms, counts, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Atom Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Atom Type Distribution in ZINC Training Set', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('atom_distribution.png', dpi=150, bbox_inches='tight')
    print("\nAtom distribution saved to atom_distribution.png")
    plt.show()
    
    return fig

if __name__ == "__main__":
    print("Loading ZINC dataset...")
    dataset = LoadData('ZINC')
    
    # Print statistics
    print_dataset_statistics(dataset)
    
    # Visualize atom distribution
    print("\nGenerating atom distribution plot...")
    visualize_atom_distribution(dataset)
    
    # Visualize sample molecules
    print("\nGenerating molecule visualizations...")
    print("Train set samples:")
    visualize_multiple_molecules(dataset, num_samples=6, split='train')
    
    print("\nTest set samples:")
    visualize_multiple_molecules(dataset, num_samples=6, split='test')
    
    print("\nDone!")
