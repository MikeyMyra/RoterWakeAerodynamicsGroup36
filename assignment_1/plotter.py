import matplotlib.pyplot as plt

def plot(name, x_data, y_data_list, labels, xlabel="x", ylabel="y", colors=None):
    """Plot multiple datasets on the same figure"""
    plt.figure(figsize=(10, 6))
    if colors is None:
        colors = ['b', 'r', 'g', 'orange', 'purple', 'brown']

    if type(x_data[0]) != list:
        for i, (y_data, label) in enumerate(zip(y_data_list, labels)):
            plt.plot(x_data, y_data, marker='o', linestyle='-',
                    color=colors[i % len(colors)], label=label, markersize=4)
    else:
        for i, (x, y_data, label) in enumerate(zip(x_data, y_data_list, labels)):
            plt.plot(x, y_data, marker='o', linestyle='-',
                    color=colors[i % len(colors)], label=label, markersize=4)
    
    plt.title(name, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()