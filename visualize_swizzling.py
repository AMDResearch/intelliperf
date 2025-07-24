import os
import re
import ast
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path 

def parse_swizzling_pattern(file_path):
    """Parses the swizzling pattern from a log file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    match = re.search(r'(\[\[\[.*\]\]\])', content, re.DOTALL)
    if not match:
        # try to find 2d pattern
        match = re.search(r'(\[\[.*\]\])', content, re.DOTALL)
        if not match:
            # try to find 1d pattern
            match = re.search(r'(\[.*\])', content, re.DOTALL)
            if not match:
                return None
    
    try:
        pattern_str = match.group(1)
        # Fix potential formatting issues for ast.literal_eval
        pattern_str = re.sub(r'\s+', ' ', pattern_str)
        pattern_str = pattern_str.replace(' ', ', ')
        pattern_str = pattern_str.replace('[,', '[')
        pattern_str = pattern_str.replace(',]', ']')
        
        # A bit of a hack to handle parsing, might need to be more robust
        pattern_str = re.sub(r',+', ',', pattern_str)
        pattern_str = pattern_str.replace('[,', '[')
        while '[, ' in pattern_str:
            pattern_str = pattern_str.replace('[, ', '[')

        # To handle cases like '[1, 2, 3, ...]'
        pattern_str = pattern_str.replace('...,', '')
        pattern_str = pattern_str.replace('... ', '')
        
        # Find the last valid closing bracket
        open_brackets = 0
        last_valid_index = -1
        for i, char in enumerate(pattern_str):
            if char == '[':
                open_brackets += 1
            elif char == ']':
                open_brackets -= 1
                if open_brackets == 0:
                    last_valid_index = i
                    break
        
        if last_valid_index != -1:
            pattern_str = pattern_str[:last_valid_index+1]

        # Final cleanup
        pattern_str = pattern_str.strip()
        if pattern_str.endswith(','):
            pattern_str = pattern_str[:-1]

        return np.array(ast.literal_eval(pattern_str))
    except (ValueError, SyntaxError) as e:
        print(f"Could not parse pattern from {file_path}: {e}")
        return None

def visualize_mapping(swizzled_pids, kernel_name, output_dir, num_xcds=32):
    """Visualizes the PID to XCD mapping before and after swizzling."""
    
    pid_shape = swizzled_pids.shape
    dim = len(pid_shape)
    
    if dim == 1:
        old_pids_x = np.arange(pid_shape[0])
        old_linear_pids = old_pids_x
    elif dim == 2:
        old_pids_y, old_pids_x = np.indices(pid_shape)
        old_linear_pids = (old_pids_y * pid_shape[1] + old_pids_x).flatten()
    elif dim == 3:
        old_pids_z, old_pids_y, old_pids_x = np.indices(pid_shape)
        old_linear_pids = (old_pids_z * pid_shape[1] * pid_shape[2] + old_pids_y * pid_shape[2] + old_pids_x).flatten()
    else:
        print(f"Unsupported dimension {dim} for {kernel_name}")
        return

    xcd_before = old_linear_pids % num_xcds
    xcd_after = swizzled_pids.flatten() % num_xcds
    
    # Create a consistent color map for all XCDs
    colors = plt.cm.get_cmap('gist_rainbow', num_xcds)
    norm = mcolors.Normalize(vmin=0, vmax=num_xcds-1)

    fig = plt.figure(figsize=(20, 8))
    fig.suptitle(f'PID to XCD Mapping for {kernel_name} (num_xcds={num_xcds})', fontsize=16)

    # Before swizzling
    if dim == 3:
        ax1 = fig.add_subplot(121, projection='3d')
        scatter1 = ax1.scatter(old_pids_x.flatten(), old_pids_y.flatten(), old_pids_z.flatten(), c=xcd_before, cmap=colors, norm=norm)
        ax1.set_zlabel('PID Z')
    else:
        ax1 = fig.add_subplot(121)
        if dim == 1:
            scatter1 = ax1.scatter(old_pids_x, np.zeros_like(old_pids_x), c=xcd_before, cmap=colors, norm=norm)
        else: # dim == 2
            scatter1 = ax1.scatter(old_pids_x.flatten(), old_pids_y.flatten(), c=xcd_before, cmap=colors, norm=norm)
    
    ax1.set_title('Before Swizzling (Round-Robin)')
    ax1.set_xlabel('PID X')
    if dim > 1:
        ax1.set_ylabel('PID Y')


    # After swizzling
    if dim == 3:
        ax2 = fig.add_subplot(122, projection='3d')
        scatter2 = ax2.scatter(old_pids_x.flatten(), old_pids_y.flatten(), old_pids_z.flatten(), c=xcd_after, cmap=colors, norm=norm)
        ax2.set_zlabel('PID Z')
    else:
        ax2 = fig.add_subplot(122)
        if dim == 1:
            scatter2 = ax2.scatter(old_pids_x, np.zeros_like(old_pids_x), c=xcd_after, cmap=colors, norm=norm)
        else: # dim == 2
            scatter2 = ax2.scatter(old_pids_x.flatten(), old_pids_y.flatten(), c=xcd_after, cmap=colors, norm=norm)
            
    ax2.set_title('After Swizzling')
    ax2.set_xlabel('PID X')
    if dim > 1:
        ax2.set_ylabel('PID Y')
    
    # Add a colorbar
    cbar = fig.colorbar(scatter1, ax=[ax1, ax2], orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('XCD ID')
    
    output_path = output_dir / f"{kernel_name}_swizzling_viz.png"
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved visualization to {output_path}")


def main():
    base_dir = Path(__file__).parent.parent
    log_dirs = [
        base_dir / "examples/triton/autogen_10/swizzling_logs_2",
        base_dir / "examples/triton/autogen_science_10/swizzling_logs_2"
    ]
    output_dir = base_dir / "swizzling_visualizations"
    output_dir.mkdir(exist_ok=True)
    
    num_xcds = 32 # or get from args

    for log_dir in log_dirs:
        if not log_dir.exists():
            print(f"Directory not found: {log_dir}")
            continue
            
        for log_file in log_dir.glob("*_log_final.txt"):
            kernel_name = log_file.name.replace("_log_final.txt", "")
            print(f"Processing {kernel_name}...")
            
            pattern = parse_swizzling_pattern(log_file)
            
            if pattern is not None:
                visualize_mapping(pattern, kernel_name, output_dir, num_xcds)
            else:
                print(f"No valid pattern found for {kernel_name}")

if __name__ == "__main__":
    main() 