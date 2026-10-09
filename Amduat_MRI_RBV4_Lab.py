import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
from typing import List, Tuple
import os
import crystal_lab as lab   # quantum scars / birthmark / chiral phonons

# Try importing PIL for image handling
try:
    from PIL import Image
except ImportError:
    print("Error: PIL (Pillow) library not found. Please install it using 'pip install Pillow'")
    exit()

# ============================================================================
# PART 1: THE SACRED ENCODER (Text Logic)
# ============================================================================

class SacredEncoder:
    def __init__(self):
        self.cymatic_map = {
            'a': 110, 'i': 220, 'u': 90, 'y': 210, 'w': 100,
            '3': 50, "'": 60, '.': 0, 's': 800, 'z': 750, 'f': 400, 
            'h': 300, 'H': 350, 'x': 600, 'X': 650, 'k': 550, 'q': 500, 
            'g': 120, 't': 450, 'T': 480, 'd': 130, 'D': 140, 'p': 400, 
            'b': 110, 'r': 320, 'l': 310, 'm': 200, 'n': 210
        }

    def normalize_input(self, text: str) -> str:
        return text.lower().strip()

    def get_cymatic_frequencies(self, text: str) -> List[float]:
        text = self.normalize_input(text)
        return [self.cymatic_map.get(c, 100) for c in text if c != '.']

# ============================================================================
# PART 2: IMAGE PROCESSOR (With Depth Control)
# ============================================================================

class ImageProcessor:
    @staticmethod
    def load_and_process(filepath: str, resolution: int = 100, depth: int = 20):
        print(f" > Loading artifact: {filepath}")
        try:
            img = Image.open(filepath).convert("RGBA")
        except Exception as e:
            print(f"Failed to load image: {e}")
            return None, None

        # Resize for voxel grid
        img = img.resize((resolution, resolution), Image.Resampling.BILINEAR)
        data = np.array(img) / 255.0 # Normalize 0-1

        # Calculate Luminance for Structure (Standard Grayscale conversion)
        luminance = (data[:,:,0] * 0.299 + data[:,:,1] * 0.587 + data[:,:,2] * 0.114)
        alpha = data[:,:,3]
        
        # Create solid structure where image is visible/bright
        # Adjust threshold based on your specific image darkness
        structure_slice = (luminance > 0.15) & (alpha > 0.1)

        # Initialize 3D Volume arrays
        volume = np.zeros((resolution, resolution, depth), dtype=bool)
        voxel_colors = np.zeros((resolution, resolution, depth, 4))

        print(f" > Extruding Z-Axis to depth: {depth}...")
        
        # Extrude the 2D image along the Z-axis
        for z in range(depth):
            volume[:, :, z] = structure_slice
            voxel_colors[:, :, z, :] = data 

        return volume, voxel_colors

# ============================================================================
# PART 3: TEXT GENERATOR
# ============================================================================

def generate_chladni_volume(text, encoder, depth=20):
    freqs = encoder.get_cymatic_frequencies(text)
    res = 64
    
    # If user selected a specific depth, use it; otherwise dynamic based on text length
    actual_depth = depth if depth > 0 else len(freqs) * 5
    
    volume = np.zeros((res, res, actual_depth), dtype=bool)
    
    x, y = np.linspace(-1, 1, res), np.linspace(-1, 1, res)
    X, Y = np.meshgrid(x, y)
    
    # Generate layers
    z_layer = 0
    # Loop through frequencies, cycling if depth is larger than text length
    while z_layer < actual_depth:
        for f in freqs:
            if z_layer >= actual_depth: break
            
            n, m = (f / 50.0) + 1, (f / 80.0) + 1
            Z = np.cos(n*np.pi*X)*np.cos(m*np.pi*Y) - np.cos(m*np.pi*X)*np.cos(n*np.pi*Y)
            structure = (np.abs(Z) < 0.2) 
            
            # Add thickness per letter
            for _ in range(3): 
                if z_layer < actual_depth:
                    volume[:, :, z_layer] = structure
                    z_layer += 1

    return volume, None 

# ============================================================================
# PART 4: OBJ EXPORTER (Marching Cubes)
# ============================================================================

def save_obj(volume, colors, filename="artifact.obj"):
    print(" > Generating 3D Mesh (Marching Cubes)...")
    
    # 1. Generate Mesh
    try:
        # marching_cubes returns: verts, faces, normals, values
        verts, faces, normals, values = measure.marching_cubes(volume, level=0.5)
    except RuntimeError:
        print(" ! Error: Volume is empty or too complex. Try different settings.")
        return
    except Exception as e:
        print(f" ! Error during meshing: {e}")
        return

    # 2. Map Colors
    print(" > Mapping thermal data to mesh vertices...")
    vertex_lines = []
    
    depth_limit = volume.shape[2] - 1
    res_limit = volume.shape[0] - 1
    
    # Pre-calculate colors if they exist
    has_color = (colors is not None)
    
    for v in verts:
        # Get coordinates
        x, y, z = v[0], v[1], v[2]
        
        # Format vertex line: v X Y Z [R G B]
        line = f"v {x:.4f} {y:.4f} {z:.4f}"
        
        if has_color:
            # Clamp integer coordinates to avoid index errors
            ix = min(int(round(x)), res_limit)
            iy = min(int(round(y)), res_limit)
            iz = min(int(round(z)), depth_limit)
            
            rgba = colors[ix, iy, iz]
            line += f" {rgba[0]:.4f} {rgba[1]:.4f} {rgba[2]:.4f}"
            
        vertex_lines.append(line)

    # 3. Write File
    print(f" > Writing to file: {filename}...")
    with open(filename, 'w') as f:
        f.write(f"# Duat-MRI Export: {filename}\n")
        
        for line in vertex_lines:
            f.write(line + "\n")
            
        for face in faces:
            # OBJ indices start at 1
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            
    print(f" > SUCCESS: Saved as {filename}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("--- AMDUAT MRI RECONSTRUCTION v3 ---")
    print("[1] Text Input (Chladni)")
    print("[2] Image Upload (Quantum Tunneling)")
    print("[3] Quantum Scar eigenstate stack (stadium billiard, z = energy)")
    print("[4] Chiral phonon crystal pulse (z = time, light-driven)")
    
    mode = input("\nSelect Mode: ").strip()
    
    vol = None
    cols = None
    name = "export"
    
    if mode == '1':
        text = input("Enter Text: ")
        d_input = input("Enter Z-Depth (default 30): ").strip()
        depth = int(d_input) if d_input.isdigit() else 30
        
        encoder = SacredEncoder()
        vol, cols = generate_chladni_volume(text, encoder, depth=depth)
        name = text
        
    elif mode == '3':
        text = input("Enter seed text: ")
        d_input = input("Enter Z-Depth = number of eigenstates (default 30): ").strip()
        depth = int(d_input) if d_input.isdigit() else 30
        print(" > Diagonalising stadium billiard...")
        vol, cols = lab.scar_volume(text, depth=depth, res=64)
        name = text + "_scar"

    elif mode == '4':
        text = input("Enter seed text: ")
        d_input = input("Enter Z-Depth = time slices (default 30): ").strip()
        depth = int(d_input) if d_input.isdigit() else 30
        hel = input("Light helicity (+ / - / enter for +): ").strip()
        print(" > Driving crystal with THz pulse...")
        vol, cols = lab.phonon_volume(text, depth=depth, res=64, helicity=-1 if hel == '-' else 1)
        name = text + "_phonon"

    elif mode == '2':
        path = input("Enter PNG path: ").strip().replace('"', '').replace("'", "")
        
        if os.path.exists(path):
            d_input = input("Enter Z-Depth (default 15): ").strip()
            depth = int(d_input) if d_input.isdigit() else 15
            
            # Using 100 resolution for a good balance of detail and speed
            vol, cols = ImageProcessor.load_and_process(path, resolution=100, depth=depth)
            name = os.path.splitext(os.path.basename(path))[0]
        else:
            print("File not found.")

    if vol is not None:
        # Prompt for Export
        save = input(f"\nSave '{name}' as 3D Object (.obj)? (y/n): ").lower()
        if save == 'y':
            filename = f"{name}.obj"
            save_obj(vol, cols, filename)
        else:
            # Fallback to visualization
            print(" > Skipping export, showing preview...")
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_facecolor('black')
            if cols is not None:
                step = 2
                ax.voxels(vol[::step,::step,::step], facecolors=cols[::step,::step,::step])
            else:
                ax.voxels(vol, edgecolor='k', cmap='magma')
            plt.show()