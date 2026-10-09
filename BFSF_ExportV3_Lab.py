import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.ndimage import gaussian_filter
import sys
import os 
import crystal_lab as lab   # quantum scars / birthmark / chiral phonons

# --- QUANTUM LAB SWITCHES ---
USE_QUANTUM_SCAR = True      # medium carries a scarred-eigenstate landscape (seeded by your text = birthmark)
USE_CHIRAL_PHONON = True     # circularly travelling lattice distortion (light-driven chiral phonon)
HANDEDNESS = +1              # +1 counter-clockwise, -1 clockwise (swap to see the mirror-image response)
SCAR_STRENGTH = 0.8
PHONON_AMP = 0.6

# --- 1. CORE ALGORITHMS ---

def generate_fibonacci_spiral(n_points, radius_max, center_y, center_z):
    """
    Creates a phyllotaxis (sunflower) pattern for the laser source.
    Ensures rays are evenly packed for maximum definition.
    """
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden Angle
    indices = np.arange(n_points)
    radius = radius_max * np.sqrt(indices / float(n_points))
    theta = phi * indices
    
    y = radius * np.cos(theta) + center_y
    z = radius * np.sin(theta) + center_z
    return y, z

def generate_fractal_mesh_potential(shape, octaves=4, persistence=0.5, lacunarity=2.0, scale=10.0):
    """
    Generates the "Medium" using 4D Fractal Brownian Motion.
    This bakes the infinite-like detail into the simulation grid.
    """
    print(f"   -> Synthesizing Fractal Medium ({octaves} octaves)...")
    field = np.zeros(shape)
    frequency = 5.0
    amplitude = 5.0
    
    for _ in range(octaves):
        noise_layer = np.random.randn(*shape)
        sigma = scale / frequency
        if sigma < 0.5: sigma = 0.5
        
        smoothed = gaussian_filter(noise_layer, sigma=sigma)
        field += smoothed * amplitude
        
        amplitude *= persistence
        frequency *= lacunarity 
        
    return field

def export_to_obj(filename, hx, hy, hz):
    """
    Exports the simulation as a 3D .obj file (Lines).
    Compatible with Blender, C4D, Maya, etc.
    """
    n_steps, n_rays = hx.shape
    print(f"\n   -> Exporting 3D Object to '{filename}'...")
    print(f"      (Geometry: {n_rays} rays, {n_steps} vertices per ray)")
    
    with open(filename, 'w') as f:
        f.write(f"# Branched Flow Fractal Simulation\n")
        f.write(f"o FractalLaserBeam\n")
        
        # 1. Write Vertices
        # We iterate ray by ray to keep data organized
        vertex_count = 0
        for r in range(n_rays):
            for t in range(n_steps):
                # Write x, y, z coordinates
                f.write(f"v {hx[t, r]:.4f} {hy[t, r]:.4f} {hz[t, r]:.4f}\n")
            
            # Progress bar for large files
            if r % 500 == 0:
                sys.stdout.write(f"\r      Writing vertices... {int((r/n_rays)*100)}%")
                sys.stdout.flush()
        
        print("\n      Writing connectivity lines...")
        
        # 2. Write Lines (Connections)
        # Connect the vertices into continuous strands
        for r in range(n_rays):
            f.write("l")
            start_idx = (r * n_steps) + 1 # OBJ indices are 1-based
            for t in range(n_steps):
                f.write(f" {start_idx + t}")
            f.write("\n")
            
    print(f"   -> Export Complete: Saved as {filename}")

# --- 2. MAIN SIMULATION LOOP ---

def run_simulation(seed_text=None):
    # --- CONFIGURATION (High Quality) ---
    n_rays = 4000           # High ray count for solid looking mesh
    n_steps = 800           # High step count to capture fractal details
    step_size = 0.5
    chaos_strength = 3.0
    grid_size = 120
    
    if seed_text:
        seed = sum(ord(c) for c in seed_text)
        np.random.seed(seed)

    # 1. Generate Environment
    potential = generate_fractal_mesh_potential(
        (grid_size, grid_size, grid_size), 
        octaves=4, 
        persistence=0.6, 
        scale=8.0
    )
    potential *= chaos_strength
    grad_z, grad_y, grad_x = np.gradient(potential)

    # Quantum scar landscape: extruded along the beam axis (x), varies in the (y, z) cross-section
    if USE_QUANTUM_SCAR:
        print("   -> Solving quantum scar eigenstate for the medium cross-section...")
        S = lab.scar_potential_slice(seed_text or "", grid_size)
        scar_gy, scar_gz = np.gradient(S)

    # 2. Initialize Rays (Fibonacci)
    print("   -> Initializing Emitters...")
    pos_x = np.zeros(n_rays)
    center = grid_size // 2
    pos_y, pos_z = generate_fibonacci_spiral(n_rays, 8.0, center, center) # beam_width=8.0
    
    vel_x = np.ones(n_rays)
    vel_y = np.random.randn(n_rays) * 0.005
    vel_z = np.random.randn(n_rays) * 0.005

    # Storage
    history_x = np.zeros((n_steps, n_rays))
    history_y = np.zeros((n_steps, n_rays))
    history_z = np.zeros((n_steps, n_rays))

    # 3. Simulate
    print("   -> Ray Tracing...")
    for t in range(n_steps):
        history_x[t] = pos_x
        history_y[t] = pos_y
        history_z[t] = pos_z
        
        ix = np.clip(pos_x.astype(int), 0, grid_size-1)
        iy = np.clip(pos_y.astype(int), 0, grid_size-1)
        iz = np.clip(pos_z.astype(int), 0, grid_size-1)
        
        gx = grad_x[ix, iy, iz]
        gy = grad_y[ix, iy, iz]
        gz = grad_z[ix, iy, iz]
        
        if USE_QUANTUM_SCAR:
            gy = gy + SCAR_STRENGTH * scar_gy[iy, iz]
            gz = gz + SCAR_STRENGTH * scar_gz[iy, iz]

        vel_x -= gx * 0.05
        vel_y -= gy * 0.05
        vel_z -= gz * 0.05

        if USE_CHIRAL_PHONON:
            fy, fz = lab.chiral_lattice_force(pos_y, pos_z, t, HANDEDNESS, PHONON_AMP,
                                              center=(center, center))
            vel_y += fy * 0.05
            vel_z += fz * 0.05
        
        speed = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
        vel_x = (vel_x / speed) * step_size
        vel_y = (vel_y / speed) * step_size
        vel_z = (vel_z / speed) * step_size
        
        pos_x += vel_x
        pos_y += vel_y
        pos_z += vel_z

    # 4. Export to 3D File
    export_to_obj("fractal_flow.obj", history_x, history_y, history_z)

    # 5. Quick Preview (Matplotlib)
    print("   -> Launching Preview...")
    fig = plt.figure(figsize=(10, 6), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    ax.set_axis_off()

    # Subsample for preview performance (show only 1000 rays)
    skip = n_rays // 1000
    if skip < 1: skip = 1
    
    hx, hy, hz = history_x[:, ::skip].T, history_y[:, ::skip].T, history_z[:, ::skip].T
    points = np.array([hx, hy, hz]).T.transpose(1, 0, 2)
    
    lc = Line3DCollection(points, colors=plt.cm.magma(np.linspace(0,1,hx.shape[0])), linewidths=0.5, alpha=0.1)
    ax.add_collection3d(lc)
    
    ax.set_xlim(0, n_steps * step_size)
    ax.set_ylim(0, grid_size)
    ax.set_zlim(0, grid_size)
    ax.view_init(elev=20, azim=-45)
    
    plt.title("Preview (Full resolution saved to disk)", color='white')
    plt.show()

def OperatingSystem(fractal_flow, sierpinski_triangle, koch_snowflake, dragon_curve, dna_particles, cellular_automata, l_system_tree, mandelbrot_set, julia_set, lorenz_attractor, barnsley_fern, peano_curve, hilbert_curve, sierpinski_carpet):
    print("   -> Launching Preview...")
    fig = plt.figure(figsize=(10, 6), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    ax.set_axis_off()
    # Subsample for preview performance (show only 1000 rays)
    n_rays = fractal_flow.shape[1]
    skip = n_rays // 1000
    if skip < 1: skip = 1
    hx, hy, hz = fractal_flow[:, ::skip].T, fractal_flow[:, ::skip].T, fractal_flow[:, ::skip].T
    points = np.array([hx, hy, hz]).T.transpose(1, 0, 2)
    lc = Line3DCollection(points, colors=plt.cm.magma(np.linspace(0,1,hx.shape[0])), linewidths=0.5, alpha=0.1)
    ax.add_collection3d(lc)
    ax.set_xlim(0, fractal_flow.shape[0] * 0.5)
    ax.set_ylim(0, 120) 
    ax.set_zlim(0, 120)
    ax.view_init(elev=20, azim=-45)
    plt.title("Preview (Full resolution saved to disk)", color='white')
    plt.show()

if __name__ == "__main__":
    text = input("Enter seed text for fractal generation: ") 
    run_simulation(text)