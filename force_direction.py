import numpy as np
from scipy.spatial import KDTree
import os

def read_pdb(filename):
    """Read a PDB file and return a list of Cα atom coordinates and residue identifiers."""
    coords = []
    residues = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('ATOM') and line[13:15].strip() == 'CA':
                res_id = line[22:26].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                coords.append([x, y, z])
                residues.append(res_id)
    return np.array(coords), residues


def calculate_normal_vector(center, neighbors):
    """Calculate a normal vector based on the center atom and its neighbors."""
    if len(neighbors) < 3:
        return np.zeros(3)  # Not enough neighbors to calculate a normal

    # Use first three neighbors to calculate the normal vector
    p1, p2, p3 = neighbors[:3]
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    # Set Z component to 0

    norm = np.linalg.norm(normal)
    if norm == 0:
        return np.zeros(3)
    normal /= norm
    normal[2] = 0
    return normal


def ensure_outward_normal(center, normal, local_center, global_center=None, use_global=False):
    """Ensure the normal vector is pointing outward based on a local or global center."""
    to_center_vec = center - local_center
    if use_global:
        # Use global center to adjust normal direction
        to_global_center_vec = center - global_center
        if np.dot(normal, to_global_center_vec) < 0:
            normal = -normal
    #else:
        # Use local center to adjust normal direction
        if np.dot(normal, to_center_vec) < 0:
            normal = -normal
    return normal


def compute_surface_normals(coords, residues, global_center=None, use_global=False, max_neighbors=6):
    """Compute the normal vectors for each residue."""
    tree = KDTree(coords)
    normals = []
    for i, coord in enumerate(coords):
        # Start with a small radius and increase until we get enough neighbors
        radius = 5.0
        neighbors = []
        while len(neighbors) < 3 and radius <= 15.0:
            distances, indices = tree.query(coord, k=max_neighbors, distance_upper_bound=radius)
            valid_indices = indices[distances < float('inf')]
            # Remove the center atom itself from the neighbors
            valid_indices = valid_indices[valid_indices != i]
            neighbors = coords[valid_indices]
            radius += 1.0  # Increase radius if not enough neighbors

        if len(neighbors) >= 3:
            normal = calculate_normal_vector(coord, neighbors)
            local_center = np.mean(neighbors, axis=0)  # Calculate local center
            normal = ensure_outward_normal(coord, normal, local_center, global_center, use_global)

            if use_global:
                to_global_center_vec = coord - global_center
                to_global_center_vec[2]=0 # only X-Y direction 
                to_global_center_vec /= np.linalg.norm(to_global_center_vec)

                # Weighted Local Normal Vector and Global Normal Vector ,default 0.6
                a = 0.6  # Weight of the local normal vector (perpendicular to the residue surface)
                b = 1 - a  # Weight of the global normal vector (radially outward from the center of the residue plane)
                adjusted_normal = a* normal + b* to_global_center_vec
                adjusted_normal /= np.linalg.norm(adjusted_normal)
                normals.append(adjusted_normal)
            else:
                normals.append(normal)
        else:
            normals.append(np.zeros(3))

    return normals


def save_normals_to_file(residues, normals, filename):
    """Save the residue identifiers and their normal vectors to a file."""
    with open(filename, 'w') as file:
        for res_id, normal in zip(residues, normals):
            file.write(f'{normal[0]:.3f} {normal[1]:.3f} {normal[2]:.3f}\n')


if __name__ == "__main__":
    pdb_files = [f for f in os.listdir('.') if f.endswith('.pdb')]
    for pdb_file in pdb_files:
        coords, residues = read_pdb(pdb_file)
        global_center = np.mean(coords, axis=0) 
        normals = compute_surface_normals(coords, residues, global_center, use_global=True)
        output_filename = pdb_file.replace('.pdb', '_normals.txt')
        save_normals_to_file(residues, normals, output_filename)
        print(f"Processed {pdb_file} and saved normals to {output_filename}")

