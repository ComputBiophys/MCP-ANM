import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import time

start_time = time.time()

def read_pdb(filename):
    """
    Reads a PDB file and extracts C-alpha atom coordinates, B-factors, residue names, and chain IDs.
    Returns:
        posall: (n, 3) array of coordinates
        bf: (n,) array of B-factors
        rname: list of three-letter residue names
        resall: list of one-letter residue codes
        chainCount: number of unique chains
    """
    posall = []
    bf = []
    rname = []
    resall = []
    chain_ids = []

    three2one = {
        'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'LEU': 'L', 'ILE': 'I', 'PRO': 'P',
        'PHE': 'F', 'TRP': 'W', 'TYR': 'Y', 'SER': 'S', 'THR': 'T', 'CYS': 'C',
        'MET': 'M', 'ASN': 'N', 'GLN': 'Q', 'ASP': 'D', 'GLU': 'E', 'HIS': 'H',
        'LYS': 'K', 'ARG': 'R'
    }

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('END'):
                break
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                rn3 = line[17:20].strip()
                resname = three2one.get(rn3, '-')
                resall.append(resname)
                rname.append(rn3)

                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                posall.append([x, y, z])

                bfa = float(line[60:66])
                bf.append(bfa)

                chain_ids.append(line[21])

    posall = np.array(posall)
    bf = np.array(bf)
    unique_chains = set(chain_ids)
    chainCount = len(unique_chains)
    return posall, resall, bf, rname, chainCount

def read_mcp(filename, mcp_cut):
    """
    Reads MCP value files (supports .mcp and .csv formats) and returns 1-based indices of residues with MCP > threshold.
    
    Parameters:
        filename (str): Path to the input file (.mcp or .csv)
        mcp_cut (float): Threshold value for MCP filtering (residues with MCP > mcp_cut are selected)
    
    Returns:
        numpy.ndarray: 1-based indices of residues where MCP value exceeds the threshold
    
    Raises:
        FileNotFoundError: If the input file does not exist
        ValueError: If file has insufficient columns (CSV) or non-numeric data
        RuntimeError: If unsupported file format is provided
    """
    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    try:
        if ext == '.mcp':
            # Case 1: .mcp file (single column, no header - original logic)
            mcp_values = np.loadtxt(filename)
            # Ensure mcp_values is 1D array (handle edge case of single value)
            mcp_values = mcp_values.ravel()

        elif ext == '.csv':
            # Case 2: .csv file (read 3rd column, skip header row)
            # Use genfromtxt to skip header and read 3rd column (index=2 for 0-based)
            try:
                # Skip header row (skip_header=1), read only 3rd column (usecols=2)
                mcp_values = np.genfromtxt(
                    filename,
                    delimiter=',',
                    skip_header=1,
                    usecols=2,
                    dtype=float
                )
                # Check if CSV has at least 3 columns (genfromtxt returns nan if column missing)
                if np.isnan(mcp_values).all():
                    raise ValueError("CSV file has fewer than 3 columns")
            except Exception as e:
                raise ValueError(f"Failed to read CSV file: {str(e)}")

        else:
            # Case 3: Unsupported file format
            raise RuntimeError(
                f"Unsupported file format: {ext}\n"
                "Only .mcp (single column, no header) and .csv (3rd column with header) are supported"
            )

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filename}")
    except Exception as e:
        raise RuntimeError(f"Error reading file {filename}: {str(e)}")

    resid = np.arange(1, len(mcp_values) + 1)
    selected_residues = resid[mcp_values > mcp_cut]

    return selected_residues

def calculate_hessian_kernal(posall, mem_res, s, l, n1, k1):
    """
    Computes the Hessian matrix for membrane-aware anisotropic network model (MCP-MANM),
    using exponential distance decay and directional stiffness.

    Parameters:
        posall (np.ndarray): Nx3 array of Cα coordinates
        mem_res (list[int]): 0-based indices of membrane-contacting residues
        s (float): stiffness in membrane XY plane
        l (float): stiffness in membrane Z direction
        n1 (float): decay length scale
        k1 (float): decay power
    Returns:
        hessian (np.ndarray): 3N x 3N Hessian matrix
    """
    n = posall.shape[0]
    hessian = np.zeros((3 * n, 3 * n))
    is_mem = np.zeros(n, dtype=bool)
    is_mem[mem_res] = True

    for i in range(n):
        for j in range(i + 1, n):
            rij = posall[j] - posall[i]
            dist = np.linalg.norm(rij)
            if dist == 0:
                continue

            # Stiffness parameters
            if is_mem[i] and is_mem[j]:
                rx, ry, rz = s, s, l
            elif not is_mem[i] and not is_mem[j]:
                rx = ry = rz = 1.0
            else:
                rx = ry = np.sqrt(s)
                rz = np.sqrt(l)

            xij, yij, zij = rij
            dist2 = dist ** 2
            decay = np.exp(- (dist / n1) ** k1)

            # Force constant matrix elements with decay
            kxx = rx * decay * xij * xij / dist2
            kyy = ry * decay * yij * yij / dist2
            kzz = rz * decay * zij * zij / dist2
            kxy = np.sqrt(rx * ry) * decay * xij * yij / dist2
            kxz = np.sqrt(rx * rz) * decay * xij * zij / dist2
            kyz = np.sqrt(ry * rz) * decay * yij * zij / dist2

            # Submatrix (3x3) for H_ij
            Kij = np.array([[-kxx, -kxy, -kxz],
                            [-kxy, -kyy, -kyz],
                            [-kxz, -kyz, -kzz]])

            # Index slices
            idx_i = slice(3 * i, 3 * i + 3)
            idx_j = slice(3 * j, 3 * j + 3)

            # Off-diagonal blocks
            hessian[idx_i, idx_j] += Kij
            hessian[idx_j, idx_i] += Kij.T

            # Diagonal blocks
            hessian[idx_i, idx_i] -= Kij
            hessian[idx_j, idx_j] -= Kij.T

    return hessian

def calc_bf(eigvals, eigvecs):
    """
    Calculate B-factors from eigenvalues and eigenvectors.
    """
    eigvecs = eigvecs[:, 6:]  # Shape: (3N, M)
    eigvals = eigvals[6:]  # Shape: (M,)
    N = eigvecs.shape[0] // 3

    # Calculate B-factor for each residue
    bf = np.zeros(N)
    for i in range(N):
        for m in range(len(eigvals)):
            x = eigvecs[3 * i, m]
            y = eigvecs[3 * i + 1, m]
            z = eigvecs[3 * i + 2, m]
            bf[i] += (x**2 + y**2 + z**2) / eigvals[m]
    return bf

def filewrite(coor, rname, filename, numChains):
    """
    Write atomic coordinates to a PDB file, assigning residues based on the specified number of chains.
    """
    n = coor.shape[0]
    if n % numChains != 0:
        raise ValueError(f"Total number of residues ({n}) must be divisible by the number of chains ({numChains}).")

    residues_per_chain = n // numChains
    chain_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    with open(filename, 'w') as f:
        for chain in range(numChains):
            start_idx = chain * residues_per_chain
            end_idx = (chain + 1) * residues_per_chain
            for i in range(start_idx, end_idx):
                chain_residue_number = i - start_idx + 1
                f.write(f"ATOM  {i + 1:5d}  CA  {rname[i]:>3s} {chain_letters[chain]}{chain_residue_number:4d}    "
                        f"{coor[i, 0]:8.3f}{coor[i, 1]:8.3f}{coor[i, 2]:8.3f}\n")

def calculate_inverse_hessian(eigvals, eigvecs):
    """
    Calculate the inverse of the Hessian matrix excluding the first six zero modes.

    Parameters:
        eigvals (np.ndarray): Array of eigenvalues.
        eigvecs (np.ndarray): Matrix of eigenvectors.

    Returns:
        np.ndarray: The inverse Hessian matrix.
    """
    hessian_inverse = np.zeros_like(eigvecs @ eigvecs.T)
    for k in range(6, 26):
        vk = eigvecs[:, k].reshape(-1, 1)
        hessian_inverse += (vk @ vk.T) / eigvals[k]
    return hessian_inverse

def apply_forces(posall, normals, mem_res, push_res, membrane_tension_value):
    """
    Apply forces to membrane-contacting residues.

    Parameters:
        posall (np.ndarray): Array of coordinates.
        normals (np.ndarray): Array of normal vectors.
        mem_res (list): List of membrane-contacting residue indices.
        push_res (list): List of residue indices where force is applied.
        membrane_tension_value (float): The membrane tension value.

    Returns:
        np.ndarray: The force vector.
    """
    num_residues = posall.shape[0]
    is_mem_res = np.zeros(num_residues, dtype=bool)
    is_mem_res[mem_res] = True
    F_all = np.zeros(3 * num_residues)

    for i in range(num_residues):
        if is_mem_res[i]:
            normal = normals[i]
            norm_length = np.linalg.norm(normal)
            if norm_length == 0:
                continue
            unit_vector = normal / norm_length
            F_all[3 * i:3 * i + 3] += membrane_tension_value * unit_vector

    return F_all

def save_deformed_structure(posall, ddR, rname, output_filename, chainCount):
    """
    Save the deformed structure to a PDB file.

    Parameters:
        posall (np.ndarray): Original coordinates.
        ddR (np.ndarray): Displacement vector.
        rname (list): List of residue names.
        output_filename (str): Output file name.
        chainCount (int): Number of chains.
    """
    posallo = posall + ddR.reshape(-1, 3)
    filewrite(posallo, rname, output_filename, chainCount)

def main():
    start_time = time.time()

    # User parameters
    filename = '5z10.pdb'
    mcp_filename = '5z10.mcp'
    norm_filename='5z10_normals.txt'
    f = 0.6
    n1 = 6
    k1 = 2
    s = 64
    l = 64

    posall, resall, bf_exp, rname, chainCount = read_pdb(filename)
    mem_res = read_mcp(mcp_filename, f)
    num_residues = posall.shape[0]

    H = calculate_hessian_kernal(posall, mem_res, s, l, n1, k1)

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(H)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    bf_pred = calc_bf(eigvals, eigvecs)
    R = np.corrcoef(bf_pred, bf_exp)
    PCC = R[0, 1]

    print(f"f={f}, n1={n1}, k1={k1}, s={s}, l={l}, PCC={PCC:.4f}")

    hessian_inverse = calculate_inverse_hessian(eigvals, eigvecs)
    normals = np.loadtxt(norm_filename) #  load the membrane tension direction vectors from a file.
    normals[:, 2] = 0.0

    p_threshold = 0.9
    push_res = read_mcp(mcp_filename, p_threshold) - 1
    center = np.mean(posall[:, :2], axis=0)
    posall[:, :2] -= center

    transmembrane_pos = posall[push_res, :2]
    radius_transmembrane = np.max(np.linalg.norm(transmembrane_pos, axis=1))

    tension_range = np.arange(0.0, 1.6, 0.1)
    output_folder = filename.replace('.pdb', '_structure')
    os.makedirs(output_folder, exist_ok=True)

    for membrane_tension_value in tension_range:
        F_all = apply_forces(posall, normals, mem_res, push_res, membrane_tension_value)
        ddR = hessian_inverse @ F_all

        output_filename = os.path.join(
            output_folder,
            f"{filename.replace('.pdb', '')}_{membrane_tension_value:.2f}.pdb"
        )

        save_deformed_structure(posall, ddR, rname, output_filename, chainCount)

    print(f"Processing completed. Output files saved to: {output_folder}")
    print(f"Program execution time: {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    main()

