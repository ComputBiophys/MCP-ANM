import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import time


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

def plot_bfactors(bf_exp, bf_pred, f, n1, k1, s, l, PCC):
    """
    Plots the experimental and predicted B-factors.
    """
    plt.figure(figsize=(10, 6))

    # Plot experimental B-factor
    plt.plot(bf_exp, label='Experimental B-factor', color='blue', linewidth=2)

    # Plot predicted B-factor
    plt.plot(bf_pred, label='Predicted B-factor', color='red', linestyle='--', linewidth=2)

    # Add labels and legend
    plt.xlabel('Residue Index')
    plt.ylabel('B-factor')
    plt.title(f'Experimental vs Predicted B-factors\nf={f}, n1={n1}, k1={k1}, s={s}, l={l}, PCC={PCC:.4f}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Show plot
    plt.show()


if __name__ == '__main__':
    start_total = time.time()
    # User parameters
    filename = '5vkq.pdb'
    mcp_filename = '5vkq.mcp'
    f = 0.1
    n1 = 6
    k1 = 3
    s = 2
    l = 64

    posall, resall, bf_exp, rname, chainCount = read_pdb(filename)
    mem_res = read_mcp(mcp_filename, f)
    nres = posall.shape[0]

    H = calculate_hessian_kernal(posall, mem_res, s, l, n1, k1)

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(H)
    # Sort eigenvalues and vectors
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Calculate B-factors
    bf_pred = calc_bf(eigvals, eigvecs)

    # Pearson correlation
    R = np.corrcoef(bf_pred, bf_exp)
    PCC = R[0, 1]
    print(f"f={f}, n1={n1}, k1={k1}, s={s}, l={l}, PCC={PCC:.4f}")

    # Plot B-factors
    plot_bfactors(bf_exp, bf_pred, f, n1, k1, s, l, PCC)


    end_total = time.time()
    total_time = end_total - start_total
    print(f"cost time: {total_time:.2f} s")