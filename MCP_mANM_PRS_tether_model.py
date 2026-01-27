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


def filewrite(coor, rname, filename, numChains):
    """
    Write atomic coordinates to a PDB file, assigning residues to specified chains with TER records
    inserted for discontinuities (distance > 10 Å between consecutive residues) and at the end of each chain.
    Fully aligned with the original MATLAB implementation.

    Parameters:
        coor (np.ndarray): Nx3 array of atomic coordinates (x, y, z) (0-based indexing)
        rname (list/str):
            - List: Length-N list of 3-letter residue names (e.g., ['ALA', 'GLY', ...])
            - String: 3*N length continuous string (compatible with MATLAB input format, e.g., 'ALAGLY...')
        filename (str): Path/name of the output PDB file
        numChains (int): Number of chains to assign residues to

    Raises:
        ValueError: If total number of residues is not divisible by number of chains, or chain count exceeds 26
        TypeError: If rname is neither list nor string format
    """
    n = coor.shape[0]
    # Check if total residues are divisible by number of chains (consistent with MATLAB)
    if n % numChains != 0:
        raise ValueError(f"Total number of residues ({n}) must be divisible by number of chains ({numChains}).")

    residues_per_chain = n // numChains
    chain_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    # Check maximum chain limit (26 letters total)
    if numChains > len(chain_letters):
        raise ValueError(f"Number of chains ({numChains}) exceeds maximum supported (26).")

    with open(filename, 'w') as f:
        # Iterate over each chain (0-based in Python → mapped from 1-based in MATLAB)
        for chain in range(numChains):
            start_idx = chain * residues_per_chain  # Python 0-based start index
            end_idx = (chain + 1) * residues_per_chain  # Python 0-based end index (exclusive)
            current_chain_letter = chain_letters[chain]

            # Iterate over residues in current chain
            for i in range(start_idx, end_idx):
                # 1. Check for discontinuity (insert TER if distance > 10Å, matching MATLAB)
                if i > start_idx:  # Not the first residue in the chain
                    prev_coor = coor[i - 1]
                    current_coor = coor[i]
                    distance = np.linalg.norm(current_coor - prev_coor)
                    if distance > 10.0:  # Threshold matches MATLAB default
                        f.write('TER\n')  # Simplified TER line (MATLAB-compatible)

                # 2. Calculate residue number within current chain (aligned with MATLAB)
                chain_residue_number = (i - start_idx) + 1
                # Handle rname input format (compatible with list/string)
                if isinstance(rname, list):
                    res_name = rname[i]  # List format: direct 3-letter residue name
                elif isinstance(rname, str):
                    # String format (MATLAB-compatible): extract 3-character substring
                    res_name = rname[3*i : 3*i+3]
                else:
                    raise TypeError("rname must be a list (3-letter residue names) or string (3*N length)")

                # 3. ATOM line format strictly aligned with MATLAB's fprintf output
                atom_serial = i + 1  # Atom serial number (1-based, same as MATLAB)
                atom_line = (
                    f"ATOM  {atom_serial:5d}  CA  {res_name} {current_chain_letter}{chain_residue_number:4d}    "
                    f"{coor[i, 0]:8.3f}{coor[i, 1]:8.3f}{coor[i, 2]:8.3f}\n"
                )
                f.write(atom_line)

            # 4. Insert TER line at the end of each chain (consistent with MATLAB)
            f.write('TER\n')


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

def main(filename, mcp_filename, f, s, l, n1, k1, region_start, region_end):
    # Read the PDB file and extract CA coordinates, residue names, B-factors, and chain count
    posall, resall, bf_exp, rname, chainCount = read_pdb(filename)
    n = posall.shape[0]
    chain_length = n // chainCount

    mem_res = read_mcp(mcp_filename, f)  # Identify membrane-contacting residues

    # Compute the Hessian matrix
    hessian = calculate_hessian_kernal(posall, mem_res, s, l, n1, k1)

    # Eigendecomposition of the Hessian matrix
    eigvals, eigvecs = np.linalg.eigh(hessian)
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
    # Compute the inverse Hessian matrix (excluding the first six zero modes)
    hessian_inverse = calculate_inverse_hessian(eigvals, eigvecs)

    if region_end > chain_length:
        raise ValueError(f"The specified region exceeds the length of each chain ({chain_length} residues).")

    # Create output folder (for pulling/pushing forces)
    newfolder = filename.replace('.pdb', '_pull')
    if not os.path.exists(newfolder):
        os.makedirs(newfolder)

    # Loop over a range of force magnitudes (adjust as needed)
    for force_mag in np.arange(0, 0.021, 0.001):
        # Initialize the overall force vector for all residues (3 components per residue)
        F_all = np.zeros((3 * n, 1))
        up_force = force_mag  # Define the magnitude of the force

        # Loop over each chain (assumes all chains have equal number of residues)
        for k in range(chainCount):
            # Loop over the specified residue indices in the chain (j from 1 to region_end)
            for j in range(1, region_end + 1):  
                i = (j - 1) + k * chain_length  # Global residue index (0-based)
                # Apply force in the Z-direction
                F_all[3 * i + 2] = up_force  # Z-component for residue i

        # Compute displacements using the inverse Hessian (ddR has 3*n components)
        ddR = hessian_inverse @ F_all

        # Apply displacements to the original positions
        posallo_new = np.zeros((n, 3))
        for i in range(n):
            posallo_new[i, 0] = posall[i, 0] + ddR[3 * i, 0]      # X
            posallo_new[i, 1] = posall[i, 1] + ddR[3 * i + 1, 0]  # Y
            posallo_new[i, 2] = posall[i, 2] + ddR[3 * i + 2, 0]  # Z

        # Generate output filename using the unified file writing function
        output_filename = filename.replace('.pdb', f'_{up_force:.3f}.pdb')
        filewrite(posallo_new, rname, output_filename, chainCount)

        # Move the generated file to the output folder
        os.rename(output_filename, os.path.join(newfolder, os.path.basename(output_filename)))

if __name__ == '__main__':
    start_total = time.time()
    # Define parameters
    filename = '5vkq.pdb'
    mcp_filename = '5vkq.mcp'
    f = 0.1  # Membrane contact probability threshold
    s = 2  # Stiffness in the XY-plane
    l = 64  # Stiffness in the Z-direction
    n1 = 6  # Decay parameter of kernel function
    k1 = 3  # Decay parameter of kernel function
    region_start = 1  #The region (region_start-region_end) you need to push
    region_end = 37   ##

    main(filename,mcp_filename, f, s, l, n1, k1, region_start, region_end)

    end_total = time.time()
    total_time = end_total - start_total
    print(f"cost time: {total_time:.2f} s")
