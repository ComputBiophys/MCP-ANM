%% ======= mcpANM Setup ========
tic
% Read the PDB file and extract CA coordinates, residue names, B-factors, and chain count
filename = '5vkq.pdb';
mcp_filename = '5vkq.mcp';
[posall, resall, bf, rname, chainCount] = pdbread(filename);
n = size(posall, 1);
chain_length = n / chainCount;  % Automatically computed chain length

% Define MCP-mANM parameters
f = 0.1;  % Membrane contact probability threshold
s = 2;  % Stiffness in the XY-plane
l = 64;  % Stiffness in the Z-direction
n1=6;   %decay parameter of kernal function
k1=3;   %decay parameter of kernal function
% Compute the Hessian matrix
mem_res = MCP_read(mcp_filename, f);  % Identify membrane-contacting residues
hessian=calculateHessian_kernal(posall, mem_res, s, l,n1,k1);
% Eigendecomposition of the Hessian matrix
[V, D] = eig(hessian);
U = V';
% Compute the inverse Hessian matrix (excluding the first six zero modes)
hessian_inverse = zeros(3 * n);
for k = 7:26
    hessian_inverse = hessian_inverse + V(:, k) * U(k, :) / D(k, k);
end

%% ======= Pulling/Pushing Forces on Specific Region ========
% Define the region (residue indices) in each chain where force will be applied.
% For example, if you want to pull/push residues 1 to 50 in each chain:
region_start = 1;
region_end   = 37;
if region_end > chain_length
    error('The specified region exceeds the length of each chain (%d residues).', chain_length);
end

% Create output folder (for pulling/pushing forces)
newfolder = strrep(filename, '.pdb', '_pull');
if ~exist(newfolder, 'dir')
    mkdir(newfolder);
end

% Loop over a range of force magnitudes (adjust as needed)
for force_mag = 0:0.001:0.02
    % Initialize the overall force vector for all residues (3 components per residue)
    F_all = zeros(3 * n, 1);
    up_force = force_mag;  % Define the magnitude of the force
    
    % Loop over each chain (assumes all chains have equal number of residues)
    for k = 0:(chainCount - 1)
        % Loop over the specified residue indices in the chain
        for j = region_start:region_end
            idx = j + k * chain_length;  % global residue index
            % Create a force vector that applies force only in the Z-direction.
            % For pushing/pulling, we assume a fixed direction, e.g., upward (0,0,up_force)
            F = zeros(3 * n, 1);
            F(3 * idx) = up_force;
            F_all = F_all + F;
        end
    end

    % Compute displacements using the inverse Hessian (ddR has 3*n components)
    ddR = hessian_inverse * F_all;
    
    % Apply displacements to the original positions
    posallo_new = zeros(n, 3);
    for i = 1:n
        posallo_new(i, 1) = posall(i, 1) + ddR(3*i-2, 1);
        posallo_new(i, 2) = posall(i, 2) + ddR(3*i-1, 1);
        posallo_new(i, 3) = posall(i, 3) + ddR(3*i, 1);
    end 

    % Generate output filename using the unified file writing function (filewrite.m)
    output_filename = [strrep(filename, '.pdb', '_'), num2str(up_force, '%0.3f'), '.pdb'];
    filewrite(posallo_new, rname, output_filename, chainCount);
    
    % Move the generated file to the output folder
    movefile(output_filename, newfolder);
end
toc