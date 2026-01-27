tic
%% ====================== Initialization ============================
% Define input PDB file and corresponding MCP file
filename = '5z10.pdb';
mcp_filename =  '5z10.mcp';

% Read PDB file and extract residue positions, names, and B-factors
[posall, reall, bf, rname, chainCount] = pdbread(filename);
[num_residues, ~] = size(posall);  % Number of residues in the structure

%% ====================== MCP-mANM Hessian Calculation ============================
pmax = 0;  
% Define MCP-mANM parameters
f = 0.6;  % Membrane contact probability threshold
n1=6;   %decay parameter of kernal function
k1=2;   %decay parameter of kernal function
s = 64;  % Stiffness in the XY-plane
l = 64;  % Stiffness in the Z-direction

%% 
% Compute the Hessian matrix
mem_res = MCP_read(mcp_filename, f);  % Identify membrane-contacting residues
hessian=calculateHessian_kernal(posall, mem_res, s, l,n1,k1);

% Eigendecomposition of the Hessian matrix
[V, D] = eig(hessian);
U = V';

% Compute the inverse Hessian matrix (excluding the first six zero modes)
hessian_inverse = zeros(3 * num_residues);
for k = 7:26
    hessian_inverse = hessian_inverse + V(:, k) * U(k, :) / D(k, k);
end

%% ====================== Load Normal Vectors ============================
% Load precomputed normal vectors for each residue
norm_filename = strrep(filename, '.pdb', '_normals.txt');
normals = load(norm_filename);

% Create a logical vector to identify membrane residues
is_mem_res = false(num_residues, 1);
is_mem_res(mem_res) = true;

% Set the Z-component of normal vectors to zero (considering only in-plane forces)
normals(:, 3) = 0;

%% ====================== Apply PRS Forces on Membrane Residues ============================
p = 0.9;  % MCP cutoff for force application
push_res = MCP_read(mcp_filename, p);
N_push = length(push_res);  % Number of residues receiving force

% Compute geometric center of the structure in the XY-plane
center = mean(posall(:, 1:2), 1);
posall(:, 1:2) = posall(:, 1:2) - center;  % Shift structure to the new origin

% Compute the radius of the transmembrane region
transmembrane_pos = posall(push_res, 1:2);
radius_transmembrane = max(sqrt(sum(transmembrane_pos.^2, 2)));

%% ====================== Apply Membrane Tension and Compute Displacement ============================
% Define membrane tension range
tension_range = 0:0.1:1.5;

% Create output directory
output_folder = strrep(filename, '.pdb', '_structure');
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

for membrane_tension_value = tension_range
    % Compute force applied to the residue
    residue_force = membrane_tension_value;  % Force per residue

    % Initialize force vector
    F_all = zeros(3 * num_residues, 1);

    % Apply forces to membrane residues
    for i = 1:num_residues
        if is_mem_res(i)
            % Read and normalize the normal vector
            normal = normals(i, :);
            unit_vector = normal / norm(normal);

            % Apply force vector in the direction of the normal vector
            F = zeros(3 * num_residues, 1);
            F(3 * i - 2) = residue_force * unit_vector(1); % x-component
            F(3 * i - 1) = residue_force * unit_vector(2); % y-component
            F(3 * i) = residue_force * unit_vector(3); % z-component
            F_all = F_all + F;
        end
    end

    % Compute residue displacements using inverse Hessian
    ddR = hessian_inverse * F_all;

    % Apply displacements to the original positions
    posallo = zeros(num_residues, 3);
    for i = 1:num_residues
        posallo(i, 1) = posall(i, 1) + ddR(3*i-2, 1);
        posallo(i, 2) = posall(i, 2) + ddR(3*i-1, 1);
        posallo(i, 3) = posall(i, 3) + ddR(3*i, 1);
    end 
    % Compute displacement magnitude for each residue
    %deltaR = sqrt(sum(reshape(ddR, [num_residues, 3]).^2, 2));

    % Generate output filename
    output_filename = fullfile(output_folder, ...
    sprintf('%s_%.2f.pdb', strrep(filename, '.pdb', ''), membrane_tension_value));

    % Save displaced structure
  
    output_filename = fullfile(output_folder, sprintf('%s_%.2f.pdb', strrep(filename, '.pdb', ''), membrane_tension_value));
    filewrite(posallo, rname, output_filename, chainCount);

end
% Display completion message
fprintf('Process completed successfully. Output files are saved in: %s\n', output_folder);
toc
