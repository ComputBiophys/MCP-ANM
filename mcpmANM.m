tic
%% ====================== Initialization ============================
% Define input PDB file and corresponding MCP file
filename = '5vkq.pdb';
mcp_filename =  '5vkq.mcp';

% Read PDB file and extract residue positions, names, and B-factors
[posall, reall, bf, rname] = pdbread(filename);
[num_residues, ~] = size(posall);  % Number of residues in the structure

%% ====================== MCP-mANM Hessian Calculation ============================
pmax = 0;  
% Define MCP-mANM parameters
f = 0.1;  % Membrane contact probability threshold
n1=6;   %decay parameter of kernal function
k1=3;   %decay parameter of kernal function
s = 2;  % Stiffness in the XY-plane
l = 64;  % Stiffness in the Z-direction

% Compute the Hessian matrix
mem_res = MCP_read(mcp_filename, f);  % Identify membrane-contacting residues
hessian=calculateHessian_kernal(posall, mem_res, s, l,n1,k1);

% Eigendecomposition of the Hessian matrix
[V, D] = eig(hessian);
U = V';
d=diag(D)';
flu = sum((V(:, 7:end) .* V(:, 7:end)) ./ d(7:end), 2);
bfactors=sum(reshape(flu,3,num_residues));
R=corrcoef(bfactors,bf);
PCC=R(2);
result=[f s l n1 k1 PCC]


% ====================== B-factor Linear Calibration & Plot ============================
% Linear regression calibration: bf ≈ k * bfactors + b
p = polyfit(bfactors, bf, 1);  % Fitting coefficients
k = p(1);
b0 = p(2);

% Calibrate predicted values
bfactors_calibrated = k * bfactors + b0;

% Plotting
figure;
plot(1:num_residues, bf, 'b-', 'LineWidth', 2); hold on;
plot(1:num_residues, bfactors_calibrated, 'r--', 'LineWidth', 2);
xlabel('Residue Index');
ylabel('B-factor');
legend('Experimental B-factor', 'Predicted B-factor (calibrated)');
title(['B-factor Comparison (PCC = ' num2str(PCC, '%.3f') ')']);
grid on;

toc