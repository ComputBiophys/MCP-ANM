function hessian = calculateHessian(posall, mem_res, cutoff, s, l)
% Function to compute the Hessian matrix for an anisotropic network model (ANM)  
% with membrane-aware modifications based on membrane contact probability (MCP).
%
% Input:
%   posall  - Nx3 matrix of atomic coordinates (Cα positions)
%   mem_res - Indices of membrane-contacting residues
%   cutoff  - Distance cutoff for network connectivity
%   s       - Stiffness parameter in the XY-plane
%   l       - Stiffness parameter in the Z-direction
%
% Output:
%   hessian - 3N x 3N Hessian matrix

    n = size(posall, 1);  % Number of residues
    hessian = zeros(3 * n, 3 * n);  % Initialize Hessian matrix

    % Create a logical vector to mark whether each residue is within the membrane
    is_mem_res = false(n, 1);
    is_mem_res(mem_res) = true;

    % Compute off-diagonal elements of the Hessian matrix
    for i = 1:n
        for j = i + 1:n
            % Assign stiffness parameters based on membrane association
            if is_mem_res(i) && is_mem_res(j)
                rx = s;
                ry = s;
                rz = l;
            elseif ~is_mem_res(i) && ~is_mem_res(j)
                rx = 1;
                ry = 1;
                rz = 1;
            else
                rx = sqrt(s);
                ry = sqrt(s);
                rz = sqrt(l);
            end

            % Compute Euclidean distance between residue pairs
            dis = sqrt(sum((posall(i, :) - posall(j, :)).^2));
            if dis <= cutoff
                % Compute Hessian matrix elements
                hessian(3*i-2, 3*j-2) = -rx * (posall(j, 1) - posall(i, 1))^2 / dis^2;
                hessian(3*i-2, 3*j-1) = -sqrt(rx * ry) * (posall(j, 1) - posall(i, 1)) * (posall(j, 2) - posall(i, 2)) / dis^2;
                hessian(3*i-2, 3*j)   = -sqrt(rx * rz) * (posall(j, 1) - posall(i, 1)) * (posall(j, 3) - posall(i, 3)) / dis^2;
                hessian(3*i-1, 3*j-2) = hessian(3*i-2, 3*j-1);
                hessian(3*i-1, 3*j-1) = -ry * (posall(j, 2) - posall(i, 2))^2 / dis^2;
                hessian(3*i-1, 3*j)   = -sqrt(ry * rz) * (posall(j, 2) - posall(i, 2)) * (posall(j, 3) - posall(i, 3)) / dis^2;
                hessian(3*i, 3*j-2)   = hessian(3*i-2, 3*j);
                hessian(3*i, 3*j-1)   = hessian(3*i-1, 3*j);
                hessian(3*i, 3*j)     = -rz * (posall(j, 3) - posall(i, 3))^2 / dis^2;
            end
            
            % Due to symmetry, set the corresponding transposed elements
            hessian(3*j-2, 3*i-2) = hessian(3*i-2, 3*j-2);
            hessian(3*j-1, 3*i-2) = hessian(3*i-2, 3*j-1);
            hessian(3*j, 3*i-2)   = hessian(3*i-2, 3*j);
            hessian(3*j-2, 3*i-1) = hessian(3*i-1, 3*j-2);
            hessian(3*j-1, 3*i-1) = hessian(3*i-1, 3*j-1);
            hessian(3*j, 3*i-1)   = hessian(3*i-1, 3*j);
            hessian(3*j-2, 3*i)   = hessian(3*i, 3*j-2);
            hessian(3*j-1, 3*i)   = hessian(3*i, 3*j-1);
            hessian(3*j, 3*i)     = hessian(3*i, 3*j);
        end
    end
    
    % Adjust diagonal elements to satisfy sum rule
    for i = 1:n
        for j = 1:n
            if j ~= i
                dis = sqrt(sum((posall(i, :) - posall(j, :)).^2));
                if dis <= cutoff
                    hessian(3*i-2, 3*i-2) = hessian(3*i-2, 3*i-2) - hessian(3*i-2, 3*j-2);
                    hessian(3*i-2, 3*i-1) = hessian(3*i-2, 3*i-1) - hessian(3*i-2, 3*j-1);
                    hessian(3*i-2, 3*i)   = hessian(3*i-2, 3*i)   - hessian(3*i-2, 3*j);
                    hessian(3*i-1, 3*i-2) = hessian(3*i-1, 3*i-2) - hessian(3*i-1, 3*j-2);
                    hessian(3*i-1, 3*i-1) = hessian(3*i-1, 3*i-1) - hessian(3*i-1, 3*j-1);
                    hessian(3*i-1, 3*i)   = hessian(3*i-1, 3*i)   - hessian(3*i-1, 3*j);
                    hessian(3*i, 3*i-2)   = hessian(3*i, 3*i-2)   - hessian(3*i, 3*j-2);
                    hessian(3*i, 3*i-1)   = hessian(3*i, 3*i-1)   - hessian(3*i, 3*j-1);
                    hessian(3*i, 3*i)     = hessian(3*i, 3*i)     - hessian(3*i, 3*j);
                end
            end
        end
    end
end

