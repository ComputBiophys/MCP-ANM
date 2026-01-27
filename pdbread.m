function [posall, resall, bf, rname, chainCount] = pdbread(filename)
% PDBREAD Reads a large PDB file and extracts atomic positions, residue names, 
% B-factors, and chain information from CA atoms.
%
% Input:
%   filename - Name of the PDB file to be read.
%
% Output:
%   posall     - Matrix containing the 3D coordinates of C¦Á atoms.
%   resall     - Sequence of residue names (single-letter codes).
%   bf         - B-factors of the extracted residues.
%   rname      - Residue names in three-letter codes.
%   chainCount - Number of unique chains detected in the file.

% Close all open files
fopen('all');

% Define maximum number of lines to read
max = 50000;

% Initialize output variables
posall = [];
resall = [];
rname = [];
bf = [];
chainIDs = {};  % To store chain identifiers

% Open PDB file in read mode
ft = fopen(filename, 'r'); % Open the PDB file in read mode

% Loop through the PDB file line by line
for i = 1:max
    pl = fgetl(ft);  % Read the next line from the file
    
    % Stop reading when encountering "END"
    if strcmp(pl(1:3), 'END')
        break;
    
    % Extract C¦Á atom information
    elseif ((strcmp(pl(1:4), 'ATOM')) && (pl(14) == 'C') && (pl(15) == 'A'))
        rn = pl(18:20);  % Extract three-letter residue name
        
        % Convert three-letter residue name to single-letter code
        if strcmp(rn, 'GLY')
            resname = 'G';
        elseif strcmp(rn, 'ALA')
            resname = 'A';
        elseif strcmp(rn, 'VAL')
            resname = 'V';
        elseif strcmp(rn, 'LEU')
            resname = 'L';
        elseif strcmp(rn, 'ILE')
            resname = 'I';
        elseif strcmp(rn, 'PRO')
            resname = 'P';
        elseif strcmp(rn, 'PHE')
            resname = 'F';
        elseif strcmp(rn, 'TRP')
            resname = 'W';
        elseif strcmp(rn, 'TYR')
            resname = 'Y';
        elseif strcmp(rn, 'SER')
            resname = 'S';
        elseif strcmp(rn, 'THR')
            resname = 'T';
        elseif strcmp(rn, 'CYS')
            resname = 'C';
        elseif strcmp(rn, 'MET')
            resname = 'M';
        elseif strcmp(rn, 'ASN')
            resname = 'N';
        elseif strcmp(rn, 'GLN')
            resname = 'Q';
        elseif strcmp(rn, 'ASP')
            resname = 'D';
        elseif strcmp(rn, 'GLU')
            resname = 'E';
        elseif strcmp(rn, 'HIS')
            resname = 'H';
        elseif strcmp(rn, 'LYS')
            resname = 'K';
        elseif strcmp(rn, 'ARG')
            resname = 'R';
        else
            resname = '-';
        end
        
        % Store the single-letter residue name
        resall = [resall resname];
        
        % Extract atomic coordinates (x: columns 30-38, y: 39-46, z: 47-54)
        pos1 = str2num(pl(30:38));
        pos2 = str2num(pl(39:46));
        pos3 = str2num(pl(47:54));
        pos = [pos1 pos2 pos3];
        
        % Store atomic positions
        posall = [posall; pos];
        
        % Extract B-factor (columns 61-66)
        bfa = str2num(pl(61:66));
        bf = [bf; bfa];
        
        % Store three-letter residue name
        rname = [rname rn];
        
        % Extract chain identifier (column 22) and store it
        chainIDs{end+1} = pl(22);
    end
end

% Close all open files
fclose('all');

% Determine the number of unique chains
uniqueChains = unique(chainIDs);
chainCount = length(uniqueChains);

