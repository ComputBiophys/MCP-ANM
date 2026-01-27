function [A] = MCP_read(filename, mcp_cut)
% MCP_read Read MCP value files, supporting .mcp (single column, no header) and .csv (3rd column with header) formats
% Input:
%   filename - File path (supports .mcp/.csv extensions)
%   mcp_cut  - Threshold for MCP values, returns indices of residues with values greater than this threshold
% Output:
%   A        - Indices of residues where mcp > mcp_cut (1-based indexing)

    % ========== Step 1: Extract and standardize file extension ==========
    [~, ~, ext] = fileparts(filename);  % Split filename to get extension (includes '.')
    ext = lower(ext);  % Convert to lowercase to handle mixed cases like .CSV/.MCP
    
    % ========== Step 2: Read data by format ==========
    mcp = [];  % Initialize MCP value array
    if strcmp(ext, '.mcp')
        % Original logic for .mcp files: single column, no header
        try
            data1 = importdata(filename);
            mcp = data1(:, 1);  % Extract first column
        catch ME
            error('Failed to read .mcp file: %s\nError details: %s', filename, ME.message);
        end
        
    elseif strcmp(ext, '.csv')
        % Logic for CSV files: read 3rd column and skip header row
        try
            % Method 1: Prefer readtable (MATLAB R2013b+) for better header compatibility
            tbl = readtable(filename, 'VariableNamingRule', 'preserve');  % Preserve original column names
            
            % Check if CSV has at least 3 columns
            col_num = width(tbl);
            if col_num < 3
                error('Insufficient columns in CSV file! File %s has only %d columns (requires at least 3)', filename, col_num);
            end
            
            % Extract 3rd column values (automatically skip header, convert to numeric array)
            mcp_col = tbl{:, 3};  % Curly braces extract numeric values (excludes header)
            % Handle non-numeric data (e.g., numeric values stored as strings)
            if iscell(mcp_col)
                mcp = cell2mat(cellfun(@str2double, mcp_col, 'UniformOutput', false));
            else
                mcp = double(mcp_col);
            end
            
        catch ME
            % Fallback for older MATLAB versions (no readtable) using textscan
            try
                fid = fopen(filename, 'r');
                if fid == -1
                    error('Cannot open CSV file: %s', filename);
                end
                
                % Skip header row
                fgetl(fid);
                
                % Read all rows, delimited by commas, extract only 3rd column values
                % Format spec: %[^,] read all chars before comma (ignore first two columns), %f read 3rd column numeric, %[^\n] ignore remaining columns
                data = textscan(fid, '%[^,],%[^,],%f%[^\n]', 'Delimiter', ',', 'EmptyValue', NaN);
                fclose(fid);
                
                mcp = data{3};  % 3rd column is numeric array
            catch ME2
                error('Failed to read CSV file: %s\nError details: %s', filename, ME2.message);
            end
        end
        
    else
        % Unsupported file format
        error('Unsupported file format: %s! Only .mcp (single column, no header) and .csv (3rd column with header) are supported', ext);
    end
    
    % ========== Step 3: Validate data validity ==========
    if isempty(mcp)
        error('MCP data read from file %s is empty. Please check file content', filename);
    end
    if ~isnumeric(mcp)
        error('MCP data from file %s is non-numeric. Please check the 3rd column (CSV) or column data (MCP)', filename);
    end
    
    % ========== Step 4: Filter indices of residues above threshold ==========
    A = find(mcp > mcp_cut);

end