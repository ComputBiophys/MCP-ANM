function [] = filewrite(coor, rname, filename, numChains)
% FILEWRITE 将原子坐标写入PDB文件，并使用TER记录处理不连续性。

% 输入：
%   coor - 包含原子坐标（x, y, z）的Nx3矩阵。
%   rname - 三字母残基代码的字符串；其长度应为3*N。
%   filename - 输出PDB文件的名称。
%   numChains - 分配残基的链数。

    [n, ~] = size(coor); % 残基总数

    if mod(n, numChains) ~= 0
        error('残基总数（%d）必须能被链数（%d）整除。', n, numChains);
    end

    fd = fopen(filename, 'w'); % 打开文件用于写入
    residues_per_chain = n / numChains;
    chainLetters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'; % 支持最多26条链

    for chain = 1:numChains
        start_idx = (chain - 1) * residues_per_chain + 1;
        end_idx = chain * residues_per_chain;

        currentChainLetter = chainLetters(chain);

        for i = start_idx:end_idx
            % 检查不连续性；如果连续残基之间的距离很大
            if i > start_idx
                prev_coor = coor(i-1, :);
                current_coor = coor(i, :);
                distance = norm(current_coor - prev_coor);

                % 设定阈值距离以判断是否为不连续点
                if distance > 10 % 你可以根据需要调整此阈值
                    fprintf(fd, 'TER\n'); % 插入TER记录表示链的终止
                end
            end

            % 当前链中的残基编号
            chain_residue_number = i - (chain - 1) * residues_per_chain;

            fprintf(fd, 'ATOM  %5d  CA  %s %c%4d    %8.3f%8.3f%8.3f\n', ...
                i, rname(3*i-2:3*i), currentChainLetter, chain_residue_number, ...
                coor(i,1), coor(i,2), coor(i,3));
        end
        fprintf(fd, 'TER\n'); % 在每条链的末尾插入TER记录
    end

    fclose(fd);
end


