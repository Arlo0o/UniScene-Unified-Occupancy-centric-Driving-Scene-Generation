import torch

def chunked_sum(A, chunk_size):
    N, M = A.shape
    result = torch.zeros(M, device=A.device)
    
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        result += A[i:end].sum(0)
    
    return result

def chunked_nonzero(A, chunk_size):
    N, M = A.shape
    non_zero_indices = []

    # 按行进行分块
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        chunk = A[i:end]
        # 获取这个块中的非零元素的索引
        non_zero_chunk = torch.nonzero(chunk, as_tuple=False)

        # 将行索引恢复为全局索引
        non_zero_chunk[:, 0] += i  # 将块的局部行号转换为全局行号

        non_zero_indices.append(non_zero_chunk)

    # 拼接结果
    return torch.cat(non_zero_indices, dim=0)

def chunked_process(A, B, chunk_size, calc_sum=False):
    N = A.shape[0]
    M = B.shape[0]
    sum_result = torch.zeros(M, device=A.device)
    nonzero_result = []
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        chunk = A[i:end]
        temp = (chunk.unsqueeze(1) == B.unsqueeze(0))
        if calc_sum:
            sum_result += temp.sum(0)
        non_zero_chunk = torch.nonzero(temp, as_tuple=False)
        non_zero_chunk[:, 0] += i
        nonzero_result.append(non_zero_chunk)
    return sum_result, torch.cat(nonzero_result, dim=0)

def compute_pos_in_voxel(A):
    N = A.size(0)
    # 对A进行排序
    sorted_A, sorted_indices = torch.sort(A, stable=True)
    # 生成位置索引
    positions = torch.arange(N, device=A.device)
    # 找到值变化的位置
    value_change = (sorted_A[1:] != sorted_A[:-1]).nonzero(as_tuple=False).squeeze()
    # 确定每个组的起始和结束索引
    group_boundaries = torch.cat([torch.tensor([0], device=A.device), value_change + 1, torch.tensor([N], device=A.device)])
    # 计算每个组的大小
    group_sizes = group_boundaries[1:] - group_boundaries[:-1]
    # 获取每个组的起始位置
    group_starts = group_boundaries[:-1]
    # 为每个组生成相应的计数（组内索引）
    occurrence_counts = positions - torch.repeat_interleave(group_starts, group_sizes)
    # 创建结果张量B
    B = torch.zeros(N, device=A.device, dtype=torch.long)
    # 将结果映射回原始顺序
    B[sorted_indices] = occurrence_counts
    return B
