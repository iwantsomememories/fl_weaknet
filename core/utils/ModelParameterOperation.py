import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_scarcity_reward(global_params, client_params_list, method = "mean"):
    """
    计算平均行稀疏度
    
    Args:
        global_params (np.ndarray): 全局模型参数 (d,)
        client_params_list (List[np.ndarray]): 客户端参数列表 [ (d,), (d,), ... ]
    
    Returns:
        float: 平均行稀疏度(范围[0,1])
    """
    # 1. 计算每个客户端的参数更新向量
    updates = []
    for client_params in client_params_list:
        delta = client_params - global_params
        updates.append(delta.numpy().reshape(1, -1)) 
    
    # 2. 计算余弦相似度矩阵
    sim_matrix = cosine_similarity(np.vstack(updates)) # (N, N), S[i,j] ∈ [-1,1]
    
    # 3. 计算不相似性矩阵（稀疏性矩阵）
    W = 1 - sim_matrix  # W[i,j] ∈ [0,2]
    
    # 4. 计算每行的平均稀疏度
    row_sparsity = np.mean(W, axis=1)  # (N,)
    
    if method == "mean":
        reward = np.mean(row_sparsity)
    elif method == "max":
        reward = np.max(row_sparsity)
    else:
        raise ValueError("Invalid method. Choose 'mean' or 'max'.")
    
    return reward


# 示例使用
if __name__ == "__main__":
    from fedlab.models import CNN_CIFAR10, CNN_FEMNIST, CNN_MNIST
    from fedlab.utils import SerializationTool

    global_model = CNN_MNIST()
    global_params = SerializationTool.serialize_model(global_model)

    client_params = [
        global_params + np.random.randn(global_params.shape[0]) * 0.1,  # 与全局相似
        global_params + np.random.randn(global_params.shape[0]) * 0.1,  # 与全局相似
        global_params + np.random.randn(global_params.shape[0]) * 0.2,  # 中等差异
        global_params + np.random.randn(global_params.shape[0]) * 10.0,  # 高差异（稀缺数据）
        global_params + np.random.randn(global_params.shape[0]) * 0.3   # 中等差异
    ]
    
    # 计算稀缺性权重
    weights = compute_scarcity_reward(global_params, client_params, method="max")
    print("客户端稀缺性:", weights)