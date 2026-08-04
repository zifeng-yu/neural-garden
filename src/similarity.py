"""
Neural Garden - 余弦相似度计算模块
Lesson 03: 向量搜索入门
"""

import math
from typing import List


def dot_product(a: List[float], b: List[float]) -> float:
    """
    计算向量点积
    
    Args:
        a: 向量 A
        b: 向量 B
        
    Returns:
        点积结果
    """
    if len(a) != len(b):
        raise ValueError(f"向量维度必须相同：{len(a)} != {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def vector_norm(a: List[float]) -> float:
    """
    计算向量模长
    
    Args:
        a: 向量
        
    Returns:
        模长
    """
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    计算余弦相似度
    
    公式：cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
    
    结果范围：
    - 1.0: 完全相同
    - 0.0: 完全不相关（正交）
    - -1.0: 完全相反
    
    Args:
        a: 向量 A
        b: 向量 B
        
    Returns:
        相似度 (-1 到 1)，越接近 1 越相似
    """
    dot = dot_product(a, b)
    norm_a = vector_norm(a)
    norm_b = vector_norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0  # 零向量，定义为不相似
    
    return dot / (norm_a * norm_b)


def cosine_distance(a: List[float], b: List[float]) -> float:
    """
    计算余弦距离
    
    余弦距离 = 1 - 余弦相似度
    
    Args:
        a: 向量 A
        b: 向量 B
        
    Returns:
        距离 (0 到 2)，越小越相似
    """
    return 1 - cosine_similarity(a, b)


if __name__ == "__main__":
    # 测试用例
    print("=== 余弦相似度测试 ===\n")
    
    # 测试 1：相同向量
    v1 = [0.5, 0.5, 0.5]
    v2 = [0.5, 0.5, 0.5]
    sim = cosine_similarity(v1, v2)
    print(f"相同向量：{sim:.4f} (期望 1.0)")
    assert abs(sim - 1.0) < 0.0001, "相同向量相似度应为 1.0"
    
    # 测试 2：正交向量
    v3 = [1, 0, 0]
    v4 = [0, 1, 0]
    sim = cosine_similarity(v3, v4)
    print(f"正交向量：{sim:.4f} (期望 0.0)")
    assert abs(sim - 0.0) < 0.0001, "正交向量相似度应为 0.0"
    
    # 测试 3：相反向量
    v5 = [1, 0, 0]
    v6 = [-1, 0, 0]
    sim = cosine_similarity(v5, v6)
    print(f"相反向量：{sim:.4f} (期望 -1.0)")
    assert abs(sim - (-1.0)) < 0.0001, "相反向量相似度应为 -1.0"
    
    # 测试 4：相似向量
    v7 = [0.8, 0.3, 0.2]
    v8 = [0.7, 0.4, 0.1]
    sim = cosine_similarity(v7, v8)
    dist = cosine_distance(v7, v8)
    print(f"\n相似向量：{sim:.4f}, 距离：{dist:.4f}")
    print(f"验证：sim + dist = {sim + dist:.4f} (期望 1.0)")
    
    print("\n✅ 所有测试通过")
