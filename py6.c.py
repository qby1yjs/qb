import sys
import os
import time
import threading
import numpy as np
import pandas as pd

def load_and_preprocess(csv_path: str) -> np.ndarray:

    df = pd.read_csv(csv_path, encoding='latin1')
    print(f"[数据] 原始数据量: {len(df)} 行")

    df = df[df['Quantity'] > 0].copy()
    df = df[df['UnitPrice'] > 0].copy()
    print(f"[数据] 清洗后数据量: {len(df)} 行 (去除负值/异常后)")

    df['CustomerID'] = df['CustomerID'].fillna(0).astype(int)

    numeric_cols = ['Quantity', 'UnitPrice', 'CustomerID']
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    numeric_cols.append('TotalAmount')

    X = df[numeric_cols].values.astype(np.float64)

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0  # 避免除零
    X = (X - mean) / std

    print(f"[数据] 特征矩阵形状: {X.shape}")
    print(f"[数据] 特征列: {numeric_cols}")
    return X

def init_centroids(X: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_samples = X.shape[0]
    indices = rng.choice(n_samples, size=k, replace=False)
    return X[indices].copy()

def kmeans_assign_single(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:

    X_sq = np.sum(X ** 2, axis=1, keepdims=True)          # (n, 1)
    C_sq = np.sum(centroids ** 2, axis=1, keepdims=True)   # (k, 1)
    dists = X_sq + C_sq.T - 2.0 * X.dot(centroids.T)      # (n, k)
    dists = np.maximum(dists, 0)  # 数值稳定性

    labels = np.argmin(dists, axis=1)
    return labels

def _assign_chunk(
    X_chunk: np.ndarray,
    centroids: np.ndarray,
    start_idx: int,
    labels: np.ndarray,
):

    X_sq = np.sum(X_chunk ** 2, axis=1, keepdims=True)
    C_sq = np.sum(centroids ** 2, axis=1, keepdims=True)
    dists = X_sq + C_sq.T - 2.0 * X_chunk.dot(centroids.T)
    dists = np.maximum(dists, 0)
    chunk_labels = np.argmin(dists, axis=1)
    labels[start_idx: start_idx + len(chunk_labels)] = chunk_labels


def kmeans_assign_multithread(
    X: np.ndarray,
    centroids: np.ndarray,
    n_threads: int = 4,
) -> np.ndarray:

    n_samples = X.shape[0]
    labels = np.empty(n_samples, dtype=np.int32)
    chunk_size = (n_samples + n_threads - 1) // n_threads  # 向上取整
    threads = []

    for i in range(n_threads):
        start = i * chunk_size
        end = min(start + chunk_size, n_samples)
        if start >= n_samples:
            break

        X_chunk = X[start:end]
        t = threading.Thread(
            target=_assign_chunk,
            args=(X_chunk, centroids, start, labels),
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return labels

def verify_consistency(labels_single, labels_multi) -> bool:
    """验证单线程与多线程结果一致性"""
    match = np.array_equal(labels_single, labels_multi)
    print(f"\n[验证] 单线程 vs 多线程结果一致性: {'通过' if match else '失败'}")
    if not match:
        diff_count = np.sum(labels_single != labels_multi)
        print(f"       不一致的样本数: {diff_count}")
    return match


def benchmark_thread_counts(X, centroids, thread_counts):
    """测试不同线程数下的性能"""
    print(f"\n{'='*60}")
    print(f"多线程性能测试 (线程数对比)")
    print(f"{'='*60}")
    print(f"{'线程数':<8} {'耗时(秒)':<12} {'相对加速比':<12}")
    print(f"{'-'*32}")

    single_time = None
    for n_t in thread_counts:
        start = time.perf_counter()
        _ = kmeans_assign_multithread(X, centroids, n_threads=n_t)
        elapsed = time.perf_counter() - start

        if n_t == 1:
            single_time = elapsed

        if single_time is not None and elapsed > 0:
            speedup = single_time / elapsed
        else:
            speedup = 1.0

        print(f"{n_t:<8} {elapsed:<12.4f} {speedup:<12.2f}x")

def main():
    # 解析命令行参数
    if len(sys.argv) < 2:
        csv_path = 'OnlineRetail.csv'
    else:
        csv_path = sys.argv[1]

    if not os.path.exists(csv_path):
        print(f"错误: 找不到文件 '{csv_path}'")
        print("用法: python kmeans_multithread.py <csv文件路径>")
        sys.exit(1)

    # 参数配置
    K = 8          # 聚类数
    N_THREADS = 4  # 默认线程数

    print("=" * 60)
    print("多线程 K-Means 分配步骤实现")
    print("=" * 60)
    X = load_and_preprocess(csv_path)

    centroids = init_centroids(X, k=K)
    print(f"\n[配置] 聚类数 K={K}, 线程数={N_THREADS}")

    print("\n[执行] 单线程分配步骤 ...")
    start_single = time.perf_counter()
    labels_single = kmeans_assign_single(X, centroids)
    elapsed_single = time.perf_counter() - start_single
    print(f"[结果] 单线程耗时: {elapsed_single:.4f} 秒")

    print(f"\n[执行] 多线程({N_THREADS}线程)分配步骤 ...")
    start_multi = time.perf_counter()
    labels_multi = kmeans_assign_multithread(X, centroids, n_threads=N_THREADS)
    elapsed_multi = time.perf_counter() - start_multi
    print(f"[结果] 多线程耗时: {elapsed_multi:.4f} 秒")

    verify_consistency(labels_single, labels_multi)

    unique, counts = np.unique(labels_multi, return_counts=True)
    print("\n[结果] 各簇样本分布:")
    for idx, cnt in zip(unique, counts):
        print(f"  簇 {idx}: {cnt} 个样本")
    benchmark_thread_counts(X, centroids, thread_counts=[1, 2, 4, 8, 16])
if __name__ == "__main__":
    main()