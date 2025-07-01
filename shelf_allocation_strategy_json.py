"""  
基于JSON数据的货架分配策略实现
从JSON文件加载网络分析结果，加快测试速度
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import seaborn as sns
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ShelfAllocationStrategyJSON:
    """
    基于JSON数据的货架分配策略类
    从JSON文件加载网络分析结果，实现DBSCAN聚类和SKUs-to-cluster分配算法
    """
    
    def __init__(self, json_path="d:\\Mycode\\KIVA\\result\\problem1\\analysis_results.json"):
        """
        初始化货架分配策略
        
        Args:
            json_path: JSON文件路径，包含网络分析结果
        """
        self.json_path = json_path
        self.load_data_from_json()
        
        # 聚类相关属性
        self.clusters = None
        self.cluster_labels = None
        self.cluster_centers = None
        self.noise_points = None
        
        # 货架分配相关属性
        self.shelf_allocation = None
        self.allocation_strategy = None
        
    def load_data_from_json(self):
        """从JSON文件加载数据"""
        print(f"正在从JSON文件加载数据: {self.json_path}")
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取数据
        self.skus = data['skus']
        self.association_matrix = np.array(data['association_matrix'])
        self.analysis_results = data['analysis_results']
        
        # 重建网络图
        self.G = nx.Graph()
        self.G.add_nodes_from(self.skus)
        
        # 添加边（使用阈值过滤）
        threshold = 0.1  # 可以调整这个阈值
        num_skus = len(self.skus)
        for i in range(num_skus):
            for j in range(i+1, num_skus):
                weight = self.association_matrix[i][j]
                if weight >= threshold:
                    self.G.add_edge(self.skus[i], self.skus[j], weight=weight)
        
        print(f"数据加载完成！SKU数量: {len(self.skus)}, 网络节点: {self.G.number_of_nodes()}, 网络边: {self.G.number_of_edges()}")

    def extract_network_features(self):
        """
        从网络中提取商品特征用于聚类
        
        Returns:
            feature_matrix: 特征矩阵
            feature_names: 特征名称列表
        """
        print("正在提取网络特征...")
        
        features = []
        feature_names = [
            'degree',           # 度
            'weighted_degree',  # 加权度
            'sku_sales',        # 销量
            'sku_purchase_freq', # 购买频率
            'sku_hot_sale_inx'  # 长短期销量
        ]
        
        # 计算网络指标
        degrees = dict(self.G.degree())
        weighted_degrees = dict(self.G.degree(weight='weight'))
        
        # 获取销量, 购买频次, 长短期畅销指数
        sku_sales = self.analysis_results.get('sku_sales', {})
        sku_purchase_freq = self.analysis_results.get('sku_purchase_freq', {})
        sku_hot_sale_inx = self.analysis_results.get('sku_hot_sale_inx', {})
        
        # 为每个SKU构建特征向量
        for sku in self.skus:       
            feature_vector = [
                degrees.get(sku, 0),
                weighted_degrees.get(sku, 0),
                sku_sales.get(sku, 0),
                sku_purchase_freq.get(sku, 0),
                sku_hot_sale_inx.get(sku, 0)
            ]
            features.append(feature_vector)
        
        feature_matrix = np.array(features)
        print(f"特征提取完成，特征矩阵形状: {feature_matrix.shape}")
        
        return feature_matrix, feature_names

    def perform_dbscan_clustering(self, eps=0.5, min_samples=5, normalize=True):
        """
        执行DBSCAN聚类
        
        Args:
            eps: DBSCAN的eps参数
            min_samples: DBSCAN的min_samples参数
            normalize: 是否标准化特征
            
        Returns:
            clustering_results: 聚类结果字典
        """
        print(f"正在执行DBSCAN聚类 (eps={eps}, min_samples={min_samples})...")
        
        # 提取特征
        feature_matrix, feature_names = self.extract_network_features()
        
        # 标准化特征
        if normalize:
            scaler = StandardScaler()
            feature_matrix = scaler.fit_transform(feature_matrix)
        
        # 执行DBSCAN聚类
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(feature_matrix)
        
        # 分析聚类结果
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        print(f"聚类完成！聚类数: {n_clusters}, 噪声点数: {n_noise}")
        
        # 保存聚类结果
        self.cluster_labels = cluster_labels
        self.clusters = {}
        self.noise_points = []
        
        for i, label in enumerate(cluster_labels):
            sku = self.skus[i]
            if label == -1:
                self.noise_points.append(sku)
            else:
                if label not in self.clusters:
                    self.clusters[label] = []
                self.clusters[label].append(sku)
        
        # 计算聚类质量指标
        clustering_results = {
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'cluster_labels': cluster_labels,
            'clusters': self.clusters,
            'noise_points': self.noise_points
        }
        
        # 计算评估指标（如果有有效聚类）
        if n_clusters > 1 and len(set(cluster_labels)) > 1:
            # 过滤噪声点
            valid_indices = cluster_labels != -1
            if np.sum(valid_indices) > 1:
                valid_features = feature_matrix[valid_indices]
                valid_labels = cluster_labels[valid_indices]
                
                if len(set(valid_labels)) > 1:
                    clustering_results['silhouette_score'] = silhouette_score(valid_features, valid_labels)
                    clustering_results['calinski_harabasz_score'] = calinski_harabasz_score(valid_features, valid_labels)
                    clustering_results['davies_bouldin_score'] = davies_bouldin_score(valid_features, valid_labels)
        
        return clustering_results

    def allocate_skus_to_shelves(self, num_shelves, M=20, allocation_method='correlation_based'):
        """
        将SKU分配到货架
        
        Args:
            num_shelves: 货架数量
            M: 每个货架的最大SKU容量
            allocation_method: 分配方法 ('correlation_based' 或 'cluster_based')
        
        Returns:
            shelf_allocation: 货架分配结果
        """
        if allocation_method == 'correlation_based':
            return self._correlation_based_allocation(num_shelves, M)
        else:
            return self._cluster_based_allocation(num_shelves)
    
    def _correlation_based_allocation(self, num_shelves, M):
        """
        基于关联性的SKU到货架分配算法 (Algorithm 2)
        根据论文中的伪代码实现
        
        Args:
            num_shelves: 货架数量 |R|
            M: 每个货架的最大SKU容量
        
        Returns:
            shelf_allocation: 货架分配结果 y_i,c
        """
        print(f"执行基于关联性的分配算法，货架数量: {num_shelves}, 容量限制: M={M}")
        
        # 获取关联矩阵和SKU列表
        if not hasattr(self, 'association_matrix') or self.association_matrix is None:
            print("错误：缺少关联矩阵，请先运行网络分析")
            return {i: [] for i in range(num_shelves)}
        
        # 初始化
        R = min(num_shelves, len(self.skus) // M + 1)  # 实际需要的货架数量
        shelf_allocation = {c: [] for c in range(R)}  # y_i,c = 0 for all i,c
        
        # 创建SKU到索引的映射
        sku_to_idx = {sku: i for i, sku in enumerate(self.skus)}
        idx_to_sku = {i: sku for i, sku in enumerate(self.skus)}
        
        # 获取关联矩阵
        n_skus = len(self.skus)
        correlation_matrix = self.association_matrix
        
        # 计算每个SKU的权重 w_i (使用度中心性)
        degrees = dict(self.G.degree())
        weights = {sku: degrees.get(sku, 0) for sku in self.skus}
        
        # 可用SKU集合 I
        available_skus = set(range(n_skus))
        
        # Algorithm 2: SKUs-to-cluster assignment based on correlation
        for c in range(R):  # for c = 1 to |R| do
            if not available_skus:
                break
                
            # Step 4: initial clustering center i ← max(w_i)
            center_idx = max(available_skus, key=lambda i: weights[idx_to_sku[i]])
            
            # Step 5: y_i,c ← y_i,c + 1
            shelf_allocation[c].append(idx_to_sku[center_idx])
            
            # Step 6: x_i ← x_i - 1 (从可用集合中移除)
            available_skus.remove(center_idx)
            
            # Step 10-19: while |c| ≤ M do
            while len(shelf_allocation[c]) < M and available_skus:
                # Step 11: selecting SKU j ← max(Σ_{i∈c} s_{i,j})
                best_sku_idx = None
                max_correlation = -1
                
                for j_idx in available_skus:
                    # 计算与当前货架中所有SKU的关联度总和
                    total_correlation = 0
                    for allocated_sku in shelf_allocation[c]:
                        allocated_idx = sku_to_idx[allocated_sku]
                        total_correlation += correlation_matrix[allocated_idx][j_idx]
                    
                    if total_correlation > max_correlation:
                        max_correlation = total_correlation
                        best_sku_idx = j_idx
                
                # Step 12-13: 如果最大关联度为0，选择距离最小的SKU
                if max_correlation == 0 and available_skus:
                    # selecting SKU j ← min(Σ_{i∈c} √(s_{j,j} - s_{i,j})²)
                    min_distance = float('inf')
                    for j_idx in available_skus:
                        total_distance = 0
                        for allocated_sku in shelf_allocation[c]:
                            allocated_idx = sku_to_idx[allocated_sku]
                            # 简化距离计算
                            distance = abs(correlation_matrix[j_idx][j_idx] - correlation_matrix[allocated_idx][j_idx])
                            total_distance += distance
                        
                        if total_distance < min_distance:
                            min_distance = total_distance
                            best_sku_idx = j_idx
                
                if best_sku_idx is not None:
                    # Step 14: add j to c
                    # Step 15: y_j,c ← y_j,c + 1
                    shelf_allocation[c].append(idx_to_sku[best_sku_idx])
                    
                    # Step 16: x_j ← x_j - 1
                    available_skus.remove(best_sku_idx)
                else:
                    break
                
                # Step 20-21: if |I| == 0 then break
                if not available_skus:
                    break
        
        # 处理剩余的SKU（如果有）
        if available_skus:
            remaining_skus = [idx_to_sku[i] for i in available_skus]
            self._distribute_remaining_skus(remaining_skus, shelf_allocation, R)
        
        # 扩展到所需的货架数量
        final_allocation = {i: [] for i in range(num_shelves)}
        for i in range(min(R, num_shelves)):
            final_allocation[i] = shelf_allocation[i]
        
        print(f"分配完成！使用了 {R} 个货架，平均每个货架 {sum(len(skus) for skus in shelf_allocation.values()) / R:.1f} 个SKU")
        
        return final_allocation
    
    def _cluster_based_allocation(self, num_shelves):
        """
        基于聚类的SKU到货架分配
        
        Args:
            num_shelves: 货架数量
        
        Returns:
            shelf_allocation: 货架分配结果
        """
        if self.clusters is None:
            print("错误：请先执行聚类分析")
            return {i: [] for i in range(num_shelves)}
        
        shelf_allocation = {i: [] for i in range(num_shelves)}
        
        # 将聚类分配到货架
        cluster_ids = list(self.clusters.keys())
        for i, cluster_id in enumerate(cluster_ids[:num_shelves]):
            shelf_allocation[i] = self.clusters[cluster_id].copy()
        
        # 处理噪声点
        if self.noise_points:
            self._distribute_remaining_skus(self.noise_points, shelf_allocation, num_shelves)
        
        return shelf_allocation
    
    def _distribute_remaining_skus(self, remaining_skus, shelf_allocation, num_shelves):
        """
        将剩余的SKU分配到货架中，优先分配到SKU数量最少的货架
        
        Args:
            remaining_skus: 剩余的SKU列表
            shelf_allocation: 当前的货架分配字典
            num_shelves: 货架数量
        """
        for sku in remaining_skus:
            # 找到SKU数量最少的货架
            min_shelf = min(range(num_shelves), key=lambda x: len(shelf_allocation[x]))
            shelf_allocation[min_shelf].append(sku)

    def optimize_clustering_parameters(self, eps_range=None, min_samples_range=None):
        """
        优化DBSCAN聚类参数
        使用Silhouette Coefficient, Calinski-Harabasz Index, Davies-Bouldin Index进行综合评估
        
        Args:
            eps_range: eps参数范围
            min_samples_range: min_samples参数范围
            
        Returns:
            best_params: 最佳参数
            all_results: 所有参数组合的结果
        """
        print("正在优化聚类参数...")
        
        if eps_range is None:
            eps_range = np.arange(0.3, 1.5, 0.1)
        if min_samples_range is None:
            min_samples_range = range(1, 8)  # 包含1
        
        all_results = []
        best_composite_score = -1
        best_params = None
        
        for eps in eps_range:
            for min_samples in min_samples_range:
                try:
                    result = self.perform_dbscan_clustering(eps=eps, min_samples=min_samples)
                    
                    # 计算综合评分
                    if ('silhouette_score' in result and 
                        'calinski_harabasz_score' in result and 
                        'davies_bouldin_score' in result):
                        
                        # 标准化各个指标
                        silhouette = result['silhouette_score']  # 范围[-1,1]，越大越好
                        ch_score = result['calinski_harabasz_score']  # 越大越好
                        db_score = result['davies_bouldin_score']  # 越小越好
                        
                        # 标准化处理
                        norm_silhouette = (silhouette + 1) / 2  # 转换到[0,1]
                        norm_ch = min(ch_score / 1000, 1)  # 简单标准化
                        norm_db = max(0, 1 - db_score / 10)  # 转换为越大越好
                        
                        # 加权综合评分
                        composite_score = (0.4 * norm_silhouette + 
                                         0.3 * norm_ch + 
                                         0.3 * norm_db)
                        
                        result['composite_score'] = composite_score
                        
                        if composite_score > best_composite_score:
                            best_composite_score = composite_score
                            best_params = {'eps': eps, 'min_samples': min_samples}
                    
                    all_results.append(result)
                    
                except Exception as e:
                    print(f"参数组合 eps={eps}, min_samples={min_samples} 失败: {e}")
                    continue
        
        if best_params:
            print(f"\n最佳参数: eps={best_params['eps']:.2f}, min_samples={best_params['min_samples']}")
            
            # 找到最佳结果并打印详细信息
            best_result = None
            for result in all_results:
                if (result['eps'] == best_params['eps'] and 
                    result['min_samples'] == best_params['min_samples']):
                    best_result = result
                    break
            
            if best_result and 'composite_score' in best_result:
                print(f"综合评分: {best_result['composite_score']:.4f}")
                print(f"轮廓系数: {best_result.get('silhouette_score', 'N/A'):.4f}")
                print(f"CH指数: {best_result.get('calinski_harabasz_score', 'N/A'):.2f}")
                print(f"DB指数: {best_result.get('davies_bouldin_score', 'N/A'):.4f}")
                print(f"聚类数: {best_result['n_clusters']}")
                print(f"噪声点数: {best_result['n_noise']}")
        
        return best_params, all_results

    def run_complete_allocation_analysis(self, eps=0.8, min_samples=3, num_shelves=20, 
                                       allocation_method='cluster_based', optimize_params=True, M=20):
        """
        运行完整的货架分配分析
        
        Args:
            eps: DBSCAN的eps参数
            min_samples: DBSCAN的min_samples参数
            num_shelves: 货架数量
            allocation_method: 分配方法
            optimize_params: 是否优化参数
            
        Returns:
            results: 分析结果
        """
        print("开始完整的货架分配分析...")
        
        # 1. 参数优化（如果启用）
        if optimize_params:
            print("\n=== 参数优化阶段 ===")
            best_params, results = self.optimize_clustering_parameters()
            if best_params:
                eps = best_params['eps']
                min_samples = best_params['min_samples']
                print(f"使用优化后的参数: eps={eps}, min_samples={min_samples}")
        
        # 2. 使用最佳参数进行聚类
        print("\n=== 最终聚类阶段 ===")
        clustering_result = self.perform_dbscan_clustering(eps=eps, min_samples=min_samples)
        
        # 3. 执行货架分配
        print("\n=== 货架分配阶段 ===")
        shelf_allocation = self.allocate_skus_to_shelves(
            num_shelves=num_shelves, 
            M=M, 
            allocation_method=allocation_method
        )
        
        print("\n货架分配分析完成！")
        
        return {
            'clusters': self.clusters,
            'clustering_result': clustering_result,
            'shelf_allocation': shelf_allocation
        }

# 使用示例
if __name__ == "__main__":
    # 创建基于JSON的货架分配策略
    shelf_strategy = ShelfAllocationStrategyJSON()
    
    # 运行完整的分配分析
    results = shelf_strategy.run_complete_allocation_analysis(
        eps=0.8,
        min_samples=3,
        num_shelves=20,
        allocation_method='correlation_based',  # 使用基于关联性的分配方法
        optimize_params=True,  # 启动参数优化
        M=20  # 每个货架的最大SKU容量
    )
    
    print("\n分析完成！")