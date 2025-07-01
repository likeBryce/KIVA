import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time

class SKUAssociation_Gurobi_Solver:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.orders = None
        self.association_matrix = None
        self.G = None
        self.skus = None
        self.analysis_results = {}

    def load_data(self):
        """加载和预处理数据"""
        print("正在加载数据...")
        time1 = time.time()
        try:
            self.df = pd.read_excel(self.data_path)
            self.df['date'] = pd.to_datetime(self.df['date_crt'])
            self.df['mat_code'] = self.df['mat_code'].astype(str)
            print(f"数据加载成功！数据大小: {self.df.shape}, 加载数据用时：{time.time() - time1} \n")
            return True
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False # 返回False表示加载失败, 程序将退出

    def process_orders(self):
        """处理订单数据，生成SKU集合"""
        print("Step1:正在处理订单数据...")
        # 按订单分组生成SKU集合
        self.orders = self.df.groupby('order_num')['mat_code'].apply(set).reset_index(name='skus')
        
        # 获取所有唯一SKU - 修复：正确合并所有订单中的SKU
        all_skus = set()
        for sku_set in self.orders['skus']:
            all_skus.update(sku_set)
        self.skus = sorted(list(all_skus))  # 排序保证一致性
        
        print(f"订单总数: {len(self.orders)}")
        print(f"商品总数: {len(self.skus)}\n")
        
        # 保存基础统计信息
        self.analysis_results['total_orders'] = len(self.orders)
        self.analysis_results['total_skus'] = len(self.skus)
        
    def build_association_matrix(self):
        """构建商品关联矩阵"""
        print("Step2:正在构建商品关联矩阵...")
        num_skus = len(self.skus)
        self.association_matrix = np.zeros((num_skus, num_skus))
        
        # 创建SKU到索引的映射
        sku_to_idx = {sku: idx for idx, sku in enumerate(self.skus)}
        
        # 计算共现频次
        for _, row in self.orders.iterrows():
            sku_list = list(row['skus'])
            for i, sku1 in enumerate(sku_list):
                idx1 = sku_to_idx[sku1]
                # 对角线元素：商品出现频次
                self.association_matrix[idx1][idx1] += 1
                # 非对角线元素：共现频次
                for j, sku2 in enumerate(sku_list):
                    if i != j:
                        idx2 = sku_to_idx[sku2]
                        self.association_matrix[idx1][idx2] += 1    
        print("商品关联矩阵构建完成！\n")

    def solve_product_clustering_model(self, M, max_categories_upper_bound):
        """
        使用Gurobi求解商品关联网络分解模型
        
        参数:
        s_ij: 商品关联度矩阵
        M: 每个货架最多存储的商品种类数
        max_categories_upper_bound: 商品类别数的上界（用于定义变量范围）
        """
        print(f"Step3:正在使用Gurobi求解商品关联网络分解模型...")
        s_ij = self.association_matrix
        P = s_ij.shape[0]  # 商品数量
        
        # 创建模型
        model = gp.Model("ProductClustering")
        
        # 决策变量
        # x_i: 商品i分散存储的商品类数量（整数变量）
        x = model.addVars(P, vtype=GRB.INTEGER, name="x")
        
        # y_ic: 商品类c包含商品i则为1，否则为0（二进制变量）
        y = model.addVars(P, max_categories_upper_bound, vtype=GRB.BINARY, name="y")
        
        # z_c: 商品类c是否被使用（二进制变量）
        z = model.addVars(max_categories_upper_bound, vtype=GRB.BINARY, name="z")
        
        # C: 实际使用的商品类别数（连续变量，通过z_c的和来确定）
        C = model.addVar(vtype=GRB.INTEGER, name="C")
        
        # 目标函数：最大化商品关联度的总和，同时最小化使用的类别数
        # 使用加权目标函数：关联度收益 - 类别数惩罚
        alpha = 0.01  # 类别数惩罚系数，可调整
        objective = (gp.quicksum(y[i, c] * y[j, c] * s_ij[i, j] 
                                for i in range(P) 
                                for j in range(P) 
                                for c in range(max_categories_upper_bound)
                                if i != j) - alpha * C)
        
        model.setObjective(objective, GRB.MAXIMIZE)
        
        # 约束条件
        
        # 约束(4.2): 每个商品i必须分配到某个商品类
        for i in range(P):
            model.addConstr(gp.quicksum(y[i, c] for c in range(max_categories_upper_bound)) == x[i], 
                        name=f"assign_item_{i}")
        
        # 约束(4.3): 商品i的分散存储数量上限
        U = 5  # 假设上限为5，可根据实际情况调整
        for i in range(P):
            model.addConstr(x[i] <= U, name=f"upper_bound_{i}")
        
        # 约束(4.4): 每个商品类c最多包含M个商品
        for c in range(max_categories_upper_bound):
            model.addConstr(gp.quicksum(y[i, c] for i in range(P)) <= M * z[c], 
                        name=f"category_capacity_{c}")
        
        # 约束(4.5): 每个商品i至少分配到1个商品类
        for i in range(P):
            model.addConstr(gp.quicksum(y[i, c] for c in range(max_categories_upper_bound)) >= 1, 
                        name=f"min_assign_{i}")
        
        # 约束(4.6): 如果商品类c被使用，则至少包含1个商品
        for c in range(max_categories_upper_bound):
            model.addConstr(gp.quicksum(y[i, c] for i in range(P)) >= z[c], 
                        name=f"min_category_{c}")
        
        # 新增约束1: 如果商品类c包含商品，则该类被标记为使用
        for c in range(max_categories_upper_bound):
            model.addConstr(gp.quicksum(y[i, c] for i in range(P)) <= P * z[c], 
                        name=f"category_usage_{c}")
        
        # 新增约束2: 计算实际使用的商品类别数
        model.addConstr(C == gp.quicksum(z[c] for c in range(max_categories_upper_bound)), 
                    name="total_categories")
        
        # 约束(4.7): x_i为非负整数（已在变量定义中指定）
        
        # 约束(4.8): y_ic为二进制变量（已在变量定义中指定）
        
        # 设置求解参数
        model.setParam('TimeLimit', 60)  # 设置时间限制为1小时
        model.setParam('MIPGap', 0.01)     # 设置相对间隙为1%
        
        # 求解模型
        model.optimize()
        
        # 输出结果
        if model.status == GRB.OPTIMAL:
            print(f"找到最优解！")
            print(f"目标函数值: {model.objVal:.4f}")
            
            # 提取解
            x_solution = {i: x[i].x for i in range(P)}
            y_solution = {(i, c): y[i, c].x for i in range(P) for c in range(max_categories_upper_bound) if y[i, c].x > 0.5}
            z_solution = {c: z[c].x for c in range(max_categories_upper_bound) if z[c].x > 0.5}
            actual_categories = int(C.x)
            
            # 分析结果
            used_categories = set(c for (i, c) in y_solution.keys())
            print(f"优化得到的商品类别数: {actual_categories}")
            print(f"实际使用的商品类别数: {len(used_categories)}")
            
            # # 输出每个商品类别包含的商品
            # for c in sorted(used_categories):
            #     items_in_category = [i for (i, cat) in y_solution.keys() if cat == c]
            #     print(f"商品类别 {c}: 包含商品 {items_in_category} (共{len(items_in_category)}个商品)")
            
            return x_solution, y_solution, model.objVal, actual_categories
        
        elif model.status == GRB.INFEASIBLE:
            print("模型不可行！")
            model.computeIIS()
            model.write("infeasible.ilp")
            return None, None, None
        
        elif model.status == GRB.TIME_LIMIT:
            print("达到时间限制！")
            if model.solCount > 0:
                print(f"找到可行解，目标函数值: {model.objVal:.4f}")
                x_solution = {i: x[i].x for i in range(P)}
                y_solution = {(i, c): y[i, c].x for i in range(P) for c in range(max_categories_upper_bound) if y[i, c].x > 0.5}
                actual_categories = int(C.x)
                return x_solution, y_solution, model.objVal, actual_categories
            else:
                print("未找到可行解")
                return None, None, None
        
        else:
            print(f"求解状态: {model.status}")
            return None, None, None

    def run_complete_analysis(self):
        # 1. 加载数据
        if not self.load_data():
            return False
        # 2. 处理订单
        self.process_orders()
        
        # 3. 构建关联矩阵
        self.build_association_matrix()

        # 4. Gurobi求解
        self.solve_product_clustering_model(M=20, max_categories_upper_bound=410)

if __name__ == '__main__':
    """
    主函数：导入数据并求解模型
    """
    # 创建分析器实例
    analyzer = SKUAssociation_Gurobi_Solver("d:\\Mycode\\KIVA\\data\\outbound7-23.xlsx")
    
    # 运行完整分析
    analyzer.run_complete_analysis()