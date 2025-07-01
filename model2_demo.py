import gurobipy as gp
from gurobipy import GRB
import numpy as np

# --------------------------
# 数据准备（示例数据，需替换为实际数据）
# --------------------------
N = 50  # 订单总数
M = 20  # 货架总数
P = 10  # 商品种类总数
K = 5  # 最大批次数量
E = 32  # 每批次最大订单数

# 生成示例数据 (需替换为实际数据读取逻辑)
a = np.random.randint(0, 2, (P, N))  # a_in矩阵
x = np.random.randint(0, 2, (P, M))  # x_im矩阵

# --------------------------
# 模型构建
# --------------------------
model = gp.Model("OrderBatchScheduling")

# 定义决策变量
z = model.addVars(N, K, vtype=GRB.BINARY, name="z")  # 订单分配
w = model.addVars(K, M, vtype=GRB.BINARY, name="w")  # 货架搬运
u = model.addVars(P, K, vtype=GRB.BINARY, name="u")  # 商品批次归属

# 目标函数：最小化货架搬运次数
model.setObjective(gp.quicksum(w[k, m] for k in range(K) for m in range(M)), GRB.MINIMIZE)

# --------------------------
# 约束条件
# --------------------------
# 约束10：每个订单必须分配到且仅分配到一个批次
for n in range(N):
    model.addConstr(gp.quicksum(z[n, k] for k in range(K)) == 1, name=f"C10_{n}")

# 约束11：每个批次订单数不超过E
for k in range(K):
    model.addConstr(gp.quicksum(z[n, k] for n in range(N)) <= E, name=f"C11_{k}")

# 约束12：商品批次归属逻辑
for i in range(P):
    for k in range(K):
        expr = gp.LinExpr()
        for n in range(N):
            expr += a[i, n] * z[n, k]
        model.addConstr(expr <= P * u[i, k], name=f"C12_{i}_{k}")

# 约束13：货架必须包含批次所有商品
for i in range(P):
    for k in range(K):
        model.addConstr(gp.quicksum(x[i, m] * w[k, m] for m in range(M)) >= u[i, k], name=f"C13_{i}_{k}")

# --------------------------
# 模型求解与结果输出
# --------------------------
model.optimize()

# 检查求解状态
if model.status == GRB.OPTIMAL:
    print("\n优化结果:")
    print(f"总搬运次数: {model.objVal}")

    # 输出批次分配详情
    for k in range(K):
        assigned_orders = [n for n in range(N) if z[n, k].X >= 1]
        if assigned_orders:
            u_ik = [i for i in range(P) if u[i, k].X >= 1]
            print(f"中间变量u_ik{u_ik}")
            print(f"批次{k + 1},{len(assigned_orders)}个订单:{assigned_orders}")
            moved_shelves = [m for m in range(M) if w[k, m].X >= 1]
            print(f"搬运货架: {moved_shelves}")
else:
    print("未找到最优解")

# # --------------------------
# # 模型分析工具（可选）
# # --------------------------
# # 1. 松弛模型分析
# model_relaxed = model.relax()
# model_relaxed.optimize()
# print(f"\n松弛模型下界: {model_relaxed.objVal}")
#
# # 2. 灵敏度分析
# for v in model.getVars()[:10]:  # 查看前10个变量
#     print(f"{v.VarName}: Reduced Cost = {v.RC}")