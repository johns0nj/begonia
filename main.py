import numpy as np
import pandas as pd
from scipy.optimize import minimize

class RiskParity:
    def __init__(self, returns_data):
        """
        初始化风险平价策略
        returns_data: 资产收益率数据框(DataFrame)
        """
        self.returns = returns_data  # 资产收益率数据框
        self.cov_matrix = returns_data.cov()  # 协方差矩阵
        self.n_assets = len(returns_data.columns)  # 资产数量
        
    def risk_contribution(self, weights):
        """
        计算每个资产的风险贡献
        """
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))  # 计算投资组合的波动率
        marginal_risk_contrib = np.dot(self.cov_matrix, weights) / portfolio_vol  # 计算边际风险贡献，协方差矩阵*权重/波动率
        risk_contrib = weights * marginal_risk_contrib  # 计算每个资产的风险贡献
        return risk_contrib
    
    def risk_parity_objective(self, weights):
        """
        优化目标函数：使所有资产的风险贡献相等
        """
        risk_contrib = self.risk_contribution(weights)
        target_risk = np.mean(risk_contrib)  # 计算目标风险
        sum_sq_diff = sum([(rc - target_risk)**2 for rc in risk_contrib])  # 计算平方差和
        return sum_sq_diff
    
    def optimize(self):
        """
        优化投资组合权重
        """
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 权重和为1
        ]
        bounds = tuple((0, 1) for _ in range(self.n_assets))  # 权重在0到1之间
        
        # 初始猜测：均等权重
        initial_weights = np.array([1/self.n_assets] * self.n_assets)
        
        # 优化求解
        result = minimize(  # 使用SLSQP方法求解
            self.risk_parity_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return pd.Series(result.x, index=self.returns.columns)

# 使用示例
def run_example():
    # 创建示例数据
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    assets = ['股票', '债券', '商品', '黄金']
    
    returns = pd.DataFrame(
        np.random.randn(len(dates), len(assets)) * np.array([0.2, 0.1, 0.15, 0.12]),
        index=dates,
        columns=assets
    )
    
    # 构建风险平价组合
    rp = RiskParity(returns)
    weights = rp.optimize()
    
    print("风险平价投资组合权重：")
    print(weights)
    
    # 计算各资产风险贡献
    risk_contrib = rp.risk_contribution(weights.values)
    print("\n各资产风险贡献：")
    for asset, rc in zip(assets, risk_contrib):
        print(f"{asset}: {rc:.4f}")

if __name__ == "__main__":   # 主函数
    run_example()   # 运行示例，但是main如果作为import导入，则不会运行