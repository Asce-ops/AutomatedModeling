# 信用评分卡自动建模
建议在自动建模前完成数据的清洗和预处理工作

自动建模流程：
1. 划分 train, validation 和 oot
2. 原始特征基于信息价值和相关性进行初筛，确定潜在的入模变量
3. 对潜在的入模变量进行单调分箱，并根据分箱结果进行 WOE 变换
4. 从潜在的入模变量中剔除稳定性较差的变量
5. 从离散化后稳定性较好的变量中剔除 iv 值较低的变量
6. 对剩余变量进行逐步回归，确定最终的入模变量
7. 使用入模变量拟合评分卡

自动化建模调用示例:
```python
"""
注意，`target`和`time`都是`data`中字段，其中`target`列中的值必须是0-1变量，`time`列中的值必须是 datetime.date 对象
run 方法中的 split_time 参数也必须传入一个 datetime.date 对象
"""
model: AutomatedModeling = AutomatedModeling(data=data, target=target, time="time", not_features=not_features) # 传入数据以实例化一个 AutomatedModeling 对象
model.run(split_time=critical_date, split_validation_set=False) # 调用对象的 run 方法
```

# 如何发挥主观能动性？
放弃直接调用`run`方法，手动执行`run`方法中执行的代码并 DIY 成您的形状即可
```python
def run(self, split_time: date, split_validation_set: bool = False, validation_pct: float = 0.2,
            origin_iv: float = 0.02, corr: float = 0.7,
            max_n_bins: int = 5,
            psi_threshold: float = 0.1,
            woe_iv: float = 0.02,
            estimator: str = "ols", direction: str = "both", criterion: str = "aic",
            model_score: str = "model_score",
            n_bins: int = 50
            ) -> None:
        """自动建模的主方法

        Args:
            split_time (date): 划分 train 和 oot 的边界日期
            split_validation_set (bool, optional): 是否要在 train 中随机划分 validation. Defaults to False.
            validation_pct (float, optional): 验证集占训练集的比例. Defaults to 0.2.
            origin_iv (float, optional): 初筛时的 iv 阈值. Defaults to 0.02.
            corr (float, optional): 初筛时的相关系数阈值. Defaults to 0.7.
            max_n_bins (int, optional): 潜在入模变量单调分箱时最大分箱个数
            psi_threshold (float, optional): 剔除不稳定变量时的 PSI 阈值. Defaults to 0.1.
            woe_iv (float, optional): 离散化后用 iv 值进行筛选时的 iv 阈值. Defaults to 0.02.
            estimator (str, optional): 用于拟合的模型，["ols" | "lr" | "lasso" | "ridge"]. Defaults to "ols".
            direction (str, optional): 逐步回归的方向，["forward" | "backward" | "both"]. Defaults to "both".
            criterion (str, optional): 评判标准，["aic" | "bic" | "ks" | "auc"]. Defaults to "aic".
            model_score (str, optional): 模型分字段命名. Defaults to "model_score".
            n_bins (int, optional): 计算模型分 PSI 时的分箱个数. Defaults to 50.
        """
        self.split_train_oot(split_time=split_time, split_validation_set=split_validation_set, validation_pct=validation_pct) # 划分训练集、验证集和测试集
        print(f"训练集有 {self.train.shape[0]} 条样本，验证集有 {self.validation.shape[0]} 条样本，测试集有 {self.oot.shape[0]} 条样本")
        self.initial_screening(iv=origin_iv, corr=corr) # 原始特征信息价值和相关性初筛
        print(f"原始数据 IV 和相关性初筛后剩余 {len(self.latent_features)} 个变量，分别是 {self.latent_features}")
        self.monotonic_trend_binning(max_n_bins=max_n_bins) # 将潜在入模变量进行单调分箱并进行 WOE 变换
        stable_features: List[str] = self.eliminate_unstable_features(selected_features=self.latent_features, psi_threshold=psi_threshold) # 从潜在入模变量中剔除稳定性较差的变量
        print(f"从潜在入模变量中剔除不稳定变量后剩余 {len(stable_features)} 个变量，分别是 {stable_features}")
        selected_features: List[str] = self.iv_screening_after_woe(selected_features=stable_features, iv=woe_iv) # 从离散化后稳定性较好的变量中再剔除 iv 值较低的变量
        print(f"从离散化后稳定性较好的变量中再剔除 iv 值较低的变量后剩余 {len(selected_features)} 个变量，分别是 {selected_features}")
        used_features: List[str] = self.stepwise_after_woe_transformer(selected_features=selected_features, estimator=estimator, direction=direction, criterion=criterion) # 逐步回顾确定最终的入模变量
        print(f"经逐步回归确定入模变量共 {len(used_features)} 个，分别是 {used_features}")
        self.fit(used_features=used_features, model_score=model_score) # 拟合评分卡
        evaluation: Dict[str, float] = self.evaluate(n_bins=n_bins) # 模型评价指标
        print(f"训练集上 KS 值为：{evaluation['train_ks']}，AUC 值为：{evaluation['train_auc']}；验证集上 KS 值为：{evaluation['validation_ks']}，AUC 值为 {evaluation['validation_auc']}；测试集上 KS 值为：{evaluation['oot_ks']}，AUC 值为 {evaluation['oot_auc']}；模型分的 PSI 为 {evaluation['model_psi']}")
```
通过`AutomatedModeling`对象的`adjust_binning_rules`方法来手动分箱；注意，您只能对潜在的入模变量（`AutomatedModeling`对象的`latent_features`属性）进行手动分箱。如果您想直接干预最终的入模变量，修改传入给`fit`方法的`used_features`参数即可。