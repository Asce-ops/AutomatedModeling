from typing import List, Dict, Any, ClassVar #, override
from datetime import date
from random import seed
from pickle import dump
import warnings
warnings.filterwarnings(action="ignore") # 忽略所有警告

from toad import quality
from toad.metrics import KS, KS_bucket, PSI, AUC
import xgboost as xgb
from xgboost import DMatrix
from xgboost.core import Booster
from xgboost import plot_importance, plot_tree
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from matplotlib import pyplot as plt
from seaborn import histplot 
plt.rcParams["font.family"] = "SimSun" # 替换为你选择的字体（否则绘图中可能无法正常显示中文）

from AutomatedModeling import AutomatedModeling


class AutomatedXGBoost(AutomatedModeling):
    params: ClassVar[Dict[str, Any]] = { # 默认的模型参数
        # General Parameters
        "booster": "gbtree", # 基学习器类型，可选 {"gbtree": "树模型", "gblinear": "线性模型"}
        "nthread": 8, # XGBoost 运行时的线程数
        
        # Booster Parameters
        "eta": 0.1, # 收缩步长，[0, 1]
        "gamma": 1, # 分裂节点所需的最小损失函数下降值，[0, +∞)
        "max_depth": 2, # 树的最大深度，[1, +∞)
        "min_child_weight": 5, # 叶子节点最小的样本权重和，如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束，[0, +∞)
        "subsample": 0.6, # 每次训练所使用的数据的比例，(0, 1]
        "colsample_bytree": 0.8,  # 每棵树特征采样的比例，(0, 1]
        
        # Task Parameters
        "objective": "binary:logistic", # 学习任务及相应的学习目标。可选 {"reg:linear": "线性回归", "reg:logistic": "逻辑回归", "binary:logistic": "二分类的逻辑回归问题，输出为概率", "binary:logitraw": "二分类的逻辑回归问题，输出的结果为wTx", ...}
        "eval_metric": ["error", "logloss", "auc"], # 评价指标
        "seed": 42, # 随机种子
        # "num_boost_round": 30, # 基模型的数量（该参数似乎只能在 Booster 对象的 train 方法中以关键字参数传入才有效）
        "lambda": 5, # L2正则化系数
        "alpha": 3, # L1正则化系数
        "scale_pos_weight": 1
    }

    def __init__(self, data: DataFrame, target: str, time: str, not_features: List[str]) -> None:
        super().__init__(data=data, target=target, time=time, not_features=not_features)
        self.proba: str = "proba"
        self.Ddata: DMatrix
        self.Dtrain: DMatrix
        self.Dvalidation: DMatrix
        self.Doot: DMatrix
        self.model: Booster # Booster 对象
        self.latent_features: List[str] # 潜在入模变量

    # @override
    def split_train_oot(self, split_time: date, split_validation_set: bool = False, validation_pct: float = 0.2) -> None:
        """划分训练集和测试集（如不划分验证集则默认将测试集复制一份作为验证集）

        Args:
            split_time (date): 用于划分训练集和测试集的临界日期
            split_validation_set (bool, optional): 是否需要划分验证集. Defaults to False.
            validation_pct (float, optional): 验证集占训练集的样本. Defaults to 0.2.
        """
        super().split_train_oot(split_time=split_time, split_validation_set=split_validation_set, validation_pct=validation_pct)
        # 重新划分训练集和测试集后，重新实例化 DMatrix 对象
        # self.Ddata: DMatrix = xgb.DMatrix(data=self.data[self.features], label=self.data[self.target])
        # self.Dtrain: DMatrix = xgb.DMatrix(data=self.train[self.features], label=self.train[self.target])
        # self.Dvalidation: DMatrix = xgb.DMatrix(data=self.validation[self.features], label=self.validation[self.target])
        # self.Doot: DMatrix = xgb.DMatrix(data=self.oot[self.features], label=self.oot[self.target])

    # @override
    def eliminate_low_iv_features(self, selected_features: List[str], iv: float = 0.02, train_validation_oot: int = 1) -> List[str]:
        """剔除 iv 值较低的变量

        Args:
            selected_features (List[str]): 潜在的入模变量
            iv (float, optional): iv 阈值. Defaults to 0.02.
            train_validation_oot (int, optional): {1: "训练集", 0: "测试集", 2: "验证集"}. Defaults to 1.

        Raises:
            ValueError: 参数传值错误

        Returns:
            List[str]: iv 值超过指定阈值的变量
        """
        if train_validation_oot == 0:
            data: DataFrame = self.oot
        elif train_validation_oot == 1:
            data: DataFrame = self.train
        elif train_validation_oot == 2:
            data: DataFrame = self.validation
        else:
            raise ValueError("参数 train_validation_oot 只能传入 0 或 1 或 2")
        distinction: DataFrame = quality(dataframe=data[selected_features+[self.target]], target=self.target, iv_only=True)
        result: List[str] = distinction[distinction["iv"] >= iv].index.to_list()
        return result
    
    # @override
    def eliminate_unstable_features(self, selected_features: List[str], psi_threshold: float = 0.1, psi_original: bool = True) -> List[str]:
        """剔除稳定性较差的变量

        Args:
            selected_features (List[str]): 潜在的入模变量
            psi_threshold (float, optional): psi筛选的阈值. Defaults to 0.1.
            psi_original (bool, optional): 使用原始数据还是特征工程后的数据（True: 原始数据, False: 特征工程后的数据）. Defaults to True.

        Raises:
            ValueError: 参数传值错误
            
        Returns:
            List[str]: 稳定性较好的变量
        """
        if not psi_original:
            raise ValueError("XGBoost 无需特征工程，psi_original 参数只能传入 True")
        selected_features_psi: Series = PSI(test=self.train[selected_features], base=self.oot[selected_features]) # type: ignore
        stable_features: List[str] = selected_features_psi[selected_features_psi < psi_threshold].index.to_list()
        return stable_features
    
    def fit(self, latent_features: List[str], model_score: str = "model_score", num_boost_round: int = 30, **params: Any) -> None:
        """训练模型

        Args:
            model_score (str, optional): 模型分字段. Defaults to "model_score".
            num_boost_round (int, optional): 要生成的基模型的数量. Defaults to 30.
            **params (Any): 传入 xgboost 的参数字典（可用于新增或覆盖默认参数）
        """
        self.latent_features = latent_features

        self.Ddata: DMatrix = xgb.DMatrix(data=self.data[self.latent_features], label=self.data[self.target])
        self.Dtrain: DMatrix = xgb.DMatrix(data=self.train[self.latent_features], label=self.train[self.target])
        self.Dvalidation: DMatrix = xgb.DMatrix(data=self.validation[self.latent_features], label=self.validation[self.target])
        self.Doot: DMatrix = xgb.DMatrix(data=self.oot[self.latent_features], label=self.oot[self.target])

        merged_params: Dict[str, Any] = {**self.params, **params} # 合并默认参数和传入的参数
        # 训练
        self.model = xgb.train(
            params=merged_params,
            num_boost_round=num_boost_round, # 子树的数量
            dtrain=self.Dtrain,
            evals=[(self.Dtrain, "train"), (self.Dvalidation, "validation")],
            verbose_eval=False # 是否在训练过程中打印评估信息：若设为 True，则每次迭代都会输出；若是数字 n，则每 n 轮输出一次
        )

        self.used_features = list(self.model.get_fscore().keys()) # Booster 对象的 feature_names 属性返回的是所有用于训练的特征，不管最终是否入模
        # 预测
        self.score = model_score
        self.data[self.proba] = self.model.predict(data=self.Ddata) # 预测概率
        self.data[self.score] = self.data[self.proba].apply(func=lambda x: AutomatedModeling.proba2score(prob=x)) # 将概率映射为分数

    def run(self, split_time: date, split_validation_set: bool = False, validation_pct: float = 0.2,
            psi_threshold: float = 0.1, psi_original: bool = True,
            iv: float = 0.02, eliminate_low_iv_oot: bool = True,
            model_score: str = "model_score", num_boost_round: int = 30,
            n_bins: int = 50,
            **params: Any
            ) -> None:
        """自动建模的主方法

        Args:
            split_time (date): 划分 train 和 oot 的边界日期
            split_validation_set (bool, optional): 是否要在 train 中随机划分 validation. Defaults to False.
            validation_pct (float, optional): 验证集占训练集的比例. Defaults to 0.2.
            psi_threshold (float, optional): 剔除不稳定变量时的 PSI 阈值. Defaults to 0.1.
            psi_original (bool, optional): 使用原始数据还是特征工程后的数据（True: 原始数据, False: 特征工程后的数据）. Defaults to True.
            iv (float, optional): 用 iv 值进行筛选时的 iv 阈值. Defaults to 0.02.
            eliminate_low_iv_oot (bool, optional): 是否将测试集上 iv 值较低的变量也剔除. Defaults to True.
            model_score (str, optional): 模型分字段命名. Defaults to "model_score".
            num_boost_round (int, optional): 要生成的基模型的数量. Defaults to 30.
            n_bins (int, optional): 计算模型分 PSI 时的分箱个数. Defaults to 50.
            **params (Any): 传入 xgboost 的参数字典（可用于新增或覆盖默认参数）
        """
        self.split_train_oot(split_time=split_time, split_validation_set=split_validation_set, validation_pct=validation_pct) # 划分训练集、验证集和测试集
        print(f"训练集有 {self.train.shape[0]} 条样本，bad_rate 为 {self.train[self.target].mean()}；验证集有 {self.validation.shape[0]} 条样本，bad_rate 为 {self.validation[self.target].mean()}；测试集有 {self.oot.shape[0]} 条样本，bad_rate 为 {self.oot[self.target].mean()}")
        stable_features: List[str] = self.eliminate_unstable_features(selected_features=self.features, psi_threshold=psi_threshold, psi_original=psi_original) # 从潜在入模变量中剔除稳定性较差的变量
        print(f"从所有特征中剔除不稳定变量后剩余 {len(stable_features)} 个变量，分别是 {stable_features}")
        # 从稳定性较好的变量中再剔除 iv 值较低的变量
        selected_features: List[str] = self.eliminate_low_iv_features(selected_features=stable_features, iv=iv, train_validation_oot=1)
        if eliminate_low_iv_oot: # 如果需要剔除测试集上 iv 值较低的变量
            selected_features: List[str] = self.eliminate_low_iv_features(selected_features=selected_features, iv=iv, train_validation_oot=0)
        print(f"从稳定性较好的变量中再剔除 iv 值较低的变量后剩余 {len(selected_features)} 个变量，分别是 {selected_features}")
        self.fit(latent_features=selected_features, model_score=model_score, num_boost_round=num_boost_round, **params) # 拟合 XGB
        print(f"最终的入模变量共 {len(self.used_features)} 个，分别是 {self.used_features}")
        evaluation: Dict[str, float] = self.evaluate(n_bins=n_bins) # 模型评价指标
        print(f"训练集上 KS 值为：{evaluation['train_ks']}，AUC 值为：{evaluation['train_auc']}；验证集上 KS 值为：{evaluation['validation_ks']}，AUC 值为 {evaluation['validation_auc']}；测试集上 KS 值为：{evaluation['oot_ks']}，AUC 值为 {evaluation['oot_auc']}；模型分的 PSI 为 {evaluation['model_psi']}")

    # @override
    def calculate_model_score_psi(self, n_bins: int = 50) -> float:
        """计算模型分离散化后的 PSI

        Args:
            n_bins (int, optional): 计算模型分 PSI 时的分箱个数. Defaults to 50.

        Returns:
            float: 模型分离散化后在训练集和测试集上的 PSI
        """
        tmp: DataFrame = self.data[[self.is_train, self.score]]
        tmp[f"{self.score}_bin"] = pd.qcut(x=tmp[self.score], q=n_bins, duplicates="drop")
        result: float = PSI(test=tmp[tmp[self.is_train]==0][f"{self.score}_bin"], base=tmp[tmp[self.is_train]==1][f"{self.score}_bin"]) # type: ignore
        return result
    
    # @override
    def evaluate(self, n_bins: int = 50) -> Dict[str, float]:
        """模型评估

        Args:
            n_bins (int, optional): 计算模型分 PSI 时的分箱个数. Defaults to 50.

        Returns:
            Dict[str, float]:  模型的量化指标（KS, AUC, model_score_psi）
        """
        self.evaluation = {}
        train_ks: float = KS(score=self.data[self.data[self.is_train]==1][self.score], target=self.data[self.data[self.is_train]==1][self.target])
        validation_ks: float = KS(score=self.data[self.data[self.is_train]==2][self.score], target=self.data[self.data[self.is_train]==2][self.target])
        oot_ks: float = KS(score=self.data[self.data[self.is_train]==0][self.score], target=self.data[self.data[self.is_train]==0][self.target])
        train_auc: float = AUC(score=self.data[self.data[self.is_train]==1][self.score], target=self.data[self.data[self.is_train]==1][self.target]) # type: ignore
        validation_auc: float = AUC(score=self.data[self.data[self.is_train]==2][self.score], target=self.data[self.data[self.is_train]==2][self.target]) # type: ignore
        oot_auc: float = AUC(score=self.data[self.data[self.is_train]==0][self.score], target=self.data[self.data[self.is_train]==0][self.target]) # type: ignore
        self.evaluation["train_ks"] = train_ks
        self.evaluation["validation_ks"] = validation_ks
        self.evaluation["oot_ks"] = oot_ks
        self.evaluation["train_auc"] = train_auc
        self.evaluation["validation_auc"] = validation_auc
        self.evaluation["oot_auc"] = oot_auc
        self.evaluation["model_psi"] = self.calculate_model_score_psi(n_bins=n_bins)
        return self.evaluation
    
    # @override
    def ks_bucket(self, train_validation_oot: int = 0, n_bins: int = 10) -> DataFrame:
        """模型分分箱

        Args:
            train_validation_oot (int, optional): {1: "训练集", 0: "测试集", 2: "验证集"}. Defaults to 0.
            n_bins (int, optional): 分箱数量. Defaults to 10.

        Raises:
            ValueError: 参数传值错误

        Returns:
            DataFrame: 模型分箱后在各个箱中的效果
        """
        if train_validation_oot == 0:
            label: str = "oot"
        elif train_validation_oot == 1:
            label: str = "train"
        elif train_validation_oot == 2:
            label: str = "validation"
        else:
            raise ValueError("参数 train_validation_oot 只能传入 0 或 1 或 2")
        
        result: DataFrame = KS_bucket(
                                score=self.data[self.data[self.is_train]==train_validation_oot][self.score], # 分箱变量
                                target=self.data[self.data[self.is_train]==train_validation_oot][self.target], # 目标变量
                                bucket=n_bins, # 分箱个数，默认为10
                                method="quantile" # 默认等频分箱，{"quantile": 等频分箱, "step": 等距分箱}
                            ) # type: ignore
        
        plt.plot(result["min"], result["bad_rate"], marker="o", label=label)
        plt.xlabel(xlabel=f"{self.score}")
        plt.ylabel(ylabel="bad_rate")
        plt.legend()
        plt.show()
        
        return result
    
    # @override
    def plot_model_score_distribution(self) -> None:
        """绘制训练集、验证集和测试集上的模型分分布图"""
        plt.figure(figsize=(20, 7))
        histplot(
            x=self.data[self.data[self.is_train]==1][self.score], 
            kde=True, # 是否绘制核密度图
            color="red",
            bins=50,
            label=f"{self.score}_distribution in train",
            stat="density" # 统计方法，频率还是频数
        )
        histplot(
            x=self.data[self.data[self.is_train]==0][self.score], 
            kde=True, 
            color="green",
            bins=50,
            label=f"{self.score}_distribution in oot",
            stat="density"
        )
        histplot(
            x=self.data[self.data[self.is_train]==2][self.score], 
            kde=True, 
            color="blue",
            bins=50,
            label=f"{self.score}_distribution in validation",
            stat="density"
        )
        plt.legend()
        plt.title(label=f"Comparison {self.score}_distribution in train, validation and oot")
        plt.xlabel(xlabel=f"{self.score}")
        plt.ylabel(ylabel="frequency rate")
        plt.show()

    # @override
    def get_features_index_report(self, select_features: List[str]) -> DataFrame:
        """输出变量的 iv、psi 等评价指标

        Args:
            select_features (List[str]): 潜在入模变量或其子集

        Raises:
            ValueError: 传入了非特征字段

        Returns:
            DataFrame: 变量的 iv、psi 等评价指标
        """
        # 计算 PSI
        features_psi: List[float] = []
        features: List[str] = []
        for col in select_features:
            if col not in self.features:
                raise ValueError(f"{col} 不是特征")
            psi: float = PSI(test=self.train[col], base=self.oot[col]) # type: ignore
            features_psi.append(psi)
            features.append(col)
        features_psi_df: DataFrame = pd.DataFrame(data={"feature": features, "psi": features_psi})

        features_train_iv: DataFrame = quality(dataframe=self.train[select_features+[self.target]], target=self.target, iv_only=False)
        features_oot_iv: DataFrame = quality(dataframe=self.oot[select_features+[self.target]], target=self.target, iv_only=False)
        # 区分训练集和测试集的变量名        
        name_train: Dict[str, str] = {col: f"{col}_train" for col in features_train_iv.columns}
        name_oot: Dict[str, str] = {col: f"{col}_oot" for col in features_oot_iv.columns}
        features_train_iv.rename(columns=name_train, inplace=True)
        features_oot_iv.rename(columns=name_oot, inplace=True)
        features_train_iv.reset_index(drop=False, inplace=True, names="feature")
        features_oot_iv.reset_index(drop=False, inplace=True, names="feature")
        features_iv: DataFrame = pd.merge(left=features_train_iv, right=features_oot_iv, how="left", on="feature")
        
        result: DataFrame = pd.merge(left=features_iv, right=features_psi_df, how="left", on="feature")
        return result

    def get_used_features_importance(self) -> DataFrame:
        """输出入模变量的重要性

        Returns:
            DataFrame: 入模变量的重要性
        """
        importance_fscore: Dict[str, float] = self.model.get_fscore() # type: ignore
        feature_importance: DataFrame = pd.DataFrame(data={"feature": importance_fscore.keys()})

        index: List[str] = ["weight", "gain", "cover"]
        for idx in index:
            importance_score: Dict[str, float] = self.model.get_score(importance_type=idx) # type: ignore
            tmp: DataFrame = pd.DataFrame(data={"feature": importance_score.keys(), idx: importance_score.values()})
            feature_importance = feature_importance.merge(right=tmp, how="left", on="feature")
        
        feature_importance.sort_values(by="weight", ascending=False, inplace=True, ignore_index=True) # 按照 weight 排序

        return feature_importance
    
    def plot_used_features_importance(self) -> None:
        """绘制入模特征重要程度的直方图"""
        plot_importance(self.model) # 依据是 self.model.get_fscore() 的结果，也就是 weight
        plt.show()

    def plot_XGB_tree(self, num_trees) -> None:
        """XGB 可视化

        Args:
            num_trees (_type_): 要绘制的是哪棵树

        Raises:
            IndexError: 参数传值错误
        """
        num_boosted_rounds: int = self.model.num_boosted_rounds()
        if num_trees in range(num_boosted_rounds):
            plot_tree(booster=self.model, num_trees=num_trees)
        else:
            raise IndexError(f"num_trees 参数的赋值不在当前 XGB 的树范围 [0, {num_boosted_rounds}) 内")
        
    # @override
    def predict(self, X: DataFrame) -> Series:
        """对新数据进行预测

        Args:
            X (DataFrame): 原始数据

        Returns:
            Series: 模型分
        """
        DX: DMatrix = DMatrix(data=X[self.used_features])
        proba: Series = pd.Series(data=self.model.predict(data=DX))
        model_score: Series = pd.Series(data=[AutomatedModeling.proba2score(prob=x) for x in proba])
        return model_score

    # @override
    def export_model(self, path: str) -> None:
        """将模型以二进制导出为 pkl 格式

        Args:
            path (str): 导出路径
        """
        with open(file=path, mode="wb") as f:
            dump(obj=self.model, file=f)