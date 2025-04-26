from typing import List, Dict
from abc import ABC, abstractmethod
from datetime import date
from random import seed
from math import log

import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.series import Series


class AutomatedModeling(ABC):
    seed(a=42) # 固定随机种子

    def __init__(self, data: DataFrame, target: str, time: str, not_features: List[str]) -> None:
        """传入规定格式的数据以实例化 AutomatedModeling 对象

        Args:
            data (DataFrame): 样本数据
            target (str): 字段值为 0-1 变量
            time (str): 字段值为 date 对象
            not_features (List[str]): 除相应变量外的其他非特征字段（如时间、订单号、包名等）
        """
        self.data: DataFrame = data.reset_index(drop=True, inplace=False) # 样本
        self.target: str = target # 响应变量
        self.time: str = time # 事件日期
        self.not_features: List[str] = not_features # 除响应变量以外的其他非特征字段
        self.features: List[str] = [col for col in self.data.columns if col not in self.not_features + [self.target]] # 除 target 和 not_features 以外的所有特征字段
        self.is_train: str = "is_train" # 新增用以区分训练集、验证集和测试集的字段
        self.train: DataFrame
        self.validation: DataFrame
        self.oot: DataFrame
        self.used_features: List[str] # 最终入模特征
        self.evaluation: Dict[str, float]
        self.score: str # 模型分字段

    def split_train_oot(self, split_time: date, split_validation_set: bool = False, validation_pct: float = 0.2) -> None:
        """划分训练集和测试集（如不划分验证集则默认将测试集复制一份作为验证集）

        Args:
            split_time (date): 用于划分训练集和测试集的临界日期
            split_validation_set (bool, optional): 是否需要划分验证集. Defaults to False.
            validation_pct (float, optional): 验证集占训练集的样本. Defaults to 0.2.
        """
        self.data[self.is_train] = -1 # 在原始数据集中新增一个用于区分训练集、验证集和测试集的字段`is_train`: {1: "训练集", 0: "测试集", 2: "验证集"}
        
        self.train = self.data[self.data[self.time] <= split_time]
        self.oot = self.data[self.data[self.time] > split_time]
        
        if split_validation_set:
            self.validation = self.train.sample(frac=validation_pct, random_state=42)
            self.train.drop(index=self.validation.index, axis=0, inplace=True) # 从训练集中剔除掉验证集样本
        else:
            self.validation = self.oot.copy()
        
        self.train[self.is_train] = 1
        self.validation[self.is_train] = 2
        self.oot[self.is_train] = 0
        self.data = pd.concat(objs=[self.train, self.validation, self.oot], axis=0)

        self.train.reset_index(drop=True, inplace=True)
        self.validation.reset_index(drop=True, inplace=True)
        self.oot.reset_index(drop=True, inplace=True)
        self.data.reset_index(drop=True, inplace=True)

    @abstractmethod
    def eliminate_low_iv_features(self, selected_features: List[str], iv: float = 0.02, train_validation_oot: int = 1) -> List[str]:
        """剔除 iv 值较低的变量

        Args:
            selected_features (List[str]): 潜在的入模变量
            iv (float, optional): iv 阈值. Defaults to 0.02.
            train_validation_oot (int, optional): {1: "训练集", 0: "测试集", 2: "验证集"}. Defaults to 1.

        Returns:
            List[str]: iv 值超过指定阈值的变量
        """
        pass

    @abstractmethod
    def eliminate_unstable_features(self, selected_features: List[str], psi_threshold: float = 0.1, psi_original: bool = True) -> List[str]:
        """剔除稳定性较差的变量

        Args:
            selected_features (List[str]): 潜在的入模变量
            psi_threshold (float, optional): psi筛选的阈值. Defaults to 0.1.
            psi_original (bool, optional): 使用原始数据还是特征工程后的数据（True: 原始数据, False: 特征工程后的数据）. Defaults to True.

        Returns:
            List[str]: 稳定性较好的变量
        """
        pass

    @abstractmethod
    def calculate_model_score_psi(self, n_bins: int) -> float:
        """计算模型分离散化后的 PSI

        Args:
            n_bins (int, optional): 计算模型分 PSI 时的分箱个数

        Returns:
            float: 模型分离散化后在训练集和测试集上的 PSI
        """
        pass

    @abstractmethod
    def evaluate(self, n_bins: int) -> Dict[str, float]:
        """模型评估

        Args:
            n_bins (int, optional): 计算模型分 PSI 时的分箱个数

        Returns:
            Dict[str, float]:  模型的量化指标（KS, AUC, model_score_psi）
        """
        pass

    @abstractmethod
    def ks_bucket(self, train_validation_oot: int, n_bins: int) -> DataFrame:
        """模型分分箱

        Args:
            train_validation_oot (int, optional): {1: "训练集", 0: "测试集", 2: "验证集"}
            n_bins (int, optional): 分箱数量

        Raises:
            ValueError: 参数传值错误

        Returns:
            DataFrame: 模型分箱后在各个箱中的效果
        """
        pass
    
    @abstractmethod
    def predict(self, X: DataFrame) -> Series:
        """对新数据进行预测

        Args:
            X (DataFrame): 原始数据

        Returns:
            Series: 模型分
        """
        pass
    
    @staticmethod
    def proba2score(prob: float, pdo: int = 20, rate: float = 2, base_odds: float = 1.22, base_score: int = 600) -> float:
        """将概率转化为分数

        Args:
            prob (float): 响应概率
            pdo (int, optional): pdo. Defaults to 20.
            rate (float, optional): rate. Defaults to 2.
            base_odds (float, optional): base_odds. Defaults to 1.22.
            base_score (int, optional): base_score. Defaults to 600.

        Returns:
            float: 响应分数
        """
        factor: float = pdo / log(rate)
        offset: float = base_score - factor * log(base_odds)
        return offset - factor * log(prob / (1-prob))
    
    @abstractmethod
    def get_features_index_report(self, select_features: List[str]) -> DataFrame:
        """输出变量的 iv、psi 等评价指标

        Args:
            select_features (List[str]): 特征列表

        Returns:
            DataFrame: 变量的 iv、psi 等评价指标
        """
        pass
