from typing import List, Dict
from abc import ABC, abstractmethod
from datetime import date
from random import seed

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
        self.is_train: str = "is_train" # 新增用以区分训练集、验证集和测试集的字段
        self.train: DataFrame
        self.validation: DataFrame
        self.oot: DataFrame
        self.evaluation: Dict[str, float]

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
