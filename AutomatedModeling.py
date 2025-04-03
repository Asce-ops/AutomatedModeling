from typing import List, Dict
from datetime import date, timedelta
from random import seed
import warnings
warnings.filterwarnings(action="ignore") # 忽略所有警告

from toad import Combiner, WOETransformer, ScoreCard, quality
from toad.selection import select, stepwise
from toad.plot import bin_plot, badrate_plot
from toad.metrics import KS, KS_bucket, PSI, AUC
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from optbinning import OptimalBinning
from numpy import ndarray
from matplotlib import pyplot as plt
from seaborn import histplot 
plt.rcParams["font.family"] = "SimHei" # 替换为你选择的字体


class AutomatedModeling:
    seed(a=42) # 固定随机种子

    def __init__(self, data: DataFrame, target: str, time: str, not_features: List[str]) -> None:
        """_summary_

        Args:
            data (DataFrame): _description_
            target (str): 字段值为 0-1 变量
            time (str): 字段值为 date 对象
            not_features (List[str]): _description_
        """
        self.data: DataFrame = data.reset_index(drop=True, inplace=False) # 样本
        self.target: str = target # 响应变量
        self.time: str = time # 事件日期
        self.not_features: List[str] = not_features # 除响应变量以外的其他非特征字段
        self.is_train: str = "is_train" # 新增用以区分训练集、验证集和测试集的字段
        self.train: DataFrame
        self.validation: DataFrame
        self.oot: DataFrame
        self.binning: Dict[str, OptimalBinning]
        self.data_woe: DataFrame
        self.train_woe: DataFrame
        self.validation_woe: DataFrame
        self.oot_woe: DataFrame
        self.combiner: Combiner = Combiner()
        self.data_bin: DataFrame
        self.latent_features: List[str] # （经过初筛后）潜在的入模变量
        self.used_features: List[str] # 入模变量
        self.transformer: WOETransformer = WOETransformer()
        self.bins_score: Dict[str, Dict[str, float]] # 特征取值映射为分数
        self.score: str # 模型分字段
        self.evaluation: Dict[str, float]
        self.week: str = "week" # self.time 所在周的周一

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

    def initial_screening(self, iv: float = 0.02, corr: float = 0.7) -> List[str]:
        """原始数据 IV 和相关性初筛

        Args:
            iv (float, optional): iv 最低值. Defaults to 0.02.
            corr (float, optional): 相关系数最高值. Defaults to 0.7.

        Returns:
            List[str]: 初筛后潜在的入模特征
        """
        selected: DataFrame = select(
                                    frame=self.train,
                                    target=self.target,
                                    iv=iv,
                                    corr=corr,
                                    return_drop=False, # 是否返回被删除的变量列
                                    exclude=self.not_features+[self.is_train]
                                ) # type: ignore
        self.latent_features = selected.columns.tolist()
        self.latent_features.remove(self.target)
        self.latent_features.remove(self.is_train)
        for other in self.not_features:
            self.latent_features.remove(other)
        return self.latent_features
    
    def monotonic_trend_binning(self, max_n_bins: int = 5) -> None:
        """将潜在入模变量进行单调分箱和 WOE 变换

        Args:
            max_n_bins (int, optional): 最大化分箱个数. Defaults to 5.
        """
        self.binning = {}
        woe_dict: Dict[str, ndarray] = {}

        for col in self.latent_features:
            optb: OptimalBinning = OptimalBinning(
                                                    name=col, 
                                                    dtype="numerical", 
                                                    solver="cp", 
                                                    max_n_bins=max_n_bins, 
                                                    monotonic_trend="auto_asc_desc" # 单调分箱
                                                ) # 实例化分箱器对象
            optb.fit(x=self.train[col], y=self.train[self.target])  # 训练分箱器
            data_transform: ndarray = optb.transform(
                                                x=self.data[col], 
                                                metric="woe" # ["woe" | "indices" | "bins" | "event_rate"]
                                            )  # WOE 转换
            self.binning[col] = optb
            woe_dict[col] = data_transform * -1 # 纠正 optbinning 库中 WOE 值的正负号取反

        self.data_woe = pd.DataFrame(data=woe_dict)
        self.data_woe[self.target] = self.data[self.target]
        self.data_woe[self.is_train] = self.data[self.is_train]
        for other in self.not_features:
            self.data_woe[other] = self.data[other]

        self.train_woe = self.data_woe[self.data_woe[self.is_train] == 1]
        self.validation_woe = self.data_woe[self.data_woe[self.is_train] == 2]
        self.oot_woe = self.data_woe[self.data_woe[self.is_train] == 0]

        # 分箱规则生成指定的分箱器
        binning_rules: Dict[str, List[float]] = {}
        for col in self.latent_features:
            optb: OptimalBinning = self.binning[col]
            binning_rules[col] = optb.splits.tolist() # type: ignore

        self.combiner.set_rules(map=binning_rules, reset=True)

        self.data_bin = self.combiner.transform(X=self.data[self.latent_features+self.not_features+[self.target, self.is_train]], labels=True) # type: ignore

    def iv_screening_after_woe(self, selected_features: List[str], iv: float = 0.02) -> List[str]:
        """剔除离散化后 iv 值较低的变量

        Args:
            selected_features (List[str]): 潜在的入模变量
            iv (float, optional): iv 阈值. Defaults to 0.02.

        Returns:
            List[str]: iv 值超过指定阈值的变量
        """
        distinction: DataFrame = quality(dataframe=self.train_woe[selected_features+[self.target]], target=self.target, iv_only=True)
        result: List[str] = distinction[distinction["iv"] >= iv].index.to_list()
        return result

    def eliminate_unstable_features(self, selected_features: List[str], psi_threshold: float = 0.1) -> List[str]:
        """剔除分箱后稳定性较差的变量

        Args:
            selected_features (List[str]): 潜在的入模变量
            psi_threshold (float, optional): psi筛选的阈值. Defaults to 0.1.

        Returns:
            List[str]: 稳定性较好的变量
        """
        selected_features_bin_psi: Series = PSI(test=self.train_woe[selected_features], base=self.oot_woe[selected_features]) # type: ignore
        stable_features: List[str] = selected_features_bin_psi[selected_features_bin_psi < psi_threshold].index.to_list()
        return stable_features
    
    def stepwise_after_WOETransformer(self, selected_features: List[str], estimator: str = "ols", direction: str = "both", criterion: str = "aic") -> List[str]:
        """对 WOE 变换后的数据逐步回归，确定最终的入模变量

        Args:
            selected_features (List[str]): 潜在的入模变量
            estimator (str, optional): 用于拟合的模型，["ols" | "lr" | "lasso" | "ridge"]. Defaults to "ols".
            direction (str, optional): 逐步回归的方向，["forward" | "backward" | "both"]. Defaults to "both".
            criterion (str, optional): 评判标准，["aic" | "bic" | "ks" | "auc"]. Defaults to "aic".

        Returns:
            List[str]: 潜在的入模变量
        """
        selected: DataFrame = stepwise(
                                    frame=self.train_woe[selected_features+[self.target]],
                                    target=self.target,
                                    estimator=estimator,
                                    direction=direction,
                                    criterion=criterion,
                                    return_drop=False
                                ) # type: ignore
        self.used_features = selected.columns.tolist()

        self.used_features.remove(self.target)
        return self.used_features
    
    def fit(self, used_features: List[str], model_score: str = "model_score") -> None:
        """拟合评分卡

        Args:
            used_features (List[str]): 入模变量
            model_score (str, optional): 模型分字段命名. Defaults to "model_score".
        """
        self.transformer.fit(X=self.data_bin[self.data_bin[self.is_train]==1], y=self.data_bin[self.data_bin[self.is_train]==1][self.target], exclude=self.not_features+[self.target])

        scoreCard: ScoreCard = ScoreCard(
                                    combiner=self.combiner,
                                    transer=self.transformer,
                                    base_score=600,
                                    base_odds=1.22, # type: ignore
                                    pdo=20,
                                    rate=2
                                )
        scoreCard.fit(X=self.train_woe[used_features], y=self.train_woe[self.target])
        self.bins_score = scoreCard.export()

        self.score = model_score
        self.data_woe[self.score] = scoreCard.predict(X=self.data) # 传入原始数据而非 WOE 变换后的数据

    def calculate_model_score_psi(self, n_bins: int = 50) -> float:
        """计算模型分离散化后的 PSI

        Args:
            n_bins (int, optional): 计算模型分 PSI 时的分箱个数. Defaults to 50.

        Returns:
            float: 模型分离散化后在训练集和测试集上的 PSI
        """
        tmp: DataFrame = self.data_woe[[self.is_train, self.score]]
        tmp[f"{self.score}_bin"] = pd.qcut(x=tmp[self.score], q=n_bins, duplicates="drop")
        result: float = PSI(test=tmp[tmp[self.is_train]==0][f"{self.score}_bin"], base=tmp[tmp[self.is_train]==1][f"{self.score}_bin"]) # type: ignore
        return result

    def evaluate(self, n_bins: int = 50) -> Dict[str, float]:
        """模型评估

        Args:
            n_bins (int, optional): 计算模型分 PSI 时的分箱个数. Defaults to 50.

        Returns:
            Dict[str, float]:  模型的量化指标（KS, AUC, model_score_psi）
        """
        self.evaluation = {}
        train_ks: float = KS(score=self.data_woe[self.data_woe[self.is_train]==1][self.score], target=self.data_woe[self.data_woe[self.is_train]==1][self.target])
        validation_ks: float = KS(score=self.data_woe[self.data_woe[self.is_train]==2][self.score], target=self.data_woe[self.data_woe[self.is_train]==2][self.target])
        oot_ks: float = KS(score=self.data_woe[self.data_woe[self.is_train]==0][self.score], target=self.data_woe[self.data_woe[self.is_train]==0][self.target])
        train_auc: float = AUC(score=self.data_woe[self.data_woe[self.is_train]==1][self.score], target=self.data_woe[self.data_woe[self.is_train]==1][self.target]) # type: ignore
        validation_auc: float = AUC(score=self.data_woe[self.data_woe[self.is_train]==2][self.score], target=self.data_woe[self.data_woe[self.is_train]==2][self.target]) # type: ignore
        oot_auc: float = AUC(score=self.data_woe[self.data_woe[self.is_train]==0][self.score], target=self.data_woe[self.data_woe[self.is_train]==0][self.target]) # type: ignore
        self.evaluation["train_ks"] = train_ks
        self.evaluation["validation_ks"] = validation_ks
        self.evaluation["oot_ks"] = oot_ks
        self.evaluation["train_auc"] = train_auc
        self.evaluation["validation_auc"] = validation_auc
        self.evaluation["oot_auc"] = oot_auc
        self.evaluation["model_psi"] = self.calculate_model_score_psi(n_bins=n_bins)
        return self.evaluation
    
    def ks_bucket(self, train_validation_oot: int = 0, n_bins: int = 10) -> DataFrame:
        """模型分分箱

        Args:
            train_validation_oot (int, optional): {1: "训练集", 0: "测试集", 2: "验证集"}. Defaults to 0.
            n_bins (int, optional): 分箱数量. Defaults to 10.

        Returns:
            DataFrame: 模型分箱后在各个箱中的效果
        """
        result: DataFrame = KS_bucket(
                                score=self.data_woe[self.data_woe[self.is_train]==train_validation_oot][self.score], # 分箱变量
                                target=self.data_woe[self.data_woe[self.is_train]==train_validation_oot][self.target], # 目标变量
                                bucket=n_bins, # 分箱个数，默认为10
                                method="quantile" # 默认等频分箱，{'quantile':等频分箱, 'step':等距分箱}
                            ) # type: ignore
        return result
    
    def plot_model_score_distribution(self) -> None:
        """绘制训练集、验证集和测试集上的模型分分布图"""
        plt.figure(figsize=(20, 7))
        histplot(
            x=self.data_woe[self.data_woe[self.is_train]==1][self.score], 
            kde=True, # 是否绘制核密度图
            color="red",
            bins=50,
            label="训练集上的模型分分布",
            stat="density" # 统计方法，频率还是频数
        )
        histplot(
            x=self.data_woe[self.data_woe[self.is_train]==0][self.score], 
            kde=True, 
            color="yellow",
            bins=50,
            label="测试集上的模型分分布",
            stat="density"
        )
        histplot(
            x=self.data_woe[self.data_woe[self.is_train]==2][self.score], 
            kde=True, 
            color="green",
            bins=50,
            label="验证集上的模型分分布",
            stat="density"
        )
        plt.legend()
        plt.title(label="训练集、验证集和测试集上模型分分布对比")
        plt.xlabel(xlabel="模型分")
        plt.ylabel(ylabel="频率")
        plt.show()

    def monotonic_trend_bin_plot(self, selected_features: List[str], train_validation_oot: int = 1) -> None:
        """潜在入模变量（或其子集）单调分箱结果可视化

        Args:
            selected_features (List[str]): 潜在入模变量（或其子集）
            train_validation_oot (int, optional):  {1: "训练集", 0: "测试集", 2: "验证集"}. Defaults to 1.

        Raises:
            ValueError: 参数传值错误
            ValueError: 未对特征进行分箱
        """
        if train_validation_oot == 0:
            label: str = "validation"
        elif train_validation_oot == 1:
            label: str = "train"
        elif train_validation_oot == 2:
            label: str = "oot"
        else:
            raise ValueError(f"参数 {train_validation_oot} 只能传入 0 或 1 或 2")

        for col in selected_features:
            if col in self.latent_features:
                bin_plot(frame=self.data_bin[self.data_bin[self.is_train]==train_validation_oot], x=col, target=self.target)
                plt.title(label=label)
            else:
                raise ValueError(f"{col} 不是潜在的入模变量，未对其进行分箱")

    @staticmethod
    def get_monday(Date: date) -> str:
        """返回所在周的周一

        Args:
            Date (date): 事件日期

        Returns:
            str: "yyyy-mm-dd"
        """
        monday: date = Date - timedelta(days=Date.weekday())
        return monday.strftime("%Y-%m-%d")

    def plot_bad_rate_trend(self, selected_features: List[str], all_train_validation_oot: int = -1) -> None:
        """绘制变量对应的 bad_rate 随时间变化的趋势图

        Args:
            selected_features (List[str]): 潜在入模变量或其子集
            all_train_validation_oot (int, optional): {-1: "所有样本", 1: "训练集", 2: "验证集", 0: "测试集"}. Defaults to -1.

        Raises:
            ValueError: 参数传值错误
            ValueError: 未对特征进行分箱
        """
        self.data_bin[self.week] = self.data_bin[self.time].apply(func=self.get_monday) # 为 self.data_bin 添加表示事件时间周的字段

        if all_train_validation_oot == -1:
            data_bin: DataFrame = self.data_bin
            label: str = "all sample"
        else:
            if all_train_validation_oot == 0:
                label: str = "oot"
            elif all_train_validation_oot == 1:
                label: str = "train"
            elif all_train_validation_oot == 2:
                label: str = "validation"
            else:
                raise ValueError(f"参数 {all_train_validation_oot} 只能传入 -1 或 0 或 1 或 2")
            data_bin: DataFrame = self.data_bin[self.data_bin[self.is_train]==all_train_validation_oot]
        
        for feature in selected_features:
            if feature in self.latent_features:
                badrate_plot(frame=data_bin, x=self.week, target=self.target, by=feature)
                plt.title(label=f"{feature} in {label}")
            else:
                raise ValueError(f"{feature} 不是潜在的入模变量，未对其进行分箱")

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
        used_features: List[str] = self.stepwise_after_WOETransformer(selected_features=selected_features, estimator=estimator, direction=direction, criterion=criterion) # 逐步回顾确定最终的入模变量
        print(f"经逐步回归确定入模变量共 {len(used_features)} 个，分别是 {used_features}")
        self.fit(used_features=used_features, model_score=model_score) # 拟合评分卡
        evaluation: Dict[str, float] = self.evaluate(n_bins=n_bins) # 模型评价指标
        print(f"训练集上 KS 值为：{evaluation['train_ks']}，AUC 值为：{evaluation['train_auc']}；验证集上 KS 值为：{evaluation['validation_ks']}，AUC 值为 {evaluation['validation_auc']}；测试集上 KS 值为：{evaluation['oot_ks']}，AUC 值为 {evaluation['oot_auc']}；模型分的 PSI 为 {evaluation['model_psi']}")

    def get_feature_bin_score(self) -> Dict[str, Dict[str, float]]:
        """获取入模特征各个区间段到评分的映射

        Returns:
            Dict[str, Dict[str, float]]: 入模特征各个区间段到评分的映射
        """
        return self.bins_score
    
    def export_model(self) -> None:
        """将模型导出为 PMML 格式"""
        pass