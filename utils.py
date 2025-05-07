from typing import Tuple, Optional

from pandas.core.frame import DataFrame
from pandas.core.series import Series
import pandas as pd


def swap_in_out(data: DataFrame, is_mature: str, target: str, new_model_score: str, 
                old_risk_level: str = "risk_level", new_risk_level: str = "new_risk_level", 
                primary_key: str = "sn", 
                accept: str = "ACCEPT", review: str = "REVIEW", reject: str = "REJECT", 
                new_accept_rate: Optional[float] = None, new_reject_rate: Optional[float] = None
                ) -> Tuple[DataFrame, DataFrame]:
    """计算换入换出

    Args:
        data (DataFrame): 进件订单数据
        is_mature (str): 0-1 变量，放款订单是否成熟
        target (str): 0-1 变量，成熟订单是否坏账
        new_model_score (str): 新模型分字段
        old_risk_level (str, optional): 当前风险水平字段. Defaults to "risk_level".
        new_risk_level (str, optional): 新模型分的风险水平字段（会被新增的字段）. Defaults to "new_risk_level".
        primary_key (str, optional): 用以惟一确定一笔订单. Defaults to "sn".
        accept (str, optional): 模型通过. Defaults to "ACCEPT".
        review (str, optional): 走人审. Defaults to "REVIEW".
        reject (str, optional): 模型拒绝. Defaults to "REJECT".
        new_accept_rate (Optional[float], optional): 规定的新模型通过占比，为空则表示保持通过占比不变. Defaults to None.
        new_reject_rate (Optional[float], optional): 规定的新模型拒绝占比，为空则表示保持拒绝占比不变. Defaults to None.

    Returns:
        Tuple[DataFrame, DataFrame]: (进件订单的换入换出, 换入换出的风险情况)
    """
    num: int = data.shape[0]
    old_accept_rate: float = data[data[old_risk_level]==accept].shape[0] / data.shape[0]
    old_reject_rate: float = data[data[old_risk_level]==reject].shape[0] / data.shape[0]
    print(f"当前模型通过的占比为：{old_accept_rate:.2%}，拒绝的占比为：{old_reject_rate:.2%}")
    if new_accept_rate is None:
        new_accept_rate = old_accept_rate
        print("保持模型通过的比例不变")
    if new_reject_rate is None:
        new_reject_rate = old_reject_rate
        print("保持模型拒绝的比例不变")
    new_model_accept_score: float = data[new_model_score].quantile(q=1-new_accept_rate)
    new_model_reject_score: float = data[new_model_score].quantile(q=new_reject_rate)
    print(f"新模型通过比例定为：{new_accept_rate:.2%}，新模型通过的分数为：{new_model_accept_score:.2f}；新模型拒绝比例定为：{new_reject_rate:.2%}，新模型拒绝的分数为：{new_model_reject_score:.2f}")

    def tag_risk_level(model_score: float, model_accept_score: float, model_reject_score: float) -> str:
        if model_score <= model_reject_score:
            return reject
        elif model_score >= model_accept_score:
            return accept
        else:
            return review
    
    data[new_risk_level] = data[new_model_score].apply(func=lambda x: tag_risk_level(model_score=x, model_accept_score=new_model_accept_score, model_reject_score=new_model_reject_score))

    def verify_pct(series: Series) -> float:
        """用于计算交叉部分占总体的比例"""
        return len(series) / num
    
    def bad_rate(series: Series) -> float:
        """用于计算交叉部分的风险情况"""
        return series.mean()
    
    verify_swap: DataFrame = pd.pivot_table(
        data=data,
        columns=[old_risk_level],
        index=[new_risk_level],
        values=[primary_key],
        aggfunc={primary_key: ["count", verify_pct]} # type: ignore
    )
    
    risk_swap: DataFrame = pd.pivot_table(
        data=data[data[is_mature] == 1], # 成熟订单
        columns=[old_risk_level],
        index=[new_risk_level],
        values=[primary_key, target],
        aggfunc={primary_key: ["count"], target: [bad_rate]} # type: ignore
    )

    return verify_swap, risk_swap
