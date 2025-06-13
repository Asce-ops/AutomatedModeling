from typing import List, Dict, Any, Optional

from ruamel.yaml import YAML
from pandas import DataFrame


class MLModel:
    """用于导出将模型部署至 model-service 的 yaml 文件"""
    def __init__(self, model_file: str, dataframe: DataFrame, model_type: str, model_name: str, yname: str = "target", serializer: str = "origin", objective: str = "binary") -> None:
        self.yaml: YAML = YAML()
        self.model_file: str = model_file
        self.dataframe: DataFrame = dataframe
        self.feature_list: List[str] = list(dataframe.columns)
        self.yname: str = yname
        self.model_name: str = model_name
        self.feature_types: Dict[str, Any] = {}
        self.model_yaml: Dict[str, Any] = {
            "model_type": model_type,
            "model_path": model_file,
            "serializer": serializer,
            "model_name": model_name,
            "signature": {},
            "objective": objective,
            "is_score_card": False
        }
        self._infer_types()
        self._build_signature()

    def _build_signature(self) -> None:
        """构建模型签名"""
        inputs_list: List[Dict[str, Any]] = [{"name": col, "type": self.feature_types.get(col, "float64")} for col in self.feature_list]
        self.model_yaml["signature"]["inputs"] = inputs_list
        self.model_yaml["signature"]["outputs"] = [{"type": "float64", "name": self.yname}]

    def _infer_types(self) -> None:
        """从 DataFrame 对象中推断特征数据类型"""
        for col in self.feature_list:
            pandas_type = str(object=self.dataframe[col].dtype)
            self.feature_types[col] = pandas_type

    def save(self, name="MLmodel") -> None:
        """保存模型配置文件"""
        with open(file=name, mode="w", encoding="utf8") as t:
            self.yaml.dump(data=self.model_yaml, stream=t)

    def update_score_card_info(self, pdo: Optional[float] = None, rate: Optional[float] = None, base_score: Optional[float] = None, base_odds: Optional[float] = None) -> None:
        """更新评分卡配置信息"""
        if not self.model_yaml.get("is_score_card", False):
            self.model_yaml["is_score_card"] = True
            self.model_yaml["score_card_info"] = {
                "pdo": 30,
                "rate": 2.0,
                "base_score": 500,
                "base_odds": 1.0
            }
        if pdo is not None:
            self.model_yaml["score_card_info"]["pdo"] = pdo
        if rate is not None:
            self.model_yaml["score_card_info"]["rate"] = rate
        if base_score is not None:
            self.model_yaml["score_card_info"]["base_score"] = base_score
        if base_odds is not None:
            self.model_yaml["score_card_info"]["base_odds"] = base_odds
