"""
信息粒构建 — Information Granules for Sleep Stage

核心概念：
1. 每个 epoch (30秒) → 一个信息粒
2. 信息粒 = (特征向量 → 模糊集隶属度, 阶段标签)
3. 粒关系：相邻epoch的粒可以合并为"粗粒"（过渡期检测）

粒构建流程：
  原始特征 → 模糊化 (fuzzify) → 信息粒 → 粒关系计算 → 粒合并(过渡期检测)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from .fuzzy_sets import fuzzify, FEATURE_FUZZY_MAP


@dataclass
class InfoGranule:
    """
    信息粒: 一个epoch的模糊化表示
    
    属性:
        epoch_idx: epoch索引
        features: 原始特征值 {特征名: 值}
        fuzzy_repr: 模糊表示 {特征名: [(标签, 隶属度), ...]}
        stage: 真实阶段标签 (W/N1/N2/N3/REM)
        stage_num: 数字标签 (0=W,1=N1,2=N2,3=N3,4=REM)
    """
    epoch_idx: int
    features: Dict[str, float]
    fuzzy_repr: Dict[str, List[Tuple[str, float]]]
    stage: str
    stage_num: int = -1
    
    def __post_init__(self):
        stage_map = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4, '?': -1}
        self.stage_num = stage_map.get(self.stage, -1)


@dataclass
class GranuleRelation:
    """
    粒间关系: 两个相邻信息粒的关系
    
    属性:
        i, j: 相邻粒的索引
        similarity: 模糊相似度 [0, 1]
        inclusion: 包含度 (i被j包含的程度)
        overlap: 重叠度
        is_transition: 是否过渡期 (阶段不同)
    """
    i: int
    j: int
    similarity: float
    inclusion: float
    overlap: float
    is_transition: bool


class GranuleBuilder:
    """
    信息粒构建器
    
    从特征矩阵构建信息粒序列
    """
    
    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or list(FEATURE_FUZZY_MAP.keys())
    
    def build_granules(self, feature_matrix: np.ndarray, 
                       stages: List[str],
                       feature_names: Optional[List[str]] = None) -> List[InfoGranule]:
        """
        从特征矩阵构建信息粒序列
        
        参数:
            feature_matrix: (n_epochs, n_features) 特征矩阵
            stages: 每个epoch的阶段标签列表
            feature_names: 特征名列表（与feature_matrix列对应）
            
        返回:
            信息粒列表
        """
        fn = feature_names or self.feature_names
        n_epochs = feature_matrix.shape[0]
        
        granules = []
        for i in range(n_epochs):
            feat_dict = {fn[j]: float(feature_matrix[i, j]) for j in range(len(fn))}
            
            # 模糊化每个特征
            fuzzy_repr = {}
            for fname, val in feat_dict.items():
                if fname in FEATURE_FUZZY_MAP:
                    fuzzy_repr[fname] = fuzzify(fname, val)
            
            stage = stages[i] if i < len(stages) else '?'
            g = InfoGranule(
                epoch_idx=i,
                features=feat_dict,
                fuzzy_repr=fuzzy_repr,
                stage=stage
            )
            granules.append(g)
        
        return granules
    
    def compute_relations(self, granules: List[InfoGranule]) -> List[GranuleRelation]:
        """
        计算相邻粒间的关系
        
        返回:
            粒关系列表
        """
        relations = []
        for i in range(len(granules) - 1):
            g1, g2 = granules[i], granules[i + 1]
            
            # 计算模糊相似度 (所有特征的隶属度向量间余弦相似度)
            vec1 = self._granule_to_vector(g1)
            vec2 = self._granule_to_vector(g2)
            
            sim = self._cosine_similarity(vec1, vec2)
            
            # 包含度 (min-max 方法)
            inc = self._inclusion_degree(vec1, vec2)
            
            # 重叠度
            ovl = self._overlap_degree(vec1, vec2)
            
            rel = GranuleRelation(
                i=i, j=i+1,
                similarity=sim,
                inclusion=inc,
                overlap=ovl,
                is_transition=(g1.stage != g2.stage)
            )
            relations.append(rel)
        
        return relations
    
    def detect_transition_regions(self, granules: List[InfoGranule],
                                   relations: List[GranuleRelation],
                                   sim_threshold: float = 0.6) -> List[Tuple[int, int]]:
        """
        检测过渡期区域: 连续低相似度的epoch序列
        
        返回:
            [(start_idx, end_idx), ...] 过渡区域列表
        """
        transitions = []
        in_transition = False
        start = 0
        
        for i, rel in enumerate(relations):
            if rel.is_transition or rel.similarity < sim_threshold:
                if not in_transition:
                    start = i
                    in_transition = True
            else:
                if in_transition:
                    transitions.append((start, i))
                    in_transition = False
        
        if in_transition:
            transitions.append((start, len(relations)))
        
        return transitions
    
    def _granule_to_vector(self, g: InfoGranule) -> np.ndarray:
        """将信息粒展开为模糊隶属度向量"""
        vec = []
        for fname in self.feature_names:
            if fname in g.fuzzy_repr:
                for label, mu in g.fuzzy_repr[fname]:
                    vec.append(mu)
            else:
                vec.append(0.0)
        return np.array(vec)
    
    @staticmethod
    def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        if np.linalg.norm(v1) < 1e-10 or np.linalg.norm(v2) < 1e-10:
            return 0.0
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    
    @staticmethod
    def _inclusion_degree(v1: np.ndarray, v2: np.ndarray) -> float:
        """v1 被 v2 包含的程度"""
        inc = np.minimum(v1, v2).sum() / (v1.sum() + 1e-10)
        return float(inc)
    
    @staticmethod
    def _overlap_degree(v1: np.ndarray, v2: np.ndarray) -> float:
        """两个向量的重叠度"""
        overlap = np.minimum(v1, v2).sum()
        max_sum = max(v1.sum(), v2.sum()) + 1e-10
        return float(overlap / max_sum)
