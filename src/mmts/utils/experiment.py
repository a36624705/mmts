# src/mmts/utils/experiment.py
# -*- coding: utf-8 -*-
"""
实验管理器：统一管理实验ID、配置、输出路径
- 替代Hydra的自动输出目录管理
- 提供清晰的实验组织和版本控制
"""

from __future__ import annotations

import json
import datetime as _dt
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Union

from omegaconf import DictConfig, OmegaConf


@dataclass
class ExperimentInfo:
    """实验信息记录"""
    experiment_id: str
    name: str
    description: Optional[str] = None
    created_at: str = ""
    config_hash: str = ""
    status: str = "running"  # running, completed, failed
    metrics: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = _dt.datetime.now().isoformat()


@dataclass
class ExperimentPaths:
    """实验输出路径管理"""
    root: Path
    models: Path  # 统一的模型保存目录（lora+processor）
    logs: Path    # 日志文件
    figures: Path # 图片输出
    checkpoints: Path  # 训练checkpoints（仅权重）
    config: Path  # 实验配置备份
    
    def ensure_all(self) -> None:
        """确保所有目录存在"""
        for path in [self.root, self.models, self.logs, self.figures, self.checkpoints, self.config]:
            path.mkdir(parents=True, exist_ok=True)


class ExperimentManager:
    """实验管理器"""
    
    def __init__(self, base_dir: Union[str, Path] = "experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def create_experiment(
        self,
        name: str,
        config: DictConfig,
        description: Optional[str] = None,
        experiment_id: Optional[str] = None
    ) -> tuple[ExperimentInfo, ExperimentPaths]:
        """创建新实验"""
        
        # 生成实验ID
        if experiment_id is None:
            timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_id = f"{name}_{timestamp}"
        
        # 计算配置哈希
        config_str = OmegaConf.to_yaml(config, resolve=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        # 创建实验信息
        exp_info = ExperimentInfo(
            experiment_id=experiment_id,
            name=name,
            description=description,
            config_hash=config_hash
        )
        
        # 创建路径结构
        exp_root = self.base_dir / experiment_id
        paths = ExperimentPaths(
            root=exp_root,
            models=exp_root / "models",
            logs=exp_root / "logs", 
            figures=exp_root / "figures",
            checkpoints=exp_root / "checkpoints",
            config=exp_root / "config"
        )
        paths.ensure_all()
        
        # 保存实验信息和配置
        self._save_experiment_info(exp_info, paths)
        self._save_config(config, paths)
        
        return exp_info, paths
    
    def load_experiment(self, experiment_id: str) -> tuple[ExperimentInfo, ExperimentPaths]:
        """加载现有实验"""
        exp_root = self.base_dir / experiment_id
        if not exp_root.exists():
            raise FileNotFoundError(f"实验不存在: {experiment_id}")
            
        # 加载实验信息
        info_path = exp_root / "experiment_info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"实验信息文件不存在: {info_path}")
            
        with open(info_path, 'r', encoding='utf-8') as f:
            info_data = json.load(f)
        exp_info = ExperimentInfo(**info_data)
        
        # 创建路径对象
        paths = ExperimentPaths(
            root=exp_root,
            models=exp_root / "models",
            logs=exp_root / "logs",
            figures=exp_root / "figures", 
            checkpoints=exp_root / "checkpoints",
            config=exp_root / "config"
        )
        
        return exp_info, paths
    
    def list_experiments(self) -> list[ExperimentInfo]:
        """列出所有实验"""
        experiments = []
        for exp_dir in self.base_dir.iterdir():
            if exp_dir.is_dir():
                info_path = exp_dir / "experiment_info.json"
                if info_path.exists():
                    try:
                        with open(info_path, 'r', encoding='utf-8') as f:
                            info_data = json.load(f)
                        experiments.append(ExperimentInfo(**info_data))
                    except Exception:
                        continue
        return sorted(experiments, key=lambda x: x.created_at, reverse=True)
    
    def get_latest_experiment(self) -> Optional[tuple[ExperimentInfo, ExperimentPaths]]:
        """获取最新的实验"""
        experiments = self.list_experiments()
        if not experiments:
            return None
        return self.load_experiment(experiments[0].experiment_id)
    
    def update_experiment_status(
        self, 
        experiment_id: str, 
        status: str, 
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """更新实验状态"""
        exp_info, paths = self.load_experiment(experiment_id)
        exp_info.status = status
        if metrics is not None:
            exp_info.metrics = metrics
        self._save_experiment_info(exp_info, paths)
    
    def _save_experiment_info(self, exp_info: ExperimentInfo, paths: ExperimentPaths) -> None:
        """保存实验信息"""
        info_path = paths.root / "experiment_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(exp_info), f, ensure_ascii=False, indent=2)
    
    def _save_config(self, config: DictConfig, paths: ExperimentPaths) -> None:
        """保存实验配置"""
        config_path = paths.config / "config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            OmegaConf.save(config, f)


def get_experiment_manager(base_dir: Union[str, Path] = "experiments") -> ExperimentManager:
    """获取实验管理器实例"""
    return ExperimentManager(base_dir)
