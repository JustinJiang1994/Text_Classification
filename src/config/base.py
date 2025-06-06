from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json
import yaml
import os
from pathlib import Path
from ..utils.logger import get_logger

logger = get_logger(__name__)

class BaseConfig(ABC):
    """配置管理基类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config: Dict[str, Any] = {}
        self.config_path = config_path
        
        if config_path:
            self.load_config(config_path)
    
    @abstractmethod
    def validate_config(self) -> bool:
        """验证配置是否有效
        
        Returns:
            配置是否有效
        """
        pass
    
    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置
        
        Returns:
            默认配置字典
        """
        pass
    
    def load_config(self, config_path: str):
        """加载配置文件
        
        Args:
            config_path: 配置文件路径
        """
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"配置文件不存在: {config_path}，将使用默认配置")
            self.config = self.get_default_config()
            return
        
        # 根据文件扩展名选择加载方式
        if config_path.suffix == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        elif config_path.suffix in ['.yml', '.yaml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
        
        # 验证配置
        if not self.validate_config():
            logger.warning("配置验证失败，将使用默认配置")
            self.config = self.get_default_config()
        else:
            logger.info(f"成功加载配置文件: {config_path}")
    
    def save_config(self, config_path: Optional[str] = None):
        """保存配置到文件
        
        Args:
            config_path: 配置文件路径，如果为None则使用初始化时的路径
        """
        if config_path is None:
            if self.config_path is None:
                raise ValueError("未指定配置文件路径")
            config_path = self.config_path
        
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 根据文件扩展名选择保存方式
        if config_path.suffix == '.json':
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        elif config_path.suffix in ['.yml', '.yaml']:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(self.config, f, allow_unicode=True, default_flow_style=False)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
        
        logger.info(f"配置已保存到: {config_path}")
    
    def update_config(self, new_config: Dict[str, Any], validate: bool = True):
        """更新配置
        
        Args:
            new_config: 新的配置字典
            validate: 是否验证更新后的配置
        """
        # 递归更新配置
        def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    d[k] = deep_update(d[k], v)
                else:
                    d[k] = v
            return d
        
        self.config = deep_update(self.config, new_config)
        
        # 验证更新后的配置
        if validate and not self.validate_config():
            raise ValueError("更新后的配置验证失败")
        
        logger.info("配置已更新")
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项
        
        Args:
            key: 配置项键名
            default: 默认值
            
        Returns:
            配置项值
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any, validate: bool = True):
        """设置配置项
        
        Args:
            key: 配置项键名
            value: 配置项值
            validate: 是否验证更新后的配置
        """
        self.config[key] = value
        
        # 验证更新后的配置
        if validate and not self.validate_config():
            raise ValueError("更新后的配置验证失败")
        
        logger.info(f"配置项已更新: {key}")
    
    def __getitem__(self, key: str) -> Any:
        """通过字典方式访问配置项
        
        Args:
            key: 配置项键名
            
        Returns:
            配置项值
        """
        return self.config[key]
    
    def __setitem__(self, key: str, value: Any):
        """通过字典方式设置配置项
        
        Args:
            key: 配置项键名
            value: 配置项值
        """
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """检查配置项是否存在
        
        Args:
            key: 配置项键名
            
        Returns:
            配置项是否存在
        """
        return key in self.config
    
    def __str__(self) -> str:
        """返回配置的字符串表示"""
        return json.dumps(self.config, ensure_ascii=False, indent=2) 