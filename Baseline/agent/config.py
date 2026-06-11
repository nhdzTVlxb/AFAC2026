"""配置管理系统 - 支持YAML配置文件和环境变量覆盖"""
import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any


@dataclass
class LLMConfig:
    """LLM配置"""
    model: str = "qwen-turbo"
    api_key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 120

    def __post_init__(self):
        # 从环境变量读取（如果未设置）
        if not self.api_key:
            self.api_key = os.environ.get("LLM_API_KEY", "")
        if not self.base_url:
            self.base_url = os.environ.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        if os.environ.get("LLM_MODEL"):
            self.model = os.environ.get("LLM_MODEL")


@dataclass
class ResearchConfig:
    """科研流程配置"""
    budget: int = 10
    time_limit: int = 3600
    early_stop_patience: int = 3
    enable_literature: bool = True
    enable_diagnosis: bool = True
    enable_design: bool = True
    enable_experiment: bool = True
    max_iterations: int = 20
    output_dir: str = "./output"


@dataclass
class TaskConfig:
    """单个任务配置"""
    task_id: int = 1
    task_type: str = ""
    data_path: str = ""
    sample_path: str = ""
    model_type: str = "sage"
    hidden_dim: int = 128
    num_layers: int = 2
    lr: float = 0.01
    epochs: int = 200
    early_stop: int = 20
    batch_size: int = 256
    embedding_dim: int = 64
    max_seq_len: int = 50
    dropout: float = 0.5
    output_dir: str = "./output"
    code_dir: str = "./code"


@dataclass
class SystemConfig:
    """系统总配置"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    tasks: Dict[str, TaskConfig] = field(default_factory=dict)
    device: str = "auto"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "SystemConfig":
        """从YAML文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}

        llm_cfg = LLMConfig(**data.get("llm", {}))
        research_cfg = ResearchConfig(**data.get("research", {}))

        tasks = {}
        for task_name, task_data in data.get("tasks", {}).items():
            tasks[task_name] = TaskConfig(**task_data)

        return cls(
            llm=llm_cfg,
            research=research_cfg,
            tasks=tasks,
            device=data.get("device", "auto"),
            seed=data.get("seed", 42)
        )

    @classmethod
    def from_env(cls) -> "SystemConfig":
        """从环境变量加载配置"""
        return cls(
            llm=LLMConfig(),
            device=os.environ.get("DEVICE", "auto"),
            seed=int(os.environ.get("SEED", 42))
        )

    def to_yaml(self, path: str):
        """保存配置到YAML文件"""
        data = {
            "llm": asdict(self.llm),
            "research": asdict(self.research),
            "tasks": {k: asdict(v) for k, v in self.tasks.items()},
            "device": self.device,
            "seed": self.seed
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def get_task_config(self, task_id: int) -> TaskConfig:
        """获取指定任务的配置"""
        for name, cfg in self.tasks.items():
            if cfg.task_id == task_id:
                return cfg
        # 返回默认配置
        defaults = {
            1: TaskConfig(task_id=1, task_type="classification", model_type="sage"),
            2: TaskConfig(task_id=2, task_type="recommendation", model_type="gru4rec")
        }
        return defaults.get(task_id, TaskConfig(task_id=task_id))
