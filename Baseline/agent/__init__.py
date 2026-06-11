"""自主科研Agent框架 - 核心包

面向金融场景图学习竞赛的自动化实验Agent系统。
支持四阶段科研闭环：文献解析→瓶颈诊断→代码设计→实验验证。
"""

from .config import SystemConfig, LLMConfig, ResearchConfig, TaskConfig
from .llm_client import LLMClient
from .memory import ResearchMemory, ExperimentRecord
from .orchestrator import ResearchOrchestrator
from .phases import LiteraturePhase, DiagnosisPhase, DesignPhase, ExperimentPhase
from .tools import ToolRegistry
from .code_generator import CodeGenerator

__all__ = [
    "SystemConfig", "LLMConfig", "ResearchConfig", "TaskConfig",
    "LLMClient", "ResearchMemory", "ExperimentRecord",
    "ResearchOrchestrator",
    "LiteraturePhase", "DiagnosisPhase", "DesignPhase", "ExperimentPhase",
    "ToolRegistry", "CodeGenerator",
]
