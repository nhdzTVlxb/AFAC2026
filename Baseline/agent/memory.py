"""研究记忆系统 - 支持实验记录和状态持久化"""
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime


@dataclass
class ExperimentRecord:
    """单次实验记录"""
    iteration: int = 0
    phase: str = ""
    task_id: int = 1
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    decision: str = "CONTINUE"
    reason: str = ""
    code_changes: str = ""
    duration: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentRecord":
        return cls(**d)


class ResearchMemory:
    """
    研究记忆系统

    持久化存储所有实验记录、最佳结果和Agent状态。
    支持保存/加载JSON文件，趋势分析，智能评估。
    """

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        self.iterations: List[ExperimentRecord] = []
        self.best_result: Optional[ExperimentRecord] = None
        self.literature_summary: str = ""
        self.diagnosis_report: str = ""
        self.design_notes: str = ""
        self.current_task_id: int = 1

    def add_iteration(self, record: ExperimentRecord):
        """添加实验记录并更新最佳结果"""
        self.iterations.append(record)
        if self._is_better(record, self.best_result):
            self.best_result = record

    def add_record(self, task: str, round: int, phase: str,
                   config: dict, metrics: dict, feedback: str, duration: float):
        """添加实验记录（兼容接口，供orchestrator/phases调用）

        Args:
            task: 任务标识，如 "task1"
            round: 轮次编号
            phase: 阶段名称
            config: 配置字典
            metrics: 指标字典
            feedback: 反馈文本
            duration: 耗时（秒）
        """
        # 从task字符串中提取task_id
        task_id = 1
        if isinstance(task, str) and task.startswith('task'):
            try:
                task_id = int(task.replace('task', ''))
            except ValueError:
                pass

        record = ExperimentRecord(
            iteration=round,
            phase=phase,
            task_id=task_id,
            config=config,
            metrics=metrics,
            decision="CONTINUE",
            reason=feedback,
            duration=duration,
            timestamp=datetime.now().isoformat()
        )
        self.add_iteration(record)

    def get_iterations(self, task_id: Optional[int] = None) -> List[ExperimentRecord]:
        """获取指定任务的实验记录"""
        if task_id is None:
            return self.iterations
        return [r for r in self.iterations if r.task_id == task_id]

    def get_history(self, task: str) -> List[dict]:
        """获取指定任务的历史记录（兼容接口，返回字典列表）

        Args:
            task: 任务标识，如 "task1"

        Returns:
            历史记录字典列表
        """
        task_id = 1
        if isinstance(task, str) and task.startswith('task'):
            try:
                task_id = int(task.replace('task', ''))
            except ValueError:
                pass
        records = self.get_iterations(task_id)
        return [r.to_dict() for r in records]

    def get_best_iteration(self, task_id: Optional[int] = None) -> Optional[ExperimentRecord]:
        """获取最佳实验记录"""
        if task_id is None:
            return self.best_result
        task_records = self.get_iterations(task_id)
        if not task_records:
            return None
        best = task_records[0]
        for r in task_records[1:]:
            if self._is_better(r, best):
                best = r
        return best

    def get_metrics_history(self, metric_name: str = "acc") -> List[float]:
        """获取指标历史"""
        return [r.metrics.get(metric_name, 0.0) for r in self.iterations if metric_name in r.metrics]

    def is_improving(self, window: int = 3) -> bool:
        """判断最近window轮是否在提升"""
        history = self.get_metrics_history()
        if len(history) < window + 1:
            return True
        recent = history[-window:]
        previous = history[-window - 1:-1]
        return sum(recent) / len(recent) > sum(previous) / len(previous)

    def save(self, task_id: int):
        """保存记忆到task{N}/research_memory.json"""
        task_dir = os.path.join(self.output_dir, f"task{task_id}")
        os.makedirs(task_dir, exist_ok=True)
        path = os.path.join(task_dir, "research_memory.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.get_state_dict(), f, ensure_ascii=False, indent=2)

    def load(self, task_id: int = 0, path: str = ""):
        """从task{N}/research_memory.json加载

        Args:
            task_id: 任务ID，若为0则尝试从self.output_dir直接加载
            path: 若提供，直接从该路径加载（优先级高于task_id）
        """
        if path and os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                self.load_state_dict(json.load(f))
            return

        if task_id > 0:
            path = os.path.join(self.output_dir, f"task{task_id}", "research_memory.json")
        else:
            path = os.path.join(self.output_dir, "research_memory.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                self.load_state_dict(json.load(f))

    def generate_summary(self) -> str:
        """生成研究总结报告"""
        lines = ["# 研究总结报告\n"]
        lines.append(f"总实验轮数: {len(self.iterations)}\n")
        if self.best_result:
            lines.append(f"最佳轮次: #{self.best_result.iteration}\n")
            lines.append(f"最佳指标: {json.dumps(self.best_result.metrics, ensure_ascii=False)}\n")
        for r in self.iterations:
            lines.append(f"\n## 第{r.iteration}轮 ({r.phase})\n")
            lines.append(f"- 决策: {r.decision}\n")
            lines.append(f"- 指标: {json.dumps(r.metrics, ensure_ascii=False)}\n")
            lines.append(f"- 原因: {r.reason}\n")
        return "".join(lines)

    def get_state_dict(self) -> dict:
        """获取状态字典（用于序列化）"""
        return {
            "iterations": [r.to_dict() for r in self.iterations],
            "best_result": self.best_result.to_dict() if self.best_result else None,
            "literature_summary": self.literature_summary,
            "diagnosis_report": self.diagnosis_report,
            "design_notes": self.design_notes,
            "current_task_id": self.current_task_id
        }

    def load_state_dict(self, state: dict):
        """从状态字典恢复"""
        self.iterations = [ExperimentRecord.from_dict(d) for d in state.get("iterations", [])]
        best = state.get("best_result")
        self.best_result = ExperimentRecord.from_dict(best) if best else None
        self.literature_summary = state.get("literature_summary", "")
        self.diagnosis_report = state.get("diagnosis_report", "")
        self.design_notes = state.get("design_notes", "")
        self.current_task_id = state.get("current_task_id", 1)

    def _is_better(self, a: ExperimentRecord, b: Optional[ExperimentRecord]) -> bool:
        """判断记录a是否优于记录b"""
        if b is None:
            return True
        score_a = self._get_primary_score(a)
        score_b = self._get_primary_score(b)
        return score_a > score_b

    def _get_primary_score(self, record: ExperimentRecord) -> float:
        """获取主要评分指标"""
        m = record.metrics
        for key in ["acc", "ndcg", "mrr", "f1", "auc", "val_acc", "val_ndcg", "val_mrr"]:
            if key in m and isinstance(m[key], (int, float)):
                return m[key]
        # 只考虑数值类型的指标
        numeric_values = [v for v in m.values() if isinstance(v, (int, float))]
        return sum(numeric_values) / len(numeric_values) if numeric_values else 0.0
