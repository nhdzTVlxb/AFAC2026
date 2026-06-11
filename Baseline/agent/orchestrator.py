"""
主编排器：四阶段调度、停止条件、打包提交

ResearchOrchestrator是自主科研Agent框架的核心控制器，负责：
1. 按序执行 Literature -> Diagnosis -> Design -> Experiment 四阶段
2. 根据Experiment的决策结果实现迭代闭环（CONTINUE/PIVOT/STOP）
3. 多维度停止条件检查（budget、时间、早停、显式STOP）
4. _finalize打包最佳结果为submission格式

使用示例:
    config = SystemConfig.load("config.yaml")
    orchestrator = ResearchOrchestrator(config)
    results = orchestrator.run()
"""

import os
import sys
import time
import json
import shutil
import traceback
import zipfile
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

# 导入四阶段实现
from .phases import (
    BasePhase, LiteraturePhase, DiagnosisPhase,
    DesignPhase, ExperimentPhase
)


class ResearchOrchestrator:
    """
    科研主编排器

    负责四阶段调度和迭代闭环：
    1. 按序执行 Literature -> Diagnosis -> Design -> Experiment
    2. 根据Experiment结果决定：CONTINUE（继续优化）/ PIVOT（换方向）/ STOP（停止）
    3. STOP时调用 _finalize 打包提交

    停止条件：
    - 达到 budget 上限（每任务最大实验轮数）
    - 达到时间限制（全局时间上限）
    - 连续多轮无提升（early_stop_patience）
    - LLM 明确建议 STOP

    迭代闭环逻辑：
    - CONTINUE: 回到 Diagnosis -> Design -> Experiment
    - PIVOT: 回到 Literature（重新分析）-> Diagnosis -> Design -> Experiment
    - STOP: 调用 _finalize 打包提交
    """

    # 决策常量
    DECISION_CONTINUE = "CONTINUE"
    DECISION_PIVOT = "PIVOT"
    DECISION_STOP = "STOP"

    def __init__(self, config, task_ids=None):
        """
        初始化编排器

        Args:
            config: SystemConfig实例，包含llm、research、tasks等配置
            task_ids: 要运行的任务ID列表，None表示运行所有配置中的任务
        """
        self.config = config
        self.task_ids = task_ids  # 用户指定的任务列表

        # 记录启动时间（用于时间限制检查）
        self.start_time = time.time()

        # 初始化LLM客户端（延迟导入避免循环依赖）
        try:
            from .llm_client import LLMClient
            llm_config = getattr(config, 'llm', None)
            if llm_config:
                self.llm = LLMClient(
                    model=getattr(llm_config, 'model', 'qwen-turbo'),
                    api_key=getattr(llm_config, 'api_key', ''),
                    base_url=getattr(llm_config, 'base_url', ''),
                    temperature=getattr(llm_config, 'temperature', 0.7)
                )
            else:
                # 创建默认LLM客户端
                self.llm = LLMClient()
        except ImportError:
            # LLM模块不可用，创建一个mock
            self.llm = self._create_mock_llm()

        # 初始化记忆系统
        try:
            from .memory import ResearchMemory
            research_config = getattr(config, 'research', None)
            if research_config:
                output_dir = getattr(research_config, 'output_dir', './output')
            else:
                output_dir = './output'
            self.memory = ResearchMemory(output_dir=output_dir)
        except ImportError:
            # Memory模块不可用，创建mock
            self.memory = self._create_mock_memory()

        # 初始化工具注册表
        try:
            from .tools import ToolRegistry
            self.tools = ToolRegistry()
        except ImportError:
            self.tools = self._create_mock_tools()

        # 阶段实例缓存（按需创建）
        self._phases = {}

        # 每任务的迭代计数器
        self._iteration_counters = {}

        # 每任务的连续PIVOT计数（防止无限PIVOT循环）
        self._pivot_counters = {}

        # 每任务的最佳结果
        self._best_results = {}

        # 全局输出目录
        research_config = getattr(config, 'research', None)
        self.output_dir = getattr(research_config, 'output_dir', './output') if research_config else './output'
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"[Orchestrator] 初始化完成")
        print(f"  输出目录: {self.output_dir}")
        print(f"  预算: {getattr(research_config, 'budget', 10) if research_config else 10}")
        print(f"  时间限制: {getattr(research_config, 'time_limit', 3600) if research_config else 3600}s")

    # ------------------------------------------------------------------
    # 主入口方法
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """
        运行所有任务的科研流程

        遍历配置中的所有任务，为每个任务执行完整的科研闭环。

        Returns:
            {
                "task_results": {1: {...}, 2: {...}},
                "submission_dir": "output/submission",
                "total_time": 1234.5
            }
        """
        print("\n" + "=" * 60)
        print("ResearchOrchestrator: 启动自主科研流程")
        print("=" * 60)

        total_start = time.time()

        # 获取任务列表
        tasks = self._get_tasks()

        # 过滤用户指定的任务
        if self.task_ids is not None:
            tasks = {tid: tc for tid, tc in tasks.items() if tid in self.task_ids}

        if not tasks:
            print("[Orchestrator] 警告: 没有匹配的任务需要运行")
            return {
                "task_results": {},
                "submission_dir": "",
                "total_time": 0.0
            }

        print(f"[Orchestrator] 发现 {len(tasks)} 个任务: {list(tasks.keys())}")

        task_results = {}

        # 依次运行每个任务
        for task_id, task_config in tasks.items():
            # 检查全局时间限制
            if self._check_time_limit():
                print(f"[Orchestrator] 时间限制已到，跳过后续任务")
                break

            # 运行单个任务
            result = self.run_task(task_id)
            task_results[task_id] = result

        total_time = time.time() - total_start

        # 打包所有任务的提交产物
        submission_dir = self._finalize_all(task_results)

        final_result = {
            "task_results": task_results,
            "submission_dir": submission_dir,
            "total_time": round(total_time, 2)
        }

        print("\n" + "=" * 60)
        print("ResearchOrchestrator: 所有任务完成")
        print(f"总耗时: {total_time:.1f}s")
        print(f"提交目录: {submission_dir}")
        print("=" * 60)

        return final_result

    def run_task(self, task_id: int) -> Dict[str, Any]:
        """
        为指定任务运行完整科研流程

        流程：
        1. Phase 1: Literature（仅第1次）
        2. Phase 2: Diagnosis
        3. Phase 3: Design
        4. Phase 4: Experiment
        5. 根据决策继续迭代或停止
        6. _finalize 打包

        Args:
            task_id: 任务ID（1, 2, ...）

        Returns:
            {
                "task_id": 1,
                "literature": {...},
                "iterations": [
                    {
                        "iteration": 1,
                        "diagnosis": {...},
                        "design": {...},
                        "experiment": {...}
                    }
                ],
                "best_metrics": {...},
                "submission_file": "output/submission/A1.csv",
                "status": "completed"
            }
        """
        print(f"\n{'=' * 60}")
        print(f"[Orchestrator] 开始任务 {task_id}")
        print(f"{'=' * 60}")

        task_start = time.time()

        # 获取任务配置
        task_config = self._get_task_config(task_id)
        if not task_config:
            print(f"[Orchestrator] 任务 {task_id} 配置不存在，跳过")
            return {"task_id": task_id, "status": "skipped", "reason": "config_missing"}

        # 设置任务专属输出目录，避免多任务间文件覆盖
        task_output_dir = os.path.join(self.output_dir, f"task{task_id}")
        task_config.output_dir = task_output_dir
        os.makedirs(task_output_dir, exist_ok=True)
        print(f"[Orchestrator] 任务 {task_id} 输出目录: {task_output_dir}")

        # 初始化任务状态
        self._iteration_counters[task_id] = 0
        self._pivot_counters[task_id] = 0

        result = {
            "task_id": task_id,
            "literature": None,
            "iterations": [],
            "best_metrics": {},
            "submission_file": "",
            "status": "running"
        }

        # ================================================================
        # Phase 1: Literature（文献解析，仅执行一次）
        # ================================================================
        if self._should_run_phase('literature'):
            try:
                lit_phase = self._get_phase('literature', task_config)
                lit_result = lit_phase.run()
                result["literature"] = lit_result
                print(f"\n[Phase 1完成] 关键挑战: {lit_result.get('key_challenges', [])}")
            except Exception as e:
                print(f"[Phase 1错误] {e}")
                traceback.print_exc()
                result["literature"] = {"error": str(e)}
        else:
            print("[Phase 1] 配置中禁用")

        # ================================================================
        # 迭代闭环: Diagnosis -> Design -> Experiment
        # ================================================================
        decision = self.DECISION_CONTINUE

        while decision in (self.DECISION_CONTINUE, self.DECISION_PIVOT):
            # 检查停止条件
            iteration = self._iteration_counters.get(task_id, 0)
            if self._should_stop(task_id, iteration, decision):
                print(f"\n[Orchestrator] 停止条件触发，终止迭代")
                break

            # 检查时间限制
            if self._check_time_limit():
                print(f"\n[Orchestrator] 全局时间限制触发，终止迭代")
                break

            # 增加迭代计数
            self._iteration_counters[task_id] = iteration + 1
            current_iter = self._iteration_counters[task_id]

            print(f"\n{'-' * 50}")
            print(f"[Orchestrator] 迭代 #{current_iter} (决策: {decision})")
            print(f"{'-' * 50}")

            iter_record = {
                "iteration": current_iter,
                "diagnosis": None,
                "design": None,
                "experiment": None
            }

            # --------------------------------------------------------
            # Phase 2: Diagnosis（瓶颈诊断）
            # --------------------------------------------------------
            if self._should_run_phase('diagnosis'):
                try:
                    diag_phase = self._get_phase('diagnosis', task_config)
                    diag_result = diag_phase.run()
                    iter_record["diagnosis"] = diag_result
                    print(f"\n[Phase 2完成] 瓶颈: {diag_result.get('bottlenecks', [])}")
                    print(f"  假设: {[h.get('description', '')[:50] for h in diag_result.get('hypotheses', [])]}")
                except Exception as e:
                    print(f"[Phase 2错误] {e}")
                    traceback.print_exc()
                    iter_record["diagnosis"] = {"error": str(e)}
                    # 诊断失败时使用默认假设继续
                    diag_result = {
                        "bottlenecks": ["诊断失败，使用默认假设"],
                        "hypotheses": [{"description": "调整学习率", "expected_improvement": "1-3%", "test_method": "对比实验"}]
                    }
            else:
                print("[Phase 2] 配置中禁用")
                diag_result = {"bottlenecks": [], "hypotheses": []}

            # --------------------------------------------------------
            # Phase 3: Design（代码设计）
            # --------------------------------------------------------
            design_result = None
            if self._should_run_phase('design'):
                try:
                    des_phase = self._get_phase('design', task_config)
                    design_result = des_phase.run(diag_result)
                    iter_record["design"] = design_result
                    print(f"\n[Phase 3完成] 修改文件: {design_result.get('files_modified', [])}")
                    print(f"  语法验证: {'通过' if design_result.get('validation_passed') else '失败'}")
                    print(f"  Smoke test: {'通过' if design_result.get('smoke_test_passed') else '失败'}")
                except Exception as e:
                    print(f"[Phase 3错误] {e}")
                    traceback.print_exc()
                    iter_record["design"] = {"error": str(e)}
                    design_result = None
            else:
                print("[Phase 3] 配置中禁用")

            # --------------------------------------------------------
            # Phase 4: Experiment（实验验证）
            # --------------------------------------------------------
            if self._should_run_phase('experiment'):
                try:
                    exp_phase = self._get_phase('experiment', task_config)
                    exp_result = exp_phase.run(current_iter)
                    iter_record["experiment"] = exp_result
                    decision = exp_result.get("decision", self.DECISION_STOP)
                    reason = exp_result.get("reason", "")

                    print(f"\n[Phase 4完成] 指标: {exp_result.get('metrics', {})}")
                    print(f"  决策: {decision} - {reason}")
                    print(f"  耗时: {exp_result.get('duration', 0):.1f}s")

                    # 更新最佳结果
                    self._update_best_result(task_id, exp_result)

                except Exception as e:
                    print(f"[Phase 4错误] {e}")
                    traceback.print_exc()
                    iter_record["experiment"] = {"error": str(e)}
                    decision = self.DECISION_PIVOT
                    reason = f"实验失败: {str(e)}"
            else:
                print("[Phase 4] 配置中禁用")
                decision = self.DECISION_STOP
                reason = "实验阶段已禁用"

            # 记录本轮迭代
            result["iterations"].append(iter_record)

            # 处理PIVOT决策
            if decision == self.DECISION_PIVOT:
                self._pivot_counters[task_id] = self._pivot_counters.get(task_id, 0) + 1
                if self._pivot_counters[task_id] >= 3:
                    print(f"[Orchestrator] 连续PIVOT超过3次，强制STOP")
                    decision = self.DECISION_STOP
                    reason = "连续PIVOT次数过多"
            else:
                # 重置PIVOT计数
                self._pivot_counters[task_id] = 0

            # 检查迭代预算
            budget = self._get_task_budget(task_id)
            if current_iter >= budget:
                print(f"[Orchestrator] 迭代预算已用完 ({current_iter}/{budget})")
                decision = self.DECISION_STOP
                reason = f"达到迭代预算上限 ({current_iter}/{budget})"

        # ================================================================
        # Finalize: 打包提交
        # ================================================================
        print(f"\n[Orchestrator] 任务 {task_id} 科研流程结束，开始打包...")

        try:
            submission_file = self._finalize(task_id)
            result["submission_file"] = submission_file
            result["best_metrics"] = self._best_results.get(task_id, {}).get("metrics", {})
            result["status"] = "completed"
            print(f"[Orchestrator] 打包完成: {submission_file}")
        except Exception as e:
            print(f"[Orchestrator] 打包失败: {e}")
            traceback.print_exc()
            result["status"] = "failed"
            result["error"] = str(e)

        task_duration = time.time() - task_start
        result["duration"] = round(task_duration, 2)

        print(f"\n[Orchestrator] 任务 {task_id} 完成")
        print(f"  总迭代: {self._iteration_counters.get(task_id, 0)}")
        print(f"  耗时: {task_duration:.1f}s")
        print(f"  状态: {result['status']}")

        return result

    # ------------------------------------------------------------------
    # 停止条件
    # ------------------------------------------------------------------

    def _should_stop(self, task_id: int, iteration: int,
                     current_decision: str) -> bool:
        """
        判断是否应该停止

        检查多维度停止条件：
        1. 达到 budget 上限
        2. 达到时间限制
        3. 连续多轮无提升（early_stop）
        4. 显式 STOP 决策
        5. 连续 PIVOT 次数过多

        Args:
            task_id: 任务ID
            iteration: 当前迭代轮次
            current_decision: 当前决策（CONTINUE/PIVOT/STOP）

        Returns:
            True表示应该停止
        """
        # 条件4: 显式STOP决策
        if current_decision == self.DECISION_STOP:
            self._log_stop_reason("LLM明确建议STOP")
            return True

        # 条件1: 达到budget上限
        budget = self._get_task_budget(task_id)
        if iteration >= budget:
            self._log_stop_reason(f"达到预算上限 ({iteration}/{budget})")
            return True

        # 条件2: 达到时间限制
        if self._check_time_limit():
            self._log_stop_reason("达到全局时间限制")
            return True

        # 条件3: 连续多轮无提升（early_stop）
        if self._check_early_stop(task_id):
            return True

        # 条件5: 连续PIVOT次数过多
        pivot_count = self._pivot_counters.get(task_id, 0)
        if pivot_count >= 3:
            self._log_stop_reason(f"连续PIVOT次数过多 ({pivot_count})")
            return True

        return False

    def _check_time_limit(self) -> bool:
        """检查是否达到全局时间限制"""
        research_config = getattr(self.config, 'research', None)
        if not research_config:
            return False

        time_limit = getattr(research_config, 'time_limit', 0)
        if time_limit <= 0:
            return False  # 0表示无限制

        elapsed = time.time() - self.start_time
        if elapsed >= time_limit:
            self._log_stop_reason(f"时间限制 {time_limit}s 已到，已用 {elapsed:.1f}s")
            return True

        return False

    def _check_early_stop(self, task_id: int) -> bool:
        """
        检查是否触发早停

        连续多轮主要指标无提升则触发早停。
        """
        research_config = getattr(self.config, 'research', None)
        if not research_config:
            return False

        patience = getattr(research_config, 'early_stop_patience', 3)
        if patience <= 0:
            return False  # 0表示禁用早停

        task_key = f"task{task_id}"
        history = self.memory.get_history(task_key)

        if len(history) < patience:
            return False

        # 获取主要指标（从历史记录中推断）
        sample_metrics = history[-1].get("metrics", {}) if history else {}
        metric_key = self._get_task_metric_key(task_id, sample_metrics)

        # 获取最近 patience 轮的指标值
        recent_values = []
        for record in history[-patience:]:
            metrics = record.get("metrics", {})
            if metric_key in metrics:
                recent_values.append(metrics[metric_key])

        if len(recent_values) < patience:
            return False

        # 找到最佳值
        best_value = max(recent_values)

        # 检查最近一轮是否是最优
        if recent_values[-1] < best_value:
            # 最近一轮不是最优，检查已经连续多少轮不是最优
            no_improve_count = 0
            for v in reversed(recent_values):
                if v < best_value:
                    no_improve_count += 1
                else:
                    break

            if no_improve_count >= patience:
                self._log_stop_reason(
                    f"早停触发: 连续{no_improve_count}轮{metric_key}无提升"
                )
                return True

        return False

    # ------------------------------------------------------------------
    # Finalize: 打包提交
    # ------------------------------------------------------------------

    def _finalize(self, task_id: int) -> str:
        """
        打包单个任务的提交产物

        1. 收集最佳结果（A1.csv 或 A2.csv）
        2. 生成实验轨迹 trajectory.json
        3. 保存到提交目录

        Args:
            task_id: 任务ID

        Returns:
            提交文件路径
        """
        print(f"\n[Finalize] 打包任务 {task_id}")

        # 创建提交目录
        submission_dir = os.path.join(self.output_dir, "submission")
        os.makedirs(submission_dir, exist_ok=True)

        # 确定输出文件名
        output_filename = f"A{task_id}.csv"
        submission_path = os.path.join(submission_dir, output_filename)

        # 查找最佳结果文件
        best_result = self._best_results.get(task_id, {})
        best_output_files = best_result.get("output_files", [])

        if best_output_files:
            # 复制最佳结果到提交目录
            best_file = best_output_files[0]
            if os.path.exists(best_file):
                shutil.copy2(best_file, submission_path)
                print(f"  复制最佳结果: {best_file} -> {submission_path}")
            else:
                # 尝试从输出目录查找
                task_output_dir = os.path.join(self.output_dir, f"task{task_id}")
                candidate_files = [
                    os.path.join(task_output_dir, output_filename),
                    os.path.join(self.output_dir, output_filename),
                ]
                for cf in candidate_files:
                    if os.path.exists(cf):
                        shutil.copy2(cf, submission_path)
                        print(f"  复制候选文件: {cf} -> {submission_path}")
                        break
                else:
                    # 创建空的提交文件（占位）
                    self._create_empty_submission(task_id, submission_path)
        else:
            # 没有实验结果，尝试查找已有的输出文件
            task_output_dir = os.path.join(self.output_dir, f"task{task_id}")
            candidate_files = [
                os.path.join(task_output_dir, output_filename),
                os.path.join(self.output_dir, output_filename),
            ]
            found = False
            for cf in candidate_files:
                if os.path.exists(cf):
                    shutil.copy2(cf, submission_path)
                    print(f"  复制已有文件: {cf} -> {submission_path}")
                    found = True
                    break
            if not found:
                self._create_empty_submission(task_id, submission_path)

        # 生成实验轨迹
        self._save_trajectory(task_id, submission_dir)

        print(f"  [Finalize] 完成: {submission_path}")
        return submission_path

    def _finalize_all(self, task_results: Dict[int, Any]) -> str:
        """
        打包所有任务的提交产物

        将所有任务的提交文件收集到一个目录中，
        并生成完整的实验轨迹文件。

        Args:
            task_results: 各任务的结果字典

        Returns:
            提交目录路径
        """
        print(f"\n[Finalize] 打包所有任务")

        submission_dir = os.path.join(self.output_dir, "submission")
        os.makedirs(submission_dir, exist_ok=True)

        # 收集所有任务的提交文件
        for task_id, result in task_results.items():
            if result.get("status") == "completed":
                submission_file = result.get("submission_file", "")
                if submission_file and os.path.exists(submission_file):
                    # 确保文件在提交目录中
                    dest = os.path.join(submission_dir, f"A{task_id}.csv")
                    if os.path.abspath(submission_file) != os.path.abspath(dest):
                        shutil.copy2(submission_file, dest)

        # 生成完整的实验轨迹
        trajectory_path = os.path.join(submission_dir, "trajectory.json")
        trajectory = {
            "timestamp": datetime.now().isoformat(),
            "tasks": {}
        }

        for task_id, result in task_results.items():
            trajectory["tasks"][f"task{task_id}"] = {
                "status": result.get("status"),
                "iterations": len(result.get("iterations", [])),
                "best_metrics": result.get("best_metrics", {}),
                "duration": result.get("duration", 0)
            }

        with open(trajectory_path, 'w', encoding='utf-8') as f:
            json.dump(trajectory, f, ensure_ascii=False, indent=2)

        print(f"  [Finalize] 提交目录: {submission_dir}")
        print(f"  [Finalize] 轨迹文件: {trajectory_path}")

        # 打包为zip
        zip_path = self._create_submission_zip(submission_dir)
        if zip_path:
            print(f"  [Finalize] ZIP文件: {zip_path}")

        return submission_dir

    def _create_submission_zip(self, submission_dir: str) -> Optional[str]:
        """
        将提交目录打包为zip文件

        Args:
            submission_dir: 提交目录路径

        Returns:
            zip文件路径，失败返回None
        """
        zip_path = f"{submission_dir}.zip"
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for root, dirs, files in os.walk(submission_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arc_name = os.path.relpath(file_path, submission_dir)
                        zf.write(file_path, arc_name)
            return zip_path
        except Exception as e:
            print(f"  [Finalize] ZIP打包失败: {e}")
            return None

    def _create_empty_submission(self, task_id: int, path: str):
        """
        创建空的提交文件（占位用）

        当没有实验结果时，创建格式正确的空文件。
        """
        print(f"  创建空提交文件: {path}")

        task_config = self._get_task_config(task_id)
        task_type = getattr(task_config, 'task_type', 'classification')

        try:
            import pandas as pd
            if task_type == 'classification':
                df = pd.DataFrame(columns=['node_id', 'label'])
            else:
                df = pd.DataFrame(columns=['uid', 'prediction'])
            df.to_csv(path, index=False)
        except ImportError:
            # 没有pandas，手写CSV
            with open(path, 'w', encoding='utf-8') as f:
                if task_type == 'classification':
                    f.write("node_id,label\n")
                else:
                    f.write("uid,prediction\n")

    def _save_trajectory(self, task_id: int, submission_dir: str):
        """
        保存实验轨迹到提交目录

        轨迹记录了完整的实验过程，包括每轮的配置、指标和决策。
        """
        task_key = f"task{task_id}"
        history = self.memory.get_history(task_key)

        if not history:
            return

        trajectory = {
            "task_id": task_id,
            "num_iterations": len(history),
            "records": []
        }

        for record in history:
            trajectory["records"].append({
                "round": record.get("round", 0),
                "phase": record.get("phase", ""),
                "config": record.get("config", {}),
                "metrics": record.get("metrics", {}),
                "feedback": record.get("feedback", ""),
                "duration": record.get("duration", 0),
                "timestamp": record.get("timestamp", "")
            })

        # 最佳记录
        best = self._best_results.get(task_id, {})
        trajectory["best_result"] = {
            "iteration": best.get("iteration", 0),
            "metrics": best.get("metrics", {})
        }

        trajectory_path = os.path.join(submission_dir, f"trajectory_task{task_id}.json")
        with open(trajectory_path, 'w', encoding='utf-8') as f:
            json.dump(trajectory, f, ensure_ascii=False, indent=2)

        print(f"  轨迹已保存: {trajectory_path}")

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _get_phase(self, phase_name: str, task_config) -> BasePhase:
        """
        获取阶段实例（懒加载缓存）

        Args:
            phase_name: 阶段名称 ('literature', 'diagnosis', 'design', 'experiment')
            task_config: 任务配置

        Returns:
            BasePhase子类实例
        """
        cache_key = f"{phase_name}_{getattr(task_config, 'task_id', 0)}"

        if cache_key not in self._phases:
            phase_map = {
                'literature': LiteraturePhase,
                'diagnosis': DiagnosisPhase,
                'design': DesignPhase,
                'experiment': ExperimentPhase
            }

            phase_class = phase_map.get(phase_name)
            if not phase_class:
                raise ValueError(f"未知的阶段名称: {phase_name}")

            self._phases[cache_key] = phase_class(
                llm_client=self.llm,
                memory=self.memory,
                tools=self.tools,
                task_config=task_config
            )

        return self._phases[cache_key]

    def _get_tasks(self) -> Dict[int, Any]:
        """
        获取所有任务配置

        从config中解析tasks配置，返回 {task_id: task_config} 字典。
        """
        tasks = {}

        # 尝试从config.tasks获取
        tasks_config = getattr(self.config, 'tasks', None)
        if tasks_config:
            for key, task_conf in tasks_config.items():
                # 从key中提取task_id，例如 "task1" -> 1
                if key.startswith('task'):
                    try:
                        task_id = int(key.replace('task', ''))
                        tasks[task_id] = task_conf
                    except ValueError:
                        pass

        # 如果没有找到任务，使用默认配置
        if not tasks:
            # 检查是否有task1/task2的直接属性
            for i in [1, 2]:
                task_attr = f"task{i}"
                if hasattr(self.config, task_attr):
                    task_conf = getattr(self.config, task_attr)
                    # 确保task_conf有task_id
                    if not hasattr(task_conf, 'task_id'):
                        task_conf.task_id = i
                    tasks[i] = task_conf

        return tasks

    def _get_task_config(self, task_id: int) -> Any:
        """获取指定任务的配置"""
        tasks = self._get_tasks()
        return tasks.get(task_id)

    def _get_task_budget(self, task_id: int) -> int:
        """获取指定任务的预算"""
        research_config = getattr(self.config, 'research', None)
        if research_config:
            return getattr(research_config, 'budget', 10)

        # 从任务配置中查找
        task_config = self._get_task_config(task_id)
        if task_config:
            return getattr(task_config, 'budget', 10)

        return 10  # 默认值

    def _get_task_metric_key(self, task_id: int, metrics: Dict[str, Any] = None) -> str:
        """获取指定任务的主要指标键名"""
        task_config = self._get_task_config(task_id)
        if task_config:
            task_type = getattr(task_config, 'task_type', 'classification')
            if task_type == 'classification':
                if metrics and 'val_acc' in metrics:
                    return 'val_acc'
                return 'acc'
            elif task_type == 'recommendation':
                if metrics and 'val_mrr' in metrics:
                    return 'val_mrr'
                return 'mrr'
        return 'acc'

    def _should_run_phase(self, phase_name: str) -> bool:
        """
        检查某阶段是否在配置中启用

        Args:
            phase_name: 阶段名称

        Returns:
            True表示应该执行该阶段
        """
        research_config = getattr(self.config, 'research', None)
        if not research_config:
            return True  # 默认全部启用

        # 使用 enable_xxx 字段控制阶段开关
        enable_map = {
            'literature': getattr(research_config, 'enable_literature', True),
            'diagnosis': getattr(research_config, 'enable_diagnosis', True),
            'design': getattr(research_config, 'enable_design', True),
            'experiment': getattr(research_config, 'enable_experiment', True),
        }
        return enable_map.get(phase_name, True)

    def _update_best_result(self, task_id: int, exp_result: Dict[str, Any]):
        """
        更新任务的最佳结果

        比较当前结果与历史最佳，如果更好则更新。
        """
        current_metrics = exp_result.get("metrics", {})
        metric_key = self._get_task_metric_key(task_id, current_metrics)
        current_value = current_metrics.get(metric_key, 0.0)

        best = self._best_results.get(task_id, {})
        best_metrics = best.get("metrics", {})
        best_value = best_metrics.get(metric_key, 0.0)

        if current_value > best_value:
            iteration = self._iteration_counters.get(task_id, 0)
            self._best_results[task_id] = {
                "iteration": iteration,
                "metrics": current_metrics.copy(),
                "output_files": exp_result.get("output_files", [])
            }
            print(f"  >>> 任务{task_id}新的最佳结果: {metric_key}={current_value:.4f}")

    def _log_stop_reason(self, reason: str):
        """打印停止原因"""
        print(f"  [StopCondition] {reason}")

    # ------------------------------------------------------------------
    # Mock对象（当依赖模块不可用时）
    # ------------------------------------------------------------------

    @staticmethod
    def _create_mock_llm():
        """创建Mock LLM客户端"""
        class MockLLM:
            def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
                return ""

            def analyze_results(self, history: list) -> str:
                return ""

            def suggest_next_config(self, task: str, history: list, current_budget: int) -> dict:
                return {}

            def generate_code(self, prompt: str) -> str:
                return ""
        return MockLLM()

    @staticmethod
    def _create_mock_memory():
        """创建Mock记忆系统"""
        class MockMemory:
            def __init__(self):
                self.records = {}

            def add_record(self, task: str, round: int, phase: str,
                          config: dict, metrics: dict, feedback: str, duration: float):
                if task not in self.records:
                    self.records[task] = []
                self.records[task].append({
                    "round": round, "phase": phase, "config": config,
                    "metrics": metrics, "feedback": feedback,
                    "duration": duration, "timestamp": ""
                })

            def get_history(self, task: str) -> list:
                return self.records.get(task, [])

            def get_best_record(self, task: str, metric_key: str = "acc") -> Optional[dict]:
                history = self.get_history(task)
                if not history:
                    return None
                best = None
                best_value = float('-inf')
                for record in history:
                    v = record.get("metrics", {}).get(metric_key, 0)
                    if v > best_value:
                        best_value = v
                        best = record
                return best

            def summarize(self, task: str) -> str:
                return f"任务 '{task}' 的摘要"

            def save(self, path: str):
                pass

            def load(self, path: str):
                pass
        return MockMemory()

    @staticmethod
    def _create_mock_tools():
        """创建Mock工具注册表"""
        class MockTools:
            def run_training(self, **kwargs) -> Dict[str, Any]:
                import random
                task_type = kwargs.get('task_type', 'classification')
                if task_type == 'classification':
                    metrics = {"acc": round(random.uniform(0.70, 0.92), 4)}
                else:
                    metrics = {"mrr": round(random.uniform(0.15, 0.45), 4)}
                return {"metrics": metrics, "output_files": []}

            def inspect_data(self, data_path: str) -> str:
                return f"数据探查: {data_path}"

            def validate_code(self, code: str) -> bool:
                return True

            def smoke_test_model(self, model_path: str) -> bool:
                return True
        return MockTools()


# ---------------------------------------------------------------------------
# 便捷入口
# ---------------------------------------------------------------------------

def run_research(config) -> Dict[str, Any]:
    """
    便捷入口函数

    一行代码启动完整科研流程。

    Args:
        config: SystemConfig实例

    Returns:
        完整结果字典
    """
    orchestrator = ResearchOrchestrator(config)
    return orchestrator.run()


# 如果直接运行此文件，显示帮助信息
if __name__ == "__main__":
    print("ResearchOrchestrator - 自主科研Agent主编排器")
    print("")
    print("使用方式:")
    print("  from orchestrator import ResearchOrchestrator")
    print("  orchestrator = ResearchOrchestrator(config)")
    print("  results = orchestrator.run()")
