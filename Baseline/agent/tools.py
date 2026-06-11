"""工具注册表 - 统一管理Agent可使用的工具"""
import os
import re
import ast
import json
import subprocess
import tempfile
import logging
from typing import Dict, Any, Callable, Optional, List

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    工具注册表

    注册和管理Agent可使用的所有工具。
    每个工具是一个函数，接受kwargs参数，返回结果字典。
    """

    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """注册默认工具集"""
        self.register("inspect_data", self._inspect_data)
        self.register("summarize_code", self._summarize_code)
        self.register("validate_code", self._validate_code)
        self.register("smoke_test_model", self._smoke_test_model)
        self.register("run_training", self._run_training)
        self.register("run_inference", self._run_inference)
        self.register("analyze_log", self._analyze_log)
        self.register("file_read", self._file_read)
        self.register("file_write", self._file_write)
        self.register("shell_exec", self._shell_exec)
        self.register("list_files", self._list_files)

    def register(self, name: str, func: Callable):
        """注册工具"""
        self.tools[name] = func

    def call(self, name: str, **kwargs) -> Dict[str, Any]:
        """调用工具"""
        if name not in self.tools:
            return {"error": f"工具 '{name}' 未注册"}
        try:
            return self.tools[name](**kwargs)
        except Exception as e:
            logger.error(f"工具 '{name}' 执行失败: {e}")
            return {"error": str(e)}

    def list_tools(self) -> List[str]:
        """列出所有可用工具"""
        return list(self.tools.keys())

    # ===== 具体工具实现 =====

    def _inspect_data(self, path: str, data_type: str = "auto") -> Dict[str, Any]:
        """探查数据结构，支持npz/csv/json"""
        result = {"path": path, "exists": os.path.exists(path)}
        if not result["exists"]:
            return result

        if path.endswith(".npz") or data_type == "npz":
            return self._inspect_npz(path)
        elif path.endswith(".csv") or data_type == "csv":
            return self._inspect_csv(path)
        elif path.endswith(".json") or data_type == "json":
            return self._inspect_json(path)
        else:
            result["type"] = "unknown"
            result["size"] = os.path.getsize(path)
            return result

    def _inspect_npz(self, path: str) -> Dict[str, Any]:
        """探查npz文件"""
        import numpy as np
        import scipy.sparse as sp
        data = np.load(path)
        info = {"type": "npz", "keys": list(data.keys())}
        for key in data.keys():
            arr = data[key]
            if isinstance(arr, np.ndarray):
                info[f"{key}_shape"] = arr.shape
                info[f"{key}_dtype"] = str(arr.dtype)
            else:
                info[f"{key}_type"] = type(arr).__name__
        # 尝试识别邻接矩阵和特征矩阵
        if 'adj_shape' in info:
            info["num_nodes"] = int(data['adj_shape'][0])
        if 'attr_shape' in info:
            info["num_features"] = int(data['attr_shape'][1])
        if 'labels' in data:
            labels = data['labels']
            info["num_classes"] = int(labels[labels >= 0].max()) + 1 if len(labels) > 0 else 0
        return info

    def _inspect_csv(self, path: str) -> Dict[str, Any]:
        """探查csv文件"""
        import pandas as pd
        df = pd.read_csv(path, nrows=5)
        info = {
            "type": "csv",
            "shape": pd.read_csv(path).shape,
            "columns": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "sample": df.head(3).to_dict()
        }
        return info

    def _inspect_json(self, path: str) -> Dict[str, Any]:
        """探查json文件"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {"type": "json", "keys": list(data.keys()) if isinstance(data, dict) else "list"}

    def _summarize_code(self, file_path: str) -> Dict[str, Any]:
        """代码摘要分析"""
        result = {"path": file_path, "exists": os.path.exists(file_path)}
        if not result["exists"]:
            return result
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        try:
            tree = ast.parse(source)
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            imports = [ast.dump(node) for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
            result.update({
                "classes": classes,
                "functions": functions,
                "num_lines": len(source.splitlines()),
                "valid_syntax": True
            })
        except SyntaxError as e:
            result.update({"valid_syntax": False, "error": str(e)})
        return result

    def _validate_code(self, code: str) -> Dict[str, Any]:
        """Python语法验证"""
        try:
            ast.parse(code)
            return {"valid": True, "errors": []}
        except SyntaxError as e:
            return {"valid": False, "errors": [str(e)]}

    def _smoke_test_model(self, code_path: str, task_type: str = "classification") -> Dict[str, Any]:
        """模型Smoke Test - 导入模型并执行前向传播"""
        import sys
        import torch

        result = {"passed": False, "error": "", "output_shape": ""}
        dir_path = os.path.dirname(os.path.abspath(code_path))
        if dir_path not in sys.path:
            sys.path.insert(0, dir_path)

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("test_models", code_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if task_type == "classification":
                if hasattr(module, 'GNNClassifier'):
                    model = module.GNNClassifier(16, 32, 5, num_layers=2)
                    x = torch.randn(10, 16)
                    adj = torch.eye(10)
                    out = model(x, adj)
                    result["passed"] = True
                    result["output_shape"] = str(out.shape)
                else:
                    result["error"] = "未找到GNNClassifier类"
            elif task_type == "recommendation":
                if hasattr(module, 'GRU4Rec'):
                    model = module.GRU4Rec(100, embedding_dim=32, hidden_dim=64)
                    seq = torch.randint(0, 100, (4, 10))
                    out = model(seq)
                    result["passed"] = True
                    result["output_shape"] = str(out.shape)
                else:
                    result["error"] = "未找到GRU4Rec类"
        except Exception as e:
            result["error"] = str(e)
        return result

    def _run_training(self, task_type: str, data_path: str, code_dir: str,
                     output_dir: str, iteration: int = 1,
                     config: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行训练脚本

        Args:
            task_type: 任务类型 ('task1' 或 'task2')
            data_path: 数据路径
            code_dir: 代码目录
            output_dir: 输出目录
            iteration: 迭代轮次
            config: 额外配置参数

        Returns:
            {"metrics": {...}, "output_files": [...], "returncode": 0}
        """
        config = config or {}
        # 根据任务类型添加默认 model_type（如果未指定）
        if 'model_type' not in config:
            if task_type == 'task1':
                config = {**config, 'model_type': 'sage'}
            elif task_type == 'task2':
                config = {**config, 'model_type': 'gru4rec'}
        cmd = ["python", os.path.join(code_dir, "train.py"), "--task", str(task_type),
               "--data_path", data_path, "--output_dir", output_dir]
        for k, v in config.items():
            cmd.extend([f"--{k}", str(v)])
        exec_result = self._shell_exec(cmd, timeout=1800)

        result = {
            "metrics": {},
            "output_files": [],
            "returncode": exec_result.get("returncode", -1),
            "stdout": exec_result.get("stdout", ""),
            "stderr": exec_result.get("stderr", "")
        }

        # 尝试读取验证指标
        if exec_result.get("returncode") == 0:
            metrics_paths = [
                os.path.join(output_dir, f"metrics_iter{iteration}.json"),
                os.path.join(output_dir, "metrics.json"),
            ]
            for mp in metrics_paths:
                if os.path.exists(mp):
                    try:
                        with open(mp, 'r', encoding='utf-8') as f:
                            loaded = json.load(f)
                        # MetricsTracker 保存的是 history 字典，取最后一轮
                        if isinstance(loaded, dict):
                            result["metrics"] = {k: (v[-1] if isinstance(v, list) and v else v)
                                                 for k, v in loaded.items()}
                    except Exception:
                        pass
                    break

            # 检查输出文件
            task_num = 1 if task_type in ('task1', 'classification') else 2
            output_file = os.path.join(output_dir, f"A{task_num}.csv")
            if os.path.exists(output_file):
                result["output_files"].append(output_file)

        return result

    def _run_inference(self, code_dir: str, task_id: int, checkpoint_path: str,
                      output_path: str) -> Dict[str, Any]:
        """执行推理脚本"""
        cmd = ["python", os.path.join(code_dir, "infer.py"), "--task", str(task_id),
               "--checkpoint", checkpoint_path, "--output_path", output_path]
        return self._shell_exec(cmd, timeout=300)

    def _analyze_log(self, log_path: str) -> Dict[str, Any]:
        """训练日志分析"""
        result = {"epochs": 0, "train_losses": [], "val_metrics": [], "overfitting_epoch": -1}
        if not os.path.exists(log_path):
            return result
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        losses = []
        metrics = []
        for line in lines:
            # 匹配 loss 值
            loss_match = re.search(r'loss[:\s]+([0-9.]+)', line, re.IGNORECASE)
            if loss_match:
                losses.append(float(loss_match.group(1)))
            # 匹配 acc/ndcg/mrr
            metric_match = re.search(r'(acc|ndcg|mrr)[:\s]+([0-9.]+)', line, re.IGNORECASE)
            if metric_match:
                metrics.append(float(metric_match.group(2)))

        result["train_losses"] = losses
        result["val_metrics"] = metrics
        result["epochs"] = len(losses)

        # 过拟合检测：验证指标连续下降
        if len(metrics) >= 3:
            for i in range(2, len(metrics)):
                if metrics[i] < metrics[i-1] < metrics[i-2]:
                    result["overfitting_epoch"] = i
                    break
        return result

    def _file_read(self, path: str, limit: int = 1000) -> Dict[str, Any]:
        """读取文件"""
        if not os.path.exists(path):
            return {"exists": False, "content": ""}
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:limit]
        return {"exists": True, "content": "".join(lines), "total_lines": len(lines)}

    def _file_write(self, path: str, content: str) -> Dict[str, Any]:
        """写入文件"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return {"success": True, "path": path, "size": len(content)}

    def _shell_exec(self, command, timeout: int = 300) -> Dict[str, Any]:
        """执行Shell命令

        Args:
            command: 字符串命令或命令列表
            timeout: 超时秒数
        """
        try:
            if isinstance(command, list):
                proc = subprocess.run(
                    command, capture_output=True, text=True,
                    timeout=timeout
                )
            else:
                proc = subprocess.run(
                    command, shell=True, capture_output=True, text=True,
                    timeout=timeout
                )
            return {
                "returncode": proc.returncode,
                "stdout": proc.stdout[:5000],
                "stderr": proc.stderr[:2000]
            }
        except subprocess.TimeoutExpired:
            return {"returncode": -1, "stdout": "", "stderr": f"超时({timeout}s)"}

    def _list_files(self, directory: str, pattern: str = "*") -> Dict[str, Any]:
        """列出目录中的文件"""
        import glob
        files = glob.glob(os.path.join(directory, pattern))
        return {
            "directory": directory,
            "files": [{"path": f, "size": os.path.getsize(f), "name": os.path.basename(f)}
                      for f in files if os.path.isfile(f)]
        }
