"""代码生成器：自主编写/修改PyTorch代码

本模块是科研Agent的核心组件，负责：
1. 根据任务描述从零生成完整的PyTorch项目代码
2. 根据诊断反馈（错误信息、性能瓶颈）修改现有代码
3. 验证生成的代码语法正确性
4. 执行Smoke Test确认模型可运行
5. 管理code/目录下的文件版本

针对金融场景图学习竞赛优化，支持图神经网络（GNN）和推荐系统模型。
"""

import os
import re
import json
import logging
from typing import Optional, Dict, Any, List
from .llm_client import LLMClient
from .tools import ToolRegistry

logger = logging.getLogger(__name__)


class CodeGenerator:
    """代码生成器

    负责根据任务需求自主编写和修改PyTorch代码。

    使用方式:
        llm = LLMClient(config)
        tools = ToolRegistry()
        gen = CodeGenerator(llm, tools, code_dir="./code")

        # 从零生成
        code = gen.generate_from_scratch("classification", "实现一个GNN模型...")
        gen.apply_changes(code)

        # 修改现有代码
        new_code = gen.modify_code("code/models.py", "添加一个注意力层")
        gen.apply_changes({"models.py": new_code})

        # 验证
        results = gen.validate_all()
    """

    # 代码模板骨架，Agent会在此基础上填充具体内容
    CODE_TEMPLATES = {
        "models": """# models.py - 模型定义
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================
# 模型类（Agent会在此处填充）
# ==============================

""",
        "datasets": """# datasets.py - 数据集定义
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# ==============================
# 数据集类（Agent会在此处填充）
# ==============================

def get_dataloader(data_path, batch_size=32, split='train', **kwargs):
    \"\"\"获取数据加载器\"\"\"
    pass  # Agent会实现

""",
        "train": """# train.py - 训练入口
import argparse
import os
import json
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import *
from datasets import get_dataloader
from utils import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_epoch(model, dataloader, optimizer, criterion, device):
    \"\"\"单个epoch的训练\"\"\"
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        pass  # Agent会实现
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    \"\"\"验证\"\"\"
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        pass  # Agent会实现
    return total_loss / len(dataloader), correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, required=True)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)

    # 加载配置
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        for k, v in config.items():
            if not hasattr(args, k) or getattr(args, k) is None:
                setattr(args, k, v)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    logger.info(f"配置: {args}")

    # Agent会在此处填充具体训练逻辑


if __name__ == '__main__':
    main()
""",
        "infer": """# infer.py - 推理入口
import argparse
import os
import json
import torch
import numpy as np
import pandas as pd
from models import *
from datasets import get_dataloader
from utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    device = torch.device(args.device)

    # 加载模型
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # Agent会在此处填充推理逻辑

    # 保存预测结果
    # Agent会实现输出保存


if __name__ == '__main__':
    main()
""",
        "utils": """# utils.py - 工具函数
import torch
import numpy as np
import random
import logging
import os
import json
import time
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

logger = logging.getLogger(__name__)


def set_seed(seed=42):
    \"\"\"设置全局随机种子，保证实验可复现\"\"\"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_metrics(predictions, labels, task_type='classification'):
    \"\"\"计算评估指标\"\"\"
    metrics = {}
    if task_type == 'classification':
        metrics['accuracy'] = accuracy_score(labels, predictions.argmax(axis=-1))
        try:
            metrics['auc'] = roc_auc_score(labels, predictions[:, 1])
        except Exception:
            pass
        metrics['f1'] = f1_score(labels, predictions.argmax(axis=-1), average='macro')
    return metrics


def save_checkpoint(model, optimizer, epoch, metrics, path):
    \"\"\"保存模型检查点\"\"\"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, path)
    logger.info(f"检查点已保存: {path}")


def load_checkpoint(model, path, device='cpu'):
    \"\"\"加载模型检查点\"\"\"
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint


def log_metrics(epoch, train_loss, val_metrics, phase='train'):
    \"\"\"格式化打印训练指标\"\"\"
    msg = f"Epoch {epoch} | train_loss: {train_loss:.4f}"
    if val_metrics:
        for k, v in val_metrics.items():
            msg += f" | val_{k}: {v:.4f}"
    logger.info(msg)

"""
    }

    def __init__(self, llm_client: LLMClient, tools: ToolRegistry,
                 code_dir: str = "./code"):
        """初始化代码生成器

        Args:
            llm_client: LLM客户端，用于代码生成
            tools: 工具注册表，用于代码验证和测试
            code_dir: 代码输出目录
        """
        self.llm = llm_client
        self.tools = tools
        self.code_dir = os.path.expanduser(code_dir)
        os.makedirs(self.code_dir, exist_ok=True)

    # ==================================================================
    # 核心接口
    # ==================================================================

    def generate_from_scratch(self, task_type: str,
                              requirements: str) -> Dict[str, str]:
        """从零开始生成完整代码

        根据任务类型和需求描述，调用LLM生成完整的项目代码。

        Args:
            task_type: "classification"（图分类）, "recommendation"（推荐）,
                      "link_prediction"（链接预测）等
            requirements: 详细需求描述，如数据集信息、模型要求、评估指标等

        Returns:
            {"models.py": "...", "datasets.py": "...", "train.py": "...",
             "infer.py": "...", "utils.py": "..."}
        """
        logger.info(f"开始从零生成代码: task_type={task_type}")

        # 构建生成提示
        prompt = self._build_generation_prompt(task_type, requirements)

        # 调用LLM生成代码
        try:
            response = self.llm.generate_code(prompt)
        except Exception as e:
            logger.error(f"LLM代码生成失败: {e}")
            # 返回模板骨架作为后备
            return self._fallback_templates(task_type)

        # 解析LLM响应，提取各文件代码
        code_files = self._parse_code_response(response)

        # 补充缺失文件（使用模板）
        for fname, template in self.CODE_TEMPLATES.items():
            py_fname = f"{fname}.py"
            if py_fname not in code_files:
                code_files[py_fname] = template

        logger.info(f"代码生成完成，共 {len(code_files)} 个文件")
        return code_files

    def modify_code(self, file_path: str,
                    modification_request: str) -> str:
        """修改现有代码

        读取指定文件，根据修改需求调用LLM生成修改后的代码。

        Args:
            file_path: 要修改的文件路径（相对code_dir或绝对路径）
            modification_request: 修改需求描述，如"添加一个注意力机制层"

        Returns:
            修改后的完整代码字符串
        """
        # 读取现有代码
        full_path = os.path.join(self.code_dir, file_path) if not os.path.isabs(file_path) else file_path

        read_result = self.tools.call("file_read", path=full_path)
        if not read_result.get("exists"):
            logger.error(f"无法读取文件: {full_path}")
            raise FileNotFoundError(f"无法读取文件: {full_path}")

        existing_code = read_result["content"]

        # 构建修改提示
        prompt = self._build_modification_prompt(
            file_path, existing_code, modification_request
        )

        # 调用LLM生成修改后的代码
        try:
            response = self.llm.generate_code(prompt)
        except Exception as e:
            logger.error(f"LLM代码修改失败: {e}")
            raise

        # 从响应中提取代码
        modified_code = self._extract_code_from_response(response, file_path)

        if not modified_code:
            logger.warning("LLM未返回有效代码，保留原代码")
            return existing_code

        # 验证修改后代码的语法
        validate_result = self.tools.call("validate_code", code=modified_code)
        if not validate_result.get("valid"):
            logger.warning(f"修改后的代码存在语法错误: {validate_result.get('errors')}")
            # 尝试修复简单语法错误
            fixed_code = self._attempt_fix_syntax(modified_code, validate_result)
            if fixed_code:
                modified_code = fixed_code

        return modified_code

    def apply_changes(self, changes: Dict[str, str]):
        """应用代码变更到code/目录

        将修改后的代码写入文件系统，并验证语法。

        Args:
            changes: {文件名: 新代码内容} 字典
        """
        logger.info(f"应用代码变更，共 {len(changes)} 个文件")

        for fname, code in changes.items():
            # 清理文件名
            fname = fname.strip()
            if not fname.endswith('.py'):
                fname += '.py'

            fpath = os.path.join(self.code_dir, fname)

            # 写入文件
            result = self.tools.call("file_write", path=fpath, content=code)
            if result.get("success"):
                logger.info(f"  已写入: {fpath} ({result.get('size', 0)} 字节)")
            else:
                logger.error(f"  写入失败: {fpath} - {result.get('error')}")

    def validate_all(self) -> Dict[str, bool]:
        """验证code/目录下所有Python代码

        使用AST语法检查验证所有.py文件。

        Returns:
            {"models.py": True, "train.py": False, ...}
        """
        logger.info("开始验证code/目录下所有代码...")

        results = {}
        list_result = self.tools.call("list_files", directory=self.code_dir, pattern="*.py")

        if not list_result.get("files"):
            logger.error(f"无法列出代码文件: {list_result.get('error', '未知错误')}")
            return results

        for detail in list_result.get("files", []):
            fname = detail["name"]
            fpath = detail["path"]

            # 读取并验证
            read_result = self.tools.call("file_read", path=fpath)
            if not read_result.get("success"):
                results[fname] = False
                continue

            validate_result = self.tools.call("validate_code", code=read_result["content"])
            is_valid = validate_result.get("valid", False)
            results[fname] = is_valid

            status = "通过" if is_valid else "失败"
            logger.info(f"  {fname}: {status}")

        return results

    def get_current_code(self) -> Dict[str, str]:
        """获取当前code/目录下的所有代码

        Returns:
            {"models.py": "...", "train.py": "...", ...}
        """
        code = {}
        list_result = self.tools.call("list_files", directory=self.code_dir, pattern="*.py")

        if not list_result.get("files"):
            return code

        for detail in list_result.get("files", []):
            fname = detail["name"]
            fpath = detail["path"]

            read_result = self.tools.call("file_read", path=fpath)
            if read_result.get("success"):
                code[fname] = read_result["content"]

        return code

    # ==================================================================
    # 辅助方法
    # ==================================================================

    def _build_generation_prompt(self, task_type: str,
                                  requirements: str,
                                  existing_code: str = "") -> str:
        """构建代码生成的prompt

        组装一个详细的prompt，指导LLM生成高质量的PyTorch代码。

        Args:
            task_type: 任务类型
            requirements: 需求描述
            existing_code: 已有代码（增量生成时使用）

        Returns:
            完整的prompt字符串
        """
        task_descriptions = {
            "classification": """图分类任务：对金融场景图数据进行分类预测。
- 输入：图结构数据（节点特征 + 边关系）
- 输出：每个图的类别标签
- 适用模型：GCN, GAT, GraphSAGE, GIN等
- 评估指标：准确率(Accuracy), AUC, F1""",
            "recommendation": """推荐任务：预测用户对金融产品的偏好。
- 输入：用户-物品交互图
- 输出：用户对物品的评分或点击概率
- 适用模型：LightGCN, NGCF, GraphSAGE
- 评估指标：AUC, Recall@K, NDCG@K""",
            "link_prediction": """链接预测任务：预测金融实体间是否存在关系。
- 输入：图结构数据
- 输出：实体对之间存在边的概率
- 适用模型：GAE, VGAE, SEAL
- 评估指标：AUC, AP""",
            "node_classification": """节点分类任务：对图中节点进行分类。
- 输入：节点特征 + 图结构
- 输出：节点类别
- 适用模型：GCN, GAT, GraphSAGE
- 评估指标：准确率, 宏F1"""
        }

        task_desc = task_descriptions.get(task_type, task_type)

        prompt = f"""你是一个专业的PyTorch深度学习工程师。请为金融场景图学习竞赛编写完整的、可直接运行的Python代码。

## 任务类型
{task_desc}

## 需求描述
{requirements}

## 要求

1. **代码完整性**：必须生成以下5个文件的所有代码，每个文件都必须是完整可运行的：
   - `models.py`: 模型定义（包含至少一个nn.Module子类）
   - `datasets.py`: 数据集定义（包含get_dataloader函数）
   - `train.py`: 训练脚本（包含完整的训练循环）
   - `infer.py`: 推理脚本（包含完整的预测逻辑）
   - `utils.py`: 工具函数（包含指标计算、检查点保存等）

2. **代码质量**：
   - 使用中文注释说明关键逻辑
   - 导入语句完整，不要遗漏依赖
   - 处理边界情况（空图、单节点等）
   - 使用logging而不是print输出信息

3. **金融场景适配**：
   - 支持大规模稀疏图数据
   - 考虑类别不平衡问题（加权损失或采样策略）
   - 支持GPU加速（device参数）

4. **输出格式**：
   使用以下格式分别输出每个文件：

   ```python
   # filename: models.py
   <代码内容>
   ```

   ```python
   # filename: datasets.py
   <代码内容>
   ```

   （依此类推，所有5个文件）
"""

        if existing_code:
            prompt += f"\n\n## 已有代码（请在此基础上修改）\n```python\n{existing_code[:3000]}\n```"

        return prompt

    def _build_modification_prompt(self, file_path: str,
                                    existing_code: str,
                                    modification_request: str) -> str:
        """构建代码修改的prompt

        Args:
            file_path: 目标文件路径
            existing_code: 当前代码内容
            modification_request: 修改需求

        Returns:
            完整的修改prompt
        """
        return f"""你是一个专业的PyTorch工程师。请修改以下代码文件。

## 文件路径
{file_path}

## 当前代码
```python
{existing_code}
```

## 修改需求
{modification_request}

## 要求
1. 返回修改后的**完整**代码（不是diff），必须可以直接保存为.py文件运行
2. 保持原有代码结构和风格
3. 添加中文注释说明修改的部分
4. 确保导入语句完整

## 输出格式
```python
# filename: {os.path.basename(file_path)}
<修改后的完整代码>
```
"""

    def _parse_code_response(self, response: str) -> Dict[str, str]:
        """从LLM响应中提取各文件代码

        支持多种代码块格式：
        - ```python\n# filename: xxx.py\n...\n```
        - ```python\n# xxx.py\n...\n```
        - <file name="xxx.py">...</file>

        Args:
            response: LLM的原始响应文本

        Returns:
            {"models.py": "...", ...}
        """
        code_files = {}

        # 方法1: 匹配 ```python\n# filename: xxx.py\n...\n```
        pattern1 = r'```python\s*\n#\s*filename:\s*(\S+\.py)\s*\n(.*?)\n```'
        matches = re.findall(pattern1, response, re.DOTALL)
        for fname, code in matches:
            code_files[fname.strip()] = code.strip()

        # 方法2: 匹配 ```python\n# xxx.py\n...\n```
        if not code_files:
            pattern2 = r'```python\s*\n#\s*(\S+\.py)\s*\n(.*?)\n```'
            matches = re.findall(pattern2, response, re.DOTALL)
            for fname, code in matches:
                code_files[fname.strip()] = code.strip()

        # 方法3: 匹配 <file name="xxx.py">...</file>
        if not code_files:
            pattern3 = r'<file\s+name="([^"]+\.py)"\s*>(.*?)</file>'
            matches = re.findall(pattern3, response, re.DOTALL)
            for fname, code in matches:
                code_files[fname.strip()] = code.strip()

        # 方法4: 如果没有明确标记，尝试按代码块分割
        if not code_files:
            blocks = re.split(r'```python\s*\n', response)
            for block in blocks[1:]:  # 跳过第一个（在```python之前的文本）
                code_end = block.find('```')
                if code_end == -1:
                    continue
                code = block[:code_end].strip()

                # 尝试从代码注释中提取文件名
                fname_match = re.search(r'#\s*(\w+\.py)', code[:200])
                if fname_match:
                    fname = fname_match.group(1)
                    code_files[fname] = code

        logger.info(f"从响应中解析出 {len(code_files)} 个文件: {list(code_files.keys())}")
        return code_files

    def _extract_code_from_response(self, response: str,
                                     expected_file: str) -> Optional[str]:
        """从LLM响应中提取单个文件的代码

        Args:
            response: LLM响应文本
            expected_file: 期望的文件名

        Returns:
            代码字符串，未找到时返回None
        """
        # 先尝试完整解析
        parsed = self._parse_code_response(response)
        basename = os.path.basename(expected_file)

        if basename in parsed:
            return parsed[basename]

        # 回退：提取第一个python代码块
        match = re.search(r'```python\s*\n(.*?)\n```', response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # 最后尝试：如果响应本身就是纯代码
        if not response.strip().startswith('#') and 'import' in response:
            return response.strip()

        return None

    def _attempt_fix_syntax(self, code: str,
                            validate_result: Dict[str, Any]) -> Optional[str]:
        """尝试修复简单的语法错误

        目前支持的修复：
        - 缺少缩进
        - 括号不匹配（简单情况）
        - 多余的空行

        Args:
            code: 原始代码
            validate_result: 验证结果（含错误信息）

        Returns:
            修复后的代码，无法修复时返回None
        """
        errors = validate_result.get("errors", [])
        fixed = code

        for error in errors:
            if isinstance(error, dict):
                msg = error.get("message", "")
                lineno = error.get("lineno")
            else:
                msg = str(error)
                lineno = None

            # 尝试修复常见错误
            if "unexpected indent" in msg.lower():
                # 移除意外缩进（尝试去除该行前导空格）
                if lineno:
                    lines = fixed.split('\n')
                    if 0 < lineno <= len(lines):
                        lines[lineno - 1] = lines[lineno - 1].lstrip()
                        fixed = '\n'.join(lines)

        # 再次验证
        second_check = self.tools.call("validate_code", code=fixed)
        if second_check.get("valid"):
            logger.info("语法错误已自动修复")
            return fixed

        return None

    def _fallback_templates(self, task_type: str) -> Dict[str, str]:
        """当LLM生成失败时，返回模板骨架代码

        Args:
            task_type: 任务类型

        Returns:
            包含基础模板代码的字典
        """
        logger.warning("LLM代码生成失败，使用模板骨架作为后备")

        # 根据任务类型定制模型模板
        if task_type == "recommendation":
            model_code = '''# models.py - 推荐模型
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    """LightGCN推荐模型 - 简化版"""
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, users, items):
        """前向传播，计算用户-物品交互分数"""
        u_emb = self.user_embedding(users)
        i_emb = self.item_embedding(items)
        scores = (u_emb * i_emb).sum(dim=-1)
        return torch.sigmoid(scores)
'''
        elif task_type == "link_prediction":
            model_code = '''# models.py - 链接预测模型
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNEncoder(nn.Module):
    """GCN编码器用于链接预测"""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = nn.Linear(in_channels, hidden_channels)
        self.conv2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index):
        """编码节点特征"""
        # 简单的消息传递模拟
        row, col = edge_index
        out = torch.zeros_like(x)
        out.index_add_(0, row, x[col])
        out = out + x  # 自连接
        deg = torch.bincount(row, minlength=x.size(0)).float().clamp(min=1)
        out = out / deg.view(-1, 1)

        out = F.relu(self.conv1(out))
        out = self.dropout(out)
        out = self.conv2(out)
        return out


class LinkPredictor(nn.Module):
    """链接预测模型"""
    def __init__(self, in_channels, hidden_channels):
        super(LinkPredictor, self).__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels, hidden_channels)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index, edge_label_index):
        """预测边是否存在"""
        z = self.encoder(x, edge_index)
        src, dst = edge_label_index
        z_src = z[src]
        z_dst = z[dst]
        z_pair = torch.cat([z_src, z_dst], dim=-1)
        return torch.sigmoid(self.predictor(z_pair).squeeze(-1))
'''
        else:
            # 默认图分类模型
            model_code = '''# models.py - 图分类模型
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGNN(nn.Module):
    """简单图神经网络用于图分类"""
    def __init__(self, in_channels, hidden_channels=64, num_classes=2, num_layers=3):
        super(SimpleGNN, self).__init__()
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(nn.Linear(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels // 2, num_classes)
        )

    def forward(self, x, edge_index, batch=None):
        """前向传播
        
        Args:
            x: 节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
            batch: 图batch分配 [num_nodes]（可选，单图时为None）
        """
        # 简化的消息传递
        row, col = edge_index

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            # 聚合邻居特征
            out = torch.zeros_like(x)
            out.index_add_(0, row, x[col])
            out = out + x  # 自连接
            deg = torch.bincount(row, minlength=x.size(0)).float().clamp(min=1)
            out = out / deg.view(-1, 1)

            x = conv(out)
            x = bn(x)
            x = F.relu(x)

        # 全局平均池化（将所有节点特征平均得到图级别表示）
        if batch is not None:
            # 多图batch处理
            num_graphs = batch.max().item() + 1
            graph_feat = torch.zeros(num_graphs, x.size(1), device=x.device)
            graph_feat.index_add_(0, batch, x)
            count = torch.bincount(batch, minlength=num_graphs).float().clamp(min=1).view(-1, 1)
            x = graph_feat / count
        else:
            # 单图
            x = x.mean(dim=0, keepdim=True)

        return self.classifier(x)
'''

        dataset_code = '''# datasets.py - 数据集定义
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


class GraphDataset(Dataset):
    """图数据集基类"""
    def __init__(self, data_path, split='train'):
        super(GraphDataset, self).__init__()
        self.data_path = data_path
        self.split = split
        self.graphs = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        """加载图数据"""
        # 请根据实际数据格式修改
        if self.data_path.endswith('.npz'):
            data = np.load(self.data_path, allow_pickle=True)
            self.graphs = data.get('graphs', [])
            self.labels = data.get('labels', [])
        elif self.data_path.endswith('.pt'):
            loaded = torch.load(self.data_path)
            self.graphs = loaded.get('graphs', [])
            self.labels = loaded.get('labels', [])
        else:
            raise ValueError(f"不支持的数据格式: {self.data_path}")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]


def get_dataloader(data_path, batch_size=32, split='train', **kwargs):
    """获取数据加载器"""
    dataset = GraphDataset(data_path, split=split)
    shuffle = (split == 'train')
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                     num_workers=kwargs.get('num_workers', 0))
'''

        train_code = '''# train.py - 训练入口
import argparse
import os
import json
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score

from models import *
from datasets import get_dataloader
from utils import set_seed, save_checkpoint, log_metrics

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_epoch(model, dataloader, optimizer, criterion, device):
    """单个epoch的训练"""
    model.train()
    total_loss = 0.0
    for batch_idx, (data, labels) in enumerate(dataloader):
        data = data.to(device) if hasattr(data, 'to') else data
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device) if hasattr(data, 'to') else data
            labels = labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.softmax(outputs, dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels.numpy(), all_preds.argmax(dim=-1).numpy())

    metrics = {'loss': avg_loss, 'accuracy': acc}
    try:
        metrics['auc'] = roc_auc_score(all_labels.numpy(), all_preds[:, 1].numpy())
    except Exception:
        pass

    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, required=True)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # 加载配置
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        for k, v in config.items():
            if not hasattr(args, k) or getattr(args, k) is None:
                setattr(args, k, v)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    logger.info(f"配置: {args}")

    # 加载数据
    # train_loader = get_dataloader(..., split='train', batch_size=args.batch_size)
    # val_loader = get_dataloader(..., split='val', batch_size=args.batch_size)

    # 创建模型
    # model = SimpleGNN(...).to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 训练循环
    best_metric = 0.0
    for epoch in range(1, args.epochs + 1):
        # train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        # val_loss, val_metrics = validate(model, val_loader, criterion, device)
        # log_metrics(epoch, train_loss, val_metrics)

        # if val_metrics.get('accuracy', 0) > best_metric:
        #     best_metric = val_metrics['accuracy']
        #     save_checkpoint(model, optimizer, epoch, val_metrics,
        #                     os.path.join(args.output_dir, 'best.pt'))
        pass  # 请根据实际模型和数据集实现

    logger.info(f"训练完成，最佳指标: {best_metric:.4f}")


if __name__ == '__main__':
    main()
'''

        infer_code = '''# infer.py - 推理入口
import argparse
import os
import torch
import numpy as np
import pandas as pd

from models import *
from utils import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    device = torch.device(args.device)

    # 加载模型
    # model = SimpleGNN(...).to(device)
    # checkpoint = load_checkpoint(model, args.checkpoint, device)
    # model.eval()

    # 加载测试数据
    # test_data = ...

    # 推理
    # with torch.no_grad():
    #     predictions = model(test_data)

    # 保存结果
    # df = pd.DataFrame({'prediction': predictions.cpu().numpy()})
    # df.to_csv(args.output, index=False)

    print(f"推理完成，结果保存至: {args.output}")


if __name__ == '__main__':
    main()
'''

        utils_code = self.CODE_TEMPLATES["utils"]

        return {
            "models.py": model_code,
            "datasets.py": dataset_code,
            "train.py": train_code,
            "infer.py": infer_code,
            "utils.py": utils_code
        }

    def smoke_test(self, model_file: str = "models.py",
                   task_type: str = "classification") -> Dict[str, Any]:
        """对模型执行Smoke Test

        便捷方法：验证code/目录下的指定模型文件。

        Args:
            model_file: 模型文件名（在code_dir下）
            task_type: 任务类型

        Returns:
            Smoke Test结果字典
        """
        model_path = os.path.join(self.code_dir, model_file)
        return self.tools.call("smoke_test_model", code_path=model_path,
                               task_type=task_type)
