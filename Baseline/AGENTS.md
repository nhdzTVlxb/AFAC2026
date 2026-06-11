# 自主科研Agent框架 - AGENTS.md

> 本文件面向AI Coding Agent读者。项目所有注释和文档均使用中文，Agent在处理代码时应保持中文注释风格。

---

## 项目概述

本项目是一个**端到端自主科研Agent框架**，面向金融场景图学习任务的自动化实验控制竞赛。系统在零人工干预下完成"文献解析 → 瓶颈诊断 → 代码设计 → 实验验证 → 提交打包"的完整科研闭环。

项目包含两个子任务：
- **Task 1 (A1)**: 产品分类 —— 图节点分类任务，使用GNN模型（GCN / GraphSAGE / GAT）
- **Task 2 (A2)**: 产品推荐 —— 序列推荐任务，使用GRU4Rec / SASRec模型

### 目录结构

```
├── framework/              # 项目主代码
│   ├── agent/              # Agent核心系统
│   │   ├── __init__.py     # 包导出
│   │   ├── config.py       # 配置管理（LLMConfig / ResearchConfig / TaskConfig）
│   │   ├── orchestrator.py # 主编排器：四阶段调度、停止条件、打包提交
│   │   ├── phases.py       # 四阶段实现：Literature / Diagnosis / Design / Experiment
│   │   ├── memory.py       # 研究记忆系统：实验记录持久化
│   │   ├── llm_client.py   # OpenAI兼容LLM客户端
│   │   ├── tools.py        # 工具注册表：文件读写、Shell执行、代码验证等
│   │   └── code_generator.py # 代码生成器：自主编写/修改PyTorch代码
│   ├── code/               # 自主生成的模型与训练代码（Agent可修改）
│   │   ├── models.py       # GNNClassifier + GRU4Rec + SASRec模型定义
│   │   ├── datasets.py     # GraphDataset + RecDataset数据集定义
│   │   ├── train.py        # 训练入口：支持分类和推荐两种任务
│   │   ├── infer.py        # 推理入口：生成预测并输出提交格式
│   │   └── utils.py        # 工具函数：邻接矩阵处理、序列处理、评估指标
│   ├── main.py             # CLI主入口
│   ├── config.yaml         # YAML配置文件示例
│   └── requirements.txt    # Python依赖
├── .venv/                  # Python虚拟环境（已初始化）
├── SPEC_v2.md              # 系统架构规格说明（中文）
└── plan.md                 # 竞赛代码构建计划（中文）
```

---

## 技术栈

- **语言**: Python 3
- **深度学习框架**: PyTorch >= 2.0.0
- **科学计算**: numpy >= 1.24.0, scipy >= 1.10.0, pandas >= 2.0.0
- **机器学习**: scikit-learn >= 1.3.0
- **配置**: PyYAML >= 6.0
- **API调用**: requests >= 2.31.0（OpenAI兼容API）
- **环境管理**: 使用 `.venv/` 目录下的虚拟环境

---

## 构建与运行命令

### 环境准备

虚拟环境已存在于 `.venv/` 目录下。激活方式：

```bash
# Windows (Git Bash)
source .venv/Scripts/activate

# Windows (CMD)
.venv\Scripts\activate.bat

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

依赖安装（如需要）：
```bash
pip install -r framework/requirements.txt
```

### 运行主程序

```bash
cd framework

# 完整运行两个任务
python main.py --task 1 --task 2 --budget 10

# 仅运行任务1（分类）
python main.py --task 1 --budget 5

# 从配置文件加载
python main.py --config config.yaml

# 恢复之前运行
python main.py --resume ./output/task1/research_memory.json
```

### 独立运行训练/推理脚本

```bash
cd framework/code

# Task 1: GNN节点分类训练
python train.py --task task1 --data_path data/cls_data/A1.npz --model_type sage \
    --hidden_dim 128 --num_layers 2 --lr 0.01 --epochs 200 \
    --output_dir ../output/task1

# Task 1: 推理
python infer.py --task task1 --data_path data/cls_data/A1.npz \
    --checkpoint ../output/task1/best_model.pt --output_path A1.csv

# Task 2: 序列推荐训练
python train.py --task task2 --data_path data/task2/ --model_type gru4rec \
    --embedding_dim 64 --hidden_dim 128 --lr 0.001 --epochs 50 \
    --output_dir ../output/task2

# Task 2: 推理
python infer.py --task task2 --data_path data/task2/ \
    --checkpoint ../output/task2/best_model.pt --output_path A2.csv
```

---

## 代码风格指南

### 注释语言
- **所有注释和文档字符串必须使用中文**。这是项目统一规范，不可改为英文。
- 代码中的字符串输出（如日志、用户提示）也使用中文。

### 命名规范
- 类名：`PascalCase`（如 `GNNClassifier`, `ResearchOrchestrator`）
- 函数/变量：`snake_case`（如 `hidden_dim`, `train_task1`）
- 常量：`UPPER_SNAKE_CASE`（如 `DECISION_CONTINUE`）
- 私有方法：以 `_` 开头（如 `_check_time_limit`）

### 文档字符串风格
- 使用 `"""三重双引号"""` 包裹模块和类文档字符串
- 函数文档字符串包含：功能描述、Args、Returns
- 示例：
  ```python
  def normalize_adj(adj):
      """对称归一化邻接矩阵: D^(-1/2) * (A + I) * D^(-1/2)

      Args:
          adj: 邻接矩阵, scipy稀疏矩阵或numpy数组或torch.Tensor

      Returns:
          归一化后的邻接矩阵, torch.Tensor (N, N)
      """
  ```

### 代码组织
- 每个 `.py` 文件顶部有模块级文档字符串，说明职责
- 使用 `===== 章节标题 =====` 风格的注释分隔不同功能区块
- 导入顺序：标准库 → 第三方库 → 项目内部模块
- 避免循环导入，必要时使用延迟导入（函数内 `import`）

### 日志规范
- 使用 `logging` 模块，不使用 `print`（Agent系统本身使用 `print` 输出阶段信息，这是设计选择）
- 训练/推理代码中使用 `logging.info()` / `logging.warning()`
- 日志格式：`[%(asctime)s] %(levelname)s - %(message)s`

---

## 核心架构说明

### 四阶段科研闭环

```
┌─────────────────────────────────────────────────────────────┐
│                    ResearchOrchestrator                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │Literature│→ │Diagnosis │→ │  Design  │→ │Experiment│   │
│  │  Phase   │  │  Phase   │  │  Phase   │  │  Phase   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│         ↑_________________________________________↓         │
│                      (迭代闭环)                              │
└─────────────────────────────────────────────────────────────┘
```

1. **LiteraturePhase**: 读取README、探查数据结构（npz/csv）、审查现有代码、调用LLM生成综合分析报告
2. **DiagnosisPhase**: 分析历史实验记录、识别性能瓶颈、提出1-3个可验证优化假设
3. **DesignPhase**: 根据假设调用LLM生成代码修改、Python语法验证、Smoke Test（前向传播测试）
4. **ExperimentPhase**: 执行训练脚本、评估指标、LLM分析结果并决策（CONTINUE / PIVOT / STOP）

### 关键设计决策

- **配置管理**: 使用 `dataclasses` + YAML，支持命令行参数覆盖。配置类在 `agent/config.py` 中定义。
- **记忆系统**: `ResearchMemory` 将实验记录序列化为 JSON，保存到 `./output/task{N}/research_memory.json`。
- **LLM客户端**: `LLMClient` 支持 OpenAI 兼容 API，自动记录调用历史。API密钥从环境变量 `LLM_API_KEY` / `LLM_BASE_URL` 读取。
- **工具注册表**: `ToolRegistry` 统一管理 Agent 可调用工具（文件读写、Shell执行、数据探查、日志分析等）。
- **代码生成器**: `CodeGenerator` 提供从零生成和修改现有代码的能力，内置模板骨架作为LLM失败时的后备方案。
- **Mock机制**: 当LLM/记忆/工具模块不可用时，编排器会自动创建Mock对象，保证系统可用性。

---

## 测试策略

### 当前状态
- **项目目前没有自动化测试套件**（无 `pytest`、`unittest` 等测试文件）。
- 验证主要依靠：
  1. **Python AST语法检查**: `tools.py` 中的 `_validate_code` 使用 `ast.parse` 检查语法
  2. **Smoke Test**: `DesignPhase._run_smoke_test` 执行模型前向传播验证代码可运行
  3. **集成验证**: 运行 `main.py` 或 `train.py` 进行端到端测试

### 建议的测试方式

如需添加测试，建议放在 `framework/tests/` 目录下：

```python
# 示例：验证模型前向传播
import torch
from code.models import GNNClassifier, GRU4Rec

def test_gnn_classifier():
    model = GNNClassifier(in_dim=5, hidden_dim=16, num_classes=3, num_layers=2)
    x = torch.randn(10, 5)
    adj = torch.eye(10)
    out = model(x, adj)
    assert out.shape == (10, 3)

def test_gru4rec():
    model = GRU4Rec(num_items=20, embedding_dim=16, hidden_dim=32)
    seq = torch.randint(1, 20, (4, 10))
    lens = torch.randint(1, 10, (4,))
    out = model(seq, lens)
    assert out.shape == (4, 16)
```

---

## 部署与提交

### 输出产物

Agent运行完成后，提交产物保存在 `./output/submission/` 目录：
- `A1.csv`: Task 1 分类预测结果（格式：`node_id,predicted_category`）
- `A2.csv`: Task 2 推荐预测结果（格式：`user_id,top1,top2,...,top10`）
- `trajectory.json`: 完整实验轨迹记录
- `trajectory_task1.json` / `trajectory_task2.json`: 各任务详细轨迹
- `submission.zip`: 自动打包的提交文件

### 运行环境要求

- Python 3.8+
- CUDA（可选，CPU亦可运行）
- 至少 8GB 内存（推荐16GB）
- LLM API 访问（运行时从环境变量读取配置）

---

## 安全与合规注意事项

1. **API密钥管理**: `config.yaml` 中可能包含 LLM API 密钥。生产环境中应始终通过环境变量 `LLM_API_KEY` / `LLM_BASE_URL` 传入，不要将密钥提交到版本控制。
2. **Shell命令执行**: `ToolRegistry._shell_exec` 使用 `subprocess.run(..., shell=True)` 执行命令。Agent生成的命令应经过校验，避免命令注入。
3. **代码回滚**: `DesignPhase` 在语法验证失败时会尝试回滚修改，但当前回滚逻辑较简化（仅记录操作）。重要修改前建议手动备份。
4. **文件写入**: `ToolRegistry._file_write` 会覆盖现有文件，Agent应谨慎操作关键代码文件。
5. **数据路径**: 训练脚本接受任意数据路径，应确保路径合法性，避免目录遍历。

---

## 给Agent的关键提示

1. **所有修改保持中文注释**。不要英文化现有注释。
2. **模型代码在 `framework/code/` 目录下**，这是Agent可以自由修改的区域。`framework/agent/` 是Agent系统本身，修改需更谨慎。
3. **配置驱动**: 多数超参数可通过 `config.yaml` 或命令行参数调整，优先尝试配置修改而非代码重写。
4. **回退机制**: LLM不可用时系统会启用地启发式逻辑和Mock对象，确保流程不中断。
5. **PyTorch版本兼容性**: 代码使用 `weights_only=False` 进行 `torch.load()`，这是为了兼容旧版本保存的检查点。
