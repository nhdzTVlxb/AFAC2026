"""模型定义 - 支持GNN分类和序列推荐

本模块包含两类模型：
1. GNN分类器: GCN, GraphSAGE, 用于图节点分类(Task 1)
2. 序列推荐模型: GRU4Rec, SASRec, 用于序列推荐(Task 2)

Agent可以动态修改模型架构、层数、维度等超参数。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ===== GNN层定义 =====

class GCNLayer(nn.Module):
    """GCN层: 支持对称归一化邻接矩阵"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        # 使用Xavier初始化
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x, adj):
        """前向传播

        Args:
            x: 节点特征, shape (N, in_dim)
            adj: 归一化邻接矩阵, shape (N, N)

        Returns:
            更新后的节点表示, shape (N, out_dim)
        """
        support = self.linear(x)  # (N, out_dim)
        return adj @ support  # (N, N) @ (N, out_dim) -> (N, out_dim)


class SAGELayer(nn.Module):
    """GraphSAGE层 - 均值聚合 + 拼接"""
    def __init__(self, in_dim, out_dim, aggr='mean'):
        super().__init__()
        self.aggr = aggr
        # 拼接自身特征和邻居特征
        self.linear = nn.Linear(in_dim * 2, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x, adj):
        """前向传播

        Args:
            x: 节点特征, shape (N, in_dim)
            adj: 邻接矩阵(0/1或归一化), shape (N, N)

        Returns:
            更新后的节点表示, shape (N, out_dim)
        """
        # 聚合邻居特征: adj @ x -> (N, in_dim)
        h_neigh = adj @ x
        if self.aggr == 'mean':
            degree = adj.sum(dim=1, keepdim=True).clamp(min=1)  # (N, 1)
            h_neigh = h_neigh / degree
        # 拼接自身和邻居
        h_concat = torch.cat([x, h_neigh], dim=-1)  # (N, in_dim * 2)
        return F.relu(self.linear(h_concat))  # (N, out_dim)


class GATLayer(nn.Module):
    """GAT层 - 单头注意力"""
    def __init__(self, in_dim, out_dim, dropout=0.6, alpha=0.2):
        super().__init__()
        self.out_dim = out_dim
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Linear(in_dim, out_dim)
        self.a = nn.Linear(out_dim * 2, 1)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.weight)

    def forward(self, x, adj):
        """前向传播"""
        # x: (N, in_dim), adj: (N, N)
        Wh = self.W(x)  # (N, out_dim)
        N = Wh.size(0)

        # 计算注意力分数
        a_input = torch.cat([
            Wh.unsqueeze(1).expand(N, N, -1),  # (N, N, out_dim)
            Wh.unsqueeze(0).expand(N, N, -1)   # (N, N, out_dim)
        ], dim=-1)  # (N, N, out_dim * 2)
        e = self.leakyrelu(self.a(a_input).squeeze(-1))  # (N, N)

        # 使用邻接矩阵掩码
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, p=self.dropout, training=self.training)

        h_prime = attention @ Wh  # (N, out_dim)
        return F.elu(h_prime)


# ===== GNN分类器 =====

class GNNClassifier(nn.Module):
    """GNN节点分类器，支持GCN/SAGE/GAT

    Agent可以修改以下参数来优化性能:
    - model_type: gcn/sage/gat
    - num_layers: 层数(1-4)
    - hidden_dim: 隐藏层维度
    - dropout: Dropout率
    """
    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=2,
                 dropout=0.5, model_type="sage"):
        super().__init__()
        self.model_type = model_type
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        # 构建GNN层
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_d = in_dim if i == 0 else hidden_dim
            out_d = hidden_dim if i < num_layers - 1 else num_classes
            if model_type == "gcn":
                self.layers.append(GCNLayer(in_d, out_d))
            elif model_type == "gat":
                self.layers.append(GATLayer(in_d, out_d, dropout=dropout))
            else:  # sage or default
                self.layers.append(SAGELayer(in_d, out_d))

        # 中间层的BatchNorm(最后一层不需要)
        if num_layers > 1:
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)
            ])
        else:
            self.batch_norms = nn.ModuleList()

    def forward(self, x, adj):
        """前向传播

        Args:
            x: 节点特征, shape (N, in_dim)
            adj: 邻接矩阵, shape (N, N)

        Returns:
            节点分类logits, shape (N, num_classes)
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            if i < len(self.layers) - 1:
                x = self.batch_norms[i](x)
                x = F.relu(x)
                x = self.dropout(x)
        return x  # (N, num_classes)

    def get_node_embeddings(self, x, adj):
        """获取倒数第二层的节点嵌入"""
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            if i < len(self.layers) - 1:
                x = self.batch_norms[i](x)
                x = F.relu(x)
                x = self.dropout(x)
            elif i == len(self.layers) - 1:
                break
        return x


# ===== 序列推荐模型 =====

class GRU4Rec(nn.Module):
    """GRU4Rec序列推荐模型

    使用GRU编码用户交互序列，预测下一个物品。
    Agent可以修改embedding_dim, hidden_dim, num_layers等。
    """
    def __init__(self, num_items, embedding_dim=64, hidden_dim=128,
                 num_layers=1, dropout=0.2, max_len=50):
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # 物品嵌入(0为padding)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)

        # GRU编码器
        self.gru = nn.GRU(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 输出投影到嵌入空间
        self.output_projection = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # 初始化
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, item_seq, seq_len=None):
        """前向传播

        Args:
            item_seq: 物品序列, shape (batch, seq_len)
            seq_len: 实际序列长度, shape (batch,)

        Returns:
            序列表示, shape (batch, embedding_dim)
        """
        x = self.item_embedding(item_seq)  # (batch, seq_len, embedding_dim)
        x = self.dropout(x)
        output, _ = self.gru(x)  # (batch, seq_len, hidden_dim)
        output = self.layer_norm(output)

        # 取最后一个有效位置
        if seq_len is not None:
            idx = seq_len.clamp(min=1) - 1
            last = output[torch.arange(output.size(0)), idx]
        else:
            last = output[:, -1, :]

        return self.output_projection(last)  # (batch, embedding_dim)

    def predict(self, item_seq, seq_len, candidate_items):
        """计算候选物品的分数

        Args:
            item_seq: 物品序列
            seq_len: 序列长度
            candidate_items: 候选物品ID, shape (num_candidates,)

        Returns:
            分数, shape (batch, num_candidates)
        """
        seq_repr = self.forward(item_seq, seq_len)  # (batch, embedding_dim)
        item_emb = self.item_embedding(candidate_items)  # (num_candidates, embedding_dim)
        scores = seq_repr @ item_emb.T  # (batch, num_candidates)
        return scores


class SASRec(nn.Module):
    """SASRec自注意力序列推荐模型

    使用Transformer编码器建模序列中的长程依赖。
    Agent可以修改embedding_dim, num_heads, num_layers等。
    """
    def __init__(self, num_items, embedding_dim=64, max_len=50,
                 num_heads=2, num_layers=2, dropout=0.2):
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.max_len = max_len

        # 物品嵌入(0为padding)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)

        # 位置嵌入
        self.position_embedding = nn.Embedding(max_len, embedding_dim)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # 初始化
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.normal_(self.position_embedding.weight, 0, 0.01)

    def forward(self, item_seq):
        """前向传播

        Args:
            item_seq: 物品序列, shape (batch, seq_len)

        Returns:
            序列表示, shape (batch, embedding_dim)
        """
        batch_size, seq_len = item_seq.size()

        # 位置编码
        positions = torch.arange(seq_len, device=item_seq.device).unsqueeze(0)  # (1, seq_len)
        pos_emb = self.position_embedding(positions)  # (1, seq_len, embedding_dim)

        # 物品嵌入 + 位置嵌入
        x = self.item_embedding(item_seq) + pos_emb  # (batch, seq_len, embedding_dim)
        x = self.dropout(self.layer_norm(x))

        # 构建padding掩码和因果掩码
        padding_mask = (item_seq == 0)  # (batch, seq_len)

        # 因果掩码(防止看到未来)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=item_seq.device), diagonal=1
        ).bool()  # (seq_len, seq_len)

        output = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )  # (batch, seq_len, embedding_dim)

        # 取最后一个有效位置
        seq_lengths = (item_seq != 0).sum(dim=1).clamp(min=1) - 1  # (batch,)
        last_output = output[torch.arange(batch_size), seq_lengths]  # (batch, embedding_dim)

        return last_output

    def predict(self, item_seq, candidate_items):
        """计算候选物品的分数

        Args:
            item_seq: 物品序列
            candidate_items: 候选物品ID, shape (num_candidates,)

        Returns:
            分数, shape (batch, num_candidates)
        """
        seq_repr = self.forward(item_seq)  # (batch, embedding_dim)
        item_emb = self.item_embedding(candidate_items)  # (num_candidates, embedding_dim)
        scores = seq_repr @ item_emb.T  # (batch, num_candidates)
        return scores


class MultiHeadSASRec(nn.Module):
    """多任务/增强版SASRec - 可作为Agent实验的扩展架构"""
    def __init__(self, num_items, embedding_dim=64, max_len=50,
                 num_heads=2, num_layers=2, dropout=0.2,
                 use_item_bias=True):
        super().__init__()
        self.base = SASRec(num_items, embedding_dim, max_len,
                           num_heads, num_layers, dropout)
        self.use_item_bias = use_item_bias
        if use_item_bias:
            self.item_bias = nn.Parameter(torch.zeros(num_items + 1))

    def forward(self, item_seq):
        return self.base.forward(item_seq)

    def predict(self, item_seq, candidate_items):
        seq_repr = self.forward(item_seq)
        item_emb = self.base.item_embedding(candidate_items)
        scores = seq_repr @ item_emb.T
        if self.use_item_bias:
            scores = scores + self.item_bias[candidate_items]
        return scores
