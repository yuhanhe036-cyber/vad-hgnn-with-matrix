import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
import random
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# ==========================================
# 1. 模型定义 (HGNN + Attention)
# ==========================================
class HGNNConv(nn.Module):
    def __init__(self, in_ft, out_ft):
        super().__init__()
        self.fc = nn.Linear(in_ft, out_ft)

    def forward(self, x, H, W):
        # 传播公式: Dv^-0.5 * H * W * De^-1 * H.T * Dv^-0.5 * X
        Dv = torch.sum(H, dim=1).clamp(min=1e-5)
        De = torch.sum(H, dim=0).clamp(min=1e-5)
        inv_Dv = torch.pow(Dv, -0.5).unsqueeze(1)
        inv_De = torch.pow(De, -1.0).unsqueeze(1)
        
        x = x * inv_Dv
        x = torch.matmul(H.t(), x)
        x = x * (W.unsqueeze(1) * inv_De)
        x = torch.matmul(H, x)
        x = x * inv_Dv
        return self.fc(x)

class WSVAD_Model(nn.Module):
    def __init__(self, in_dim=2048, latent_dim=256):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(in_dim, latent_dim), nn.ReLU(), nn.Dropout(0.5))
        self.hgnn = HGNNConv(latent_dim, latent_dim)
        # 异常得分预测器
        self.scoring = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x, H, W):
        x = self.phi(x)
        x = F.relu(self.hgnn(x, H, W))
        scores = self.scoring(x) # [N, 1]
        return scores.squeeze()

# ==========================================
# 2. 训练与评估逻辑 (MIL Ranking Logic)
# ==========================================
def run_experiment():
    CACHE_DIR = r"D:\vad_hgnn\cache"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    all_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.pkl')]
    normal_data = [pickle.load(open(os.path.join(CACHE_DIR, f), 'rb')) for f in all_files if 'norm' in f or 'train' in f]
    anomaly_data = [pickle.load(open(os.path.join(CACHE_DIR, f), 'rb')) for f in all_files if 'anom' in f]
    
    # 核心：为了训练，我们从异常数据中分出一部分用于弱监督训练
    train_anom = anomaly_data[:len(anomaly_data)//2]
    test_anom = anomaly_data[len(anomaly_data)//2:]
    train_norm = normal_data[:len(normal_data)//2]
    test_norm = normal_data[len(normal_data)//2:]

    model = WSVAD_Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print(f"Training on {len(train_anom)} anom and {len(train_norm)} norm videos...")

    # --- 训练循环 ---
    for epoch in range(50):
        model.train()
        epoch_loss = 0
        random.shuffle(train_anom)
        random.shuffle(train_norm)
        
        # 每次取一个异常视频和一个正常视频组成一对 (Ranking Pair)
        for i in range(min(len(train_anom), len(train_norm))):
            optimizer.zero_grad()
            
            # 处理异常视频包
            d_a = train_anom[i]
            s_a = model(torch.from_numpy(d_a['X']).to(device).float(), 
                        torch.from_numpy(d_a['H']).to(device).float(), 
                        torch.from_numpy(d_a['W']).to(device).float())
            
            # 处理正常视频包
            d_n = train_norm[i]
            s_n = model(torch.from_numpy(d_n['X']).to(device).float(), 
                        torch.from_numpy(d_n['H']).to(device).float(), 
                        torch.from_numpy(d_n['W']).to(device).float())
            
            # 1. Ranking Loss (MIL): max(0, 1 - max(s_a) + max(s_n))
            loss_rank = torch.max(torch.tensor(0.0).to(device), 1.0 - torch.max(s_a) + torch.max(s_n))
            
            # 2. Smoothness Loss (时序平滑)
            loss_smooth = torch.sum((s_a[1:] - s_a[:-1])**2)
            
            # 3. Sparsity Loss (异常稀疏)
            loss_sparse = torch.sum(s_a)
            
            total_loss = loss_rank + 0.0001 * loss_smooth + 0.0001 * loss_sparse
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}, Avg Loss: {epoch_loss/len(train_anom):.4f}")

    # --- 评估阶段 ---
    model.eval()
    y_true, y_scores = [], []
    test_set = test_norm + test_anom
    
    print("\nCalculating AUC...")
    with torch.no_grad():
        for data in test_set:
            x, H, W = [torch.from_numpy(data[k]).to(device).float() for k in ['X', 'H', 'W']]
            scores = model(x, H, W)
            # 视频级得分 = 该视频中最高的片段得分
            y_scores.append(torch.max(scores).item())
            y_true.append(data['label'])

    auc = roc_auc_score(y_true, y_scores)
    print("="*30)
    print(f"★ Final WSVAD AUC: {auc:.4f} ★")
    print("="*30)

if __name__ == "__main__":
    run_experiment()