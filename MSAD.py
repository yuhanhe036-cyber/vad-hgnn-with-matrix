import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle
from tqdm import tqdm
import random
import warnings

warnings.filterwarnings('ignore')

class Config:
    DATA_ROOT = r"D:\vad_hgnn\msad_raw\MSAD_frames"
    # 路径配置
    TRAIN_DIR = os.path.join(DATA_ROOT, "MSAD_normal_training_blur")
    TEST_NORM_DIR = os.path.join(DATA_ROOT, "MSAD_normal_testing_blur")
    TEST_ANOM_DIR = os.path.join(DATA_ROOT, "MSAD_anomaly_blur")
    CACHE_DIR = r"D:\vad_hgnn\cache"
    
    # 构图参数 (参考论文)
    SNIPPET_LEN = 16
    STRIDE = 2 # 减小步长，增加节点密度
    K = 5      # KNN 邻居数
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_h_matrix(X):
    """构建语义+多尺度时间超图"""
    N = X.shape[0]
    H_list = []
    
    # 1. 语义超边 (KNN)
    knn = NearestNeighbors(n_neighbors=min(Config.K+1, N)).fit(X)
    _, indices = knn.kneighbors(X)
    H_knn = np.zeros((N, N))
    for i, nbrs in enumerate(indices):
        H_knn[nbrs, i] = 1.0
    H_list.append(H_knn)
    
    # 2. 时序超边 (尺度1: 邻近3帧)
    if N > 2:
        H_t1 = np.zeros((N, N-2))
        for i in range(1, N-1):
            H_t1[i-1:i+2, i-1] = 1.0
        H_list.append(H_t1)
        
    # 3. 时序超边 (尺度2: 邻近5帧)
    if N > 4:
        H_t2 = np.zeros((N, N-4))
        for i in range(2, N-2):
            H_t2[i-2:i+3, i-2] = 1.0
        H_list.append(H_t2)
        
    H = np.concatenate(H_list, axis=1).astype(np.float32)
    W = np.ones(H.shape[1], dtype=np.float32)
    return H, W

def get_i3d_model():
    i3d = torch.hub.load("facebookresearch/pytorchvideo:main", model="i3d_r50", pretrained=True)
    model = torch.nn.Sequential(*list(i3d.blocks[:-1])).to(Config.DEVICE).eval()
    return model

def process_and_save(root_dir, label, prefix, model, trans):
    print(f"Processing {prefix} set...")
    # 遍历文件夹，寻找包含图片的末端文件夹
    for root, dirs, _ in os.walk(root_dir):
        if not dirs: # 最底层文件夹
            frame_files = sorted([f for f in os.listdir(root) if f.endswith(('.jpg', '.png'))])
            if len(frame_files) < Config.SNIPPET_LEN: continue
            
            video_name = os.path.basename(root)
            cache_path = os.path.join(Config.CACHE_DIR, f"{prefix}_{video_name}.pkl")
            if os.path.exists(cache_path): continue

            # 特征提取
            feat_list = []
            with torch.no_grad():
                for i in range(0, len(frame_files) - Config.SNIPPET_LEN + 1, Config.STRIDE):
                    snippet = [trans(Image.open(os.path.join(root, frame_files[j])).convert('RGB')) 
                               for j in range(i, i + Config.SNIPPET_LEN)]
                    clip = torch.stack(snippet).permute(1, 0, 2, 3).unsqueeze(0).to(Config.DEVICE)
                    feat = model(clip)
                    feat = torch.mean(feat, dim=(2,3,4)).squeeze().cpu().numpy()
                    feat_list.append(feat)
            
            X = np.array(feat_list)
            X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
            H, W = build_h_matrix(X)
            
            with open(cache_path, 'wb') as f:
                pickle.dump({'X': X, 'H': H, 'W': W, 'label': label}, f)

if __name__ == "__main__":
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    i3d = get_i3d_model()
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    
    # 处理三部分数据
    process_and_save(Config.TRAIN_DIR, 0, "train", i3d, transform)
    process_and_save(Config.TEST_NORM_DIR, 0, "test_norm", i3d, transform)
    process_and_save(Config.TEST_ANOM_DIR, 1, "test_anom", i3d, transform)