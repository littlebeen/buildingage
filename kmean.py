import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ===================== 1. 核心函数：提取单建筑特征 =====================
def extract_building_features(mask_list, feat_list, age_label_dict=None):
    """
    从批量mask和特征图中提取所有建筑的特征向量
    参数：
        mask_list: 实例掩码列表，每个元素是tensor(B×1×512×512)
        feat_list: 特征图列表，每个元素是tensor(B×64×128×128)
        age_label_dict: 可选，建筑年代标签字典 {建筑ID: 年代(str)}，如 {1:"1970前", 2:"1980s"...}
    返回：
        all_feats: 所有建筑的特征矩阵 (N_buildings, 64)
        all_building_ids: 所有建筑的ID (N_buildings,)
        all_ages: 所有建筑的年代标签 (N_buildings,)（若age_label_dict不为None）
    """
    all_feats = []
    all_building_ids = []
    all_ages = []
    
    # 遍历每个batch样本
    for batch_idx in range(len(mask_list)):
        masks = mask_list[batch_idx]  # (B, 1, 512, 512)
        feats = feat_list[batch_idx]   # (B, 64, 128, 128)
        B = masks.shape[0]
        
        for b in range(B):
            # 单样本处理
            mask = masks[b, 0, :, :]  # (512, 512) 去掉batch和channel维
            feat = feats[b, :, :, :]   # (64, 128, 128)
            
            # 获取当前样本中的所有建筑ID（排除0背景）
            building_ids = torch.unique(mask)
            building_ids = [id for id in building_ids if id != 0]
            
            for bid in building_ids:
                # 1. 生成当前建筑的掩码
                building_mask = (mask == bid)  # (512, 512)
                
                # 2. 将掩码缩放到特征图尺寸(128×128)
                building_mask_resized = torch.nn.functional.interpolate(
                    building_mask.unsqueeze(0).unsqueeze(0).float(),
                    size=(128, 128),
                    mode='nearest'
                ).squeeze()  # (128, 128)
                
                # 3. 筛选建筑区域的特征并池化
                if torch.sum(building_mask_resized) < 1:  # 空建筑跳过
                    continue
                feat_masked = feat[:, building_mask_resized]  # (64, N_pixels)
                building_feat = feat_masked.mean(dim=1)      # (64,) 单建筑特征
                
                # 4. 收集结果
                all_feats.append(building_feat.cpu().numpy())
                all_building_ids.append(int(bid))
                
                # 5. 匹配年代标签（若有）
                if age_label_dict is not None and int(bid) in age_label_dict:
                    all_ages.append(age_label_dict[int(bid)])
    
    # 转换为numpy数组
    all_feats = np.array(all_feats)
    all_building_ids = np.array(all_building_ids)
    
    if age_label_dict is not None:
        all_ages = np.array(all_ages)
        return all_feats, all_building_ids, all_ages
    else:
        return all_feats, all_building_ids

# ===================== 2. 聚类与可视化函数 =====================
def cluster_and_visualize(all_feats, all_building_ids, all_ages=None, n_clusters=5):
    """
    特征聚类并生成可视化图表
    参数：
        all_feats: 所有建筑特征 (N, 64)
        all_building_ids: 建筑ID (N,)
        all_ages: 可选，年代标签 (N,)
        n_clusters: 聚类数（默认5，对应5个年代）
    """
    # -------------------- 2.1 K-Means聚类 --------------------
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(all_feats)
    
    # -------------------- 2.2 TSNE降维可视化 --------------------
    # 降维到2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_feats)-1))
    feat_2d = tsne.fit_transform(all_feats)
    
    # 画图
    plt.figure(figsize=(12, 8))
    if all_ages is not None:
        # 年代标签转数值（方便配色）
        age_mapping = {"1970前":0, "1980s":1, "1990s":2, "2000s":3, "2010+":4}
        age_nums = [age_mapping.get(age, 5) for age in all_ages]
        
        # 颜色=年代，形状=聚类
        scatter = plt.scatter(
            feat_2d[:,0], feat_2d[:,1],
            c=age_nums, cmap='jet', s=60, alpha=0.8,
            edgecolors=[f"C{cl}" for cl in cluster_labels],
            linewidth=1
        )
        plt.colorbar(scatter, label='Building Age', ticks=range(5), 
                     ticklabels=["1970前", "1980s", "1990s", "2000s", "2010+"])
    else:
        # 无年代标签时，颜色=聚类
        scatter = plt.scatter(
            feat_2d[:,0], feat_2d[:,1],
            c=cluster_labels, cmap='tab10', s=60, alpha=0.8
        )
        plt.colorbar(scatter, label='Cluster Label')
    
    plt.title('TSNE Clustering of Hong Kong Building Features', fontsize=14)
    plt.xlabel('TSNE Dimension 1', fontsize=12)
    plt.ylabel('TSNE Dimension 2', fontsize=12)
    plt.grid(alpha=0.3)
    plt.savefig('building_tsne_clustering.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # -------------------- 2.3 同年代特征相似度分析 --------------------
    if all_ages is not None:
        age_groups = ["1970前", "1980s", "1990s", "2000s", "2010+"]
        avg_similarity = []
        
        for age in age_groups:
            group_feats = all_feats[all_ages == age]
            if len(group_feats) < 2:
                avg_similarity.append(0)
                continue
            # 计算余弦相似度
            sim_matrix = cosine_similarity(group_feats)
            # 取上三角（排除对角线）计算平均
            avg_sim = np.mean(sim_matrix[np.triu_indices(len(sim_matrix), k=1)])
            avg_similarity.append(avg_sim)
        
        # 画柱状图
        plt.figure(figsize=(10, 6))
        bars = plt.bar(age_groups, avg_similarity, color=['#8B4513', '#CD853F', '#DAA520', '#4169E1', '#00CED1'])
        plt.title('Average Feature Similarity within Each Age Group', fontsize=14)
        plt.ylabel('Cosine Similarity', fontsize=12)
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        # 标注数值
        for bar, val in zip(bars, avg_similarity):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.02, 
                     f'{val:.2f}', ha='center', fontsize=10)
        plt.tight_layout()
        plt.savefig('age_group_similarity.png', dpi=300)
        plt.show()
    
    # -------------------- 2.4 聚类-年代匹配热力图 --------------------
    if all_ages is not None:
        # 构建聚类-年代计数矩阵
        age_unique = sorted(list(set(all_ages)))
        cluster_age_matrix = np.zeros((n_clusters, len(age_unique)))
        
        for cl in range(n_clusters):
            cl_ages = all_ages[cluster_labels == cl]
            for i, age in enumerate(age_unique):
                cluster_age_matrix[cl, i] = np.sum(cl_ages == age)
        
        # 归一化（每行和为1）
        cluster_age_matrix = cluster_age_matrix / cluster_age_matrix.sum(axis=1, keepdims=True)
        
        # 画热力图
        plt.figure(figsize=(8, 6))
        im = plt.imshow(cluster_age_matrix, cmap='Blues')
        plt.xticks(range(len(age_unique)), age_unique, rotation=45)
        plt.yticks(range(n_clusters), [f'Cluster {i}' for i in range(n_clusters)])
        plt.title('Cluster-Age Distribution (Normalized)', fontsize=14)
        plt.colorbar(im, label='Proportion')
        
        # 标注数值
        for i in range(n_clusters):
            for j in range(len(age_unique)):
                plt.text(j, i, f'{cluster_age_matrix[i,j]:.2f}', 
                         ha='center', va='center', color='black')
        plt.tight_layout()
        plt.savefig('cluster_age_heatmap.png', dpi=300)
        plt.show()
    
    return cluster_labels

# ===================== 3. 主运行流程 =====================
if __name__ == "__main__":
    # -------------------- 模拟你的数据（替换为真实数据） --------------------
    # 模拟mask_list: 每个元素是(B,1,512,512) tensor，值为0(背景)或1-5(建筑ID)
    mask_list = [
        torch.randint(0, 6, (2, 1, 512, 512)),  # batch1: 2个样本
        torch.randint(0, 6, (3, 1, 512, 512))   # batch2: 3个样本
    ]
    
    # 模拟feat_list: 每个元素是(B,64,128,128) tensor
    feat_list = [
        torch.randn(2, 64, 128, 128),  # batch1特征
        torch.randn(3, 64, 128, 128)   # batch2特征
    ]
    
    # 模拟年代标签字典（替换为你的真实标签）
    age_label_dict = {
        1: "1970前",
        2: "1980s",
        3: "1990s",
        4: "2000s",
        5: "2010+"
    }
    
    # -------------------- 运行特征提取与聚类 --------------------
    # 提取特征
    all_feats, all_building_ids, all_ages = extract_building_features(
        mask_list, feat_list, age_label_dict
    )
    
    # 聚类并出图
    cluster_labels = cluster_and_visualize(
        all_feats, all_building_ids, all_ages, n_clusters=5
    )
    
    # -------------------- 输出关键结果 --------------------
    print(f"总提取建筑数: {len(all_feats)}")
    print(f"各聚类包含建筑数: {np.bincount(cluster_labels)}")
    if all_ages is not None:
        for age in ["1970前", "1980s", "1990s", "2000s", "2010+"]:
            age_feats = all_feats[all_ages == age]
            print(f"{age} 建筑数: {len(age_feats)}")