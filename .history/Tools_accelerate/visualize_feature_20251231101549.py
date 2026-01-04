import cv2
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap  # 引入 UMAP 库

def visualize_feature_map(feature_map, original_image=None, method='max', save_path=None):
    """
    可视化256通道特征图的多种方法，直接保存为JPG格式
    
    参数:
        feature_map (numpy.ndarray): 特征图，形状为(256, 64, 64)
        original_image (numpy.ndarray, optional): 原始图像
        method (str): 可视化方法，可选'max', 'mean', 'pca', 'umap', 'grid'
        save_path (str, optional): 保存路径，如果为None则不保存
    
    返回:
        numpy.ndarray: 可视化结果
    """
    # 设置matplotlib为非交互模式，不显示图形
    plt.switch_backend('Agg')
    
    if method == 'max':
        reduced_map = np.max(feature_map, axis=0)
    elif method == 'mean':
        reduced_map = np.mean(feature_map, axis=0)
    elif method == 'pca':
        c, h, w = feature_map.shape
        reshaped = feature_map.reshape(c, -1).T
        pca = PCA(n_components=3)
        principal_components = pca.fit_transform(reshaped)
        pc_min = principal_components.min(axis=0)
        pc_max = principal_components.max(axis=0)
        principal_components = (principal_components - pc_min) / (pc_max - pc_min)
        rgb_image = (principal_components.reshape(h, w, 3) * 255).astype(np.uint8)
        
        if original_image is not None:
            rgb_image = cv2.resize(rgb_image, (original_image.shape[1], original_image.shape[0]))
            result = cv2.addWeighted(original_image, 0.5, rgb_image, 0.5, 0)
        else:
            result = rgb_image
            
        if save_path is not None:
            cv2.imwrite(save_path, result)
        return result
    elif method == 'umap':
        c, h, w = feature_map.shape
        reshaped = feature_map.reshape(c, -1).T
        reducer = umap.UMAP(n_components=3, random_state=42)
        umap_components = reducer.fit_transform(reshaped)
        umap_min = umap_components.min(axis=0)
        umap_max = umap_components.max(axis=0)
        umap_components = (umap_components - umap_min) / (umap_max - umap_min)
        rgb_image = (umap_components.reshape(h, w, 3) * 255).astype(np.uint8)
        
        if original_image is not None:
            rgb_image = cv2.resize(rgb_image, (original_image.shape[1], original_image.shape[0]))
            result = cv2.addWeighted(original_image, 0.5, rgb_image, 0.5, 0)
        else:
            result = rgb_image
            
        if save_path is not None:
            cv2.imwrite(save_path, result)
        return result
    elif method == 'grid':
        visualize_feature_map_grid(feature_map, save_path=save_path)
        return None
    
    if method in ['max', 'mean']:
        heatmap = cv2.normalize(reduced_map, None, alpha=0, beta=255, 
                               norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        if original_image is not None:
            heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
            result = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
        else:
            result = heatmap
            
        if save_path is not None:
            cv2.imwrite(save_path, result)
        return result

def visualize_feature_map_grid(feature_map, n_channels=64, save_path=None):
    """
    分块显示部分通道的特征图，直接保存为JPG格式
    """
    selected_channels = feature_map[:n_channels]
    rows = int(np.sqrt(n_channels))
    cols = n_channels // rows + (1 if n_channels % rows else 0)
    
    fig = plt.figure(figsize=(15, 15))
    for i in range(n_channels):
        plt.subplot(rows, cols, i+1)
        plt.imshow(selected_channels[i], cmap='viridis')
        plt.title(f'Channel {i}')
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=100, bbox_inches='tight', format='jpg')
    
    plt.close(fig)




if __name__ == "__main__":
    # 从 .npy 文件加载特征图
    feature_map_path = "/data2/wuxinrui/RA-L/MobileSAM/Tools_accelerate/vit_h_tiny_msam_mask&unmask/Genicanthus_watanabei###413261403.jpg.npy"  # 替换为你的特征图文件路径
    feature_map = np.load(feature_map_path)  # 加载特征图

    # 方法1: 最大响应
    visualize_feature_map(feature_map, method='max', save_path='feature_max.jpg')
    
    # 方法2: PCA降维
    visualize_feature_map(feature_map, method='pca', save_path='feature_pca.jpg')
    
    # 方法3: UMAP降维
    # visualize_feature_map(feature_map, method='umap', save_path='feature_umap.jpg')
    
    # 方法4: 网格显示
    visualize_feature_map_grid(feature_map, save_path='feature_grid.jpg')