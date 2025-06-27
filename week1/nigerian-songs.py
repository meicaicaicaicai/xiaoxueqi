import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
df = pd.read_csv("nigerian-songs.csv")

# 选择关键音频特征
features = ['danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'tempo']

# 绘制箱线图
plt.figure(figsize=(15,8))
sns.boxplot(data=df[features])
plt.title("Audio Features Distribution")
plt.xticks(rotation=45)
plt.show()

# 绘制直方图
df[features].hist(bins=30, figsize=(15,10))
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import (
    StandardScaler, PowerTransformer,
    QuantileTransformer, RobustScaler,
    MinMaxScaler, Binarizer
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 设置全局字体以解决中文显示问题
plt.rcParams['font.family'] = 'DejaVu Sans'  # 使用通用字体
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 加载数据
df = pd.read_csv("nigerian-songs.csv")

# 2. 选择音频特征列
audio_features = ['danceability', 'energy', 'loudness', 'speechiness',
                  'acousticness', 'instrumentalness', 'liveness', 'tempo']

# 3. 定制化预处理管道
preprocessor = ColumnTransformer(transformers=[
    # 正态分布特征 - 标准化
    ('danceability', StandardScaler(), ['danceability']),

    # 偏态特征 - Yeo-Johnson变换
    ('speechiness', PowerTransformer(method='yeo-johnson'), ['speechiness']),

    # 多峰分布 - 分位数变换
    ('liveness', QuantileTransformer(output_distribution='normal'), ['liveness']),

    # 能量特征 - 缩放到[-1,1]范围
    ('energy', MinMaxScaler(feature_range=(-1, 1)), ['energy']),

    # 响度特征 - 抗离群值缩放
    ('loudness', RobustScaler(), ['loudness']),

    # 原声特征 - 二值化+缩放
    ('acousticness', Pipeline(steps=[
        ('binarizer', Binarizer(threshold=0.5)),
        ('scaler', RobustScaler())
    ]), ['acousticness']),

    # 乐器特征 - 二值化+变换
    ('instrumentalness', Pipeline(steps=[
        ('binarizer', Binarizer(threshold=0.01)),
        ('scaler', PowerTransformer(method='yeo-johnson'))
    ]), ['instrumentalness']),

    # 速度特征 - 标准化
    ('tempo', StandardScaler(), ['tempo'])
])

# 4. 创建完整处理管道
cluster_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=0.95)),  # 降维去除相关性
    ('cluster', KMeans(n_clusters=5, random_state=42, n_init=10))  # 聚类
])

# 5. 应用聚类
cluster_labels = cluster_pipe.fit_predict(df[audio_features])

# 6. 评估聚类质量
transformed_features = cluster_pipe.named_steps['preprocessor'].transform(df[audio_features])

if len(np.unique(cluster_labels)) > 1:  # 确保有多个聚类
    sil_score = silhouette_score(transformed_features, cluster_labels)
    ch_score = calinski_harabasz_score(transformed_features, cluster_labels)
else:
    sil_score = -1
    ch_score = 0

print(f"Silhouette Score: {sil_score:.3f}")
print(f"Calinski-Harabasz Index: {ch_score:.3f}")


# 7. 可视化聚类结果
def visualize_clusters(features, labels, method='pca'):
    """Visualize clustering results"""
    plt.figure(figsize=(12, 8))

    if method == 'pca':
        reducer = PCA(n_components=2)
        title = "PCA Clustering Visualization"
    else:
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        title = "t-SNE Clustering Visualization"

    # 降维
    reduced_data = reducer.fit_transform(features)

    # 绘制散点图
    scatter = plt.scatter(
        reduced_data[:, 0], reduced_data[:, 1],
        c=labels, cmap='viridis', alpha=0.7, s=50
    )

    plt.title(f"{title}\nSilhouette: {sil_score:.3f} | CH Index: {ch_score:.3f}")
    plt.colorbar(scatter, label='Cluster Label')
    plt.xlabel(f"{method.upper()} Component 1")
    plt.ylabel(f"{method.upper()} Component 2")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'clustering_{method}.png', dpi=300, bbox_inches='tight')
    plt.show()


# 使用PCA可视化
print("Visualizing with PCA...")
visualize_clusters(transformed_features, cluster_labels, method='pca')

# 使用t-SNE可视化
print("Visualizing with t-SNE...")
visualize_clusters(transformed_features, cluster_labels, method='tsne')


# 8. 分析聚类特征
def analyze_clusters(df, labels, audio_features):
    """Analyze feature distribution for each cluster"""
    # 添加聚类标签到原始数据
    df_clustered = df.copy()
    df_clustered['cluster'] = labels

    # 计算每个聚类的特征均值
    cluster_means = df_clustered.groupby('cluster')[audio_features].mean()

    # 绘制雷达图
    plt.figure(figsize=(14, 10))

    # 创建子图
    ax = plt.subplot(111, polar=True)
    plt.subplots_adjust(top=0.85)

    # 角度设置
    angles = np.linspace(0, 2 * np.pi, len(audio_features), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    # 为每个聚类绘制雷达图
    for cluster in sorted(df_clustered['cluster'].unique()):
        values = cluster_means.loc[cluster].tolist()
        values += values[:1]  # 闭合
        ax.plot(angles, values, linewidth=2, label=f'Cluster {cluster}')
        ax.fill(angles, values, alpha=0.25)

    # 添加特征标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(audio_features, fontsize=10)

    # 添加标题和图例
    plt.title('Audio Feature Analysis by Cluster', size=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    plt.savefig('cluster_radar.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 返回聚类分析结果
    return cluster_means


# 执行聚类分析
print("Analyzing cluster characteristics...")
cluster_analysis = analyze_clusters(df, cluster_labels, audio_features)
print("\nCluster Feature Means:")
print(cluster_analysis)


# 9. 特征分布直方图
def plot_feature_distributions(df, features, cols=3):
    """Plot distributions of all audio features"""
    rows = (len(features) + cols - 1) // cols
    plt.figure(figsize=(15, 5 * rows))

    for i, feature in enumerate(features, 1):
        plt.subplot(rows, cols, i)
        sns.histplot(df[feature], bins=30, kde=True)
        plt.title(f'{feature.capitalize()} Distribution')
        plt.xlabel('')
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()


print("Plotting feature distributions...")
plot_feature_distributions(df, audio_features)

# 10. 保存处理后的数据
df_processed = df.copy()
df_processed['cluster'] = cluster_labels

# 保存为CSV
df_processed.to_csv('processed_songs_with_clusters.csv', index=False)
print("Processed data saved to 'processed_songs_with_clusters.csv'")


# 11. 聚类特征比较图
def plot_cluster_comparison(cluster_means):
    """Plot comparison of cluster characteristics"""
    plt.figure(figsize=(14, 8))

    # 标准化数据以便比较
    normalized_means = cluster_means.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)

    # 为每个特征绘制条形图
    for i, feature in enumerate(audio_features, 1):
        plt.subplot(2, 4, i)
        sns.barplot(x=normalized_means.index, y=normalized_means[feature])
        plt.title(feature.capitalize())
        plt.ylabel('Normalized Value')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)

    plt.suptitle('Cluster Feature Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('cluster_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


print("Plotting cluster comparison...")
plot_cluster_comparison(cluster_analysis)