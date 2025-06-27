import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, PowerTransformer,
    QuantileTransformer, RobustScaler,
    MinMaxScaler, Binarizer
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralBiclustering
from matplotlib.gridspec import GridSpec

# 1. 解决中文显示问题 - 使用更兼容的字体配置
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'sans-serif']  # 多字体备选
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 100  # 提高默认分辨率
plt.rcParams['savefig.dpi'] = 300  # 保存图片的分辨率
plt.rcParams['figure.figsize'] = (10, 6)  # 默认图形大小

# 2. 加载数据
df = pd.read_csv("nigerian-songs.csv")
print(f"数据集包含 {df.shape[0]} 首歌曲, {df.shape[1]} 个特征")

# 3. 选择音频特征列
audio_features = ['danceability', 'energy', 'loudness', 'speechiness',
                  'acousticness', 'instrumentalness', 'liveness', 'tempo']

# ======================================
# 初始数据探索可视化
# ======================================

print("\n=== 初始数据探索可视化 ===")

# 特征箱线图
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df[audio_features], ax=ax)
ax.set_title("音频特征分布与离群值检测", fontsize=14)
plt.xticks(rotation=45)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 特征直方图 - 修复显示问题
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()  # 将3x3网格展平为9个轴的列表
for i, feature in enumerate(audio_features):
    sns.histplot(df[feature], bins=30, kde=True, ax=axes[i])
    axes[i].set_title(f'{feature.capitalize()} 分布', fontsize=12)
    axes[i].set_xlabel('')
    axes[i].grid(alpha=0.3)

# 隐藏多余的子图
for j in range(len(audio_features), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("音频特征分布直方图", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为标题留空间
plt.show()

# 特征相关性热图
fig, ax = plt.subplots(figsize=(10, 8))
corr = df[audio_features].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
ax.set_title("音频特征相关性矩阵", fontsize=14)
plt.tight_layout()
plt.show()

# 流派分布分析
fig, ax = plt.subplots(figsize=(10, 6))
top_genres = df['artist_top_genre'].value_counts().head(10)
sns.barplot(x=top_genres.values, y=top_genres.index, palette='viridis', ax=ax)
ax.set_title("Top 10 音乐流派分布", fontsize=14)
ax.set_xlabel("歌曲数量")
ax.set_ylabel("音乐流派")
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# ======================================
# 数据预处理
# ======================================

print("\n=== 数据预处理 ===")

# 定制化预处理管道
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

# 创建完整处理管道
preprocess_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=0.95))  # 降维去除相关性
])

# 应用预处理
transformed_features = preprocess_pipe.fit_transform(df[audio_features])
print(f"预处理后特征维度: {transformed_features.shape}")

# ======================================
# 新增：预处理后可视化
# ======================================

print("\n=== 预处理后数据可视化 ===")


def plot_post_processing(transformed_data):
    """可视化预处理后的数据"""
    # 创建图形和子图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. 主成分散点图
    if transformed_data.shape[1] >= 2:
        axes[0].scatter(transformed_data[:, 0], transformed_data[:, 1], alpha=0.6)
        axes[0].set_title('前两个主成分分布', fontsize=14)
        axes[0].set_xlabel('第一主成分')
        axes[0].set_ylabel('第二主成分')
        axes[0].grid(True, alpha=0.3)

    # 2. 箱线图展示预处理后特征分布
    # 只展示前几个特征（如果特征太多）
    max_features_to_show = min(8, transformed_data.shape[1])
    sns.boxplot(data=pd.DataFrame(transformed_data[:, :max_features_to_show]),
                ax=axes[1])
    axes[1].set_title('预处理后特征分布', fontsize=14)
    axes[1].set_xlabel('特征')
    axes[1].set_ylabel('值')

    plt.suptitle("预处理后数据分布", fontsize=16)
    plt.tight_layout()
    plt.show()


# 调用预处理后可视化
plot_post_processing(transformed_features)

# ======================================
# 聚类分析
# ======================================

print("\n=== 聚类分析 ===")

# 应用谱双聚类
n_clusters = 5  # 设置双聚类数量
bicluster = SpectralBiclustering(
    n_clusters=(n_clusters, 3),  # (行聚类数, 列聚类数)
    random_state=42,
    n_init=10
)
bicluster.fit(transformed_features)

# 获取聚类标签
row_labels = bicluster.row_labels_  # 样本聚类标签
col_labels = bicluster.column_labels_  # 特征聚类标签

# 评估聚类质量
if len(np.unique(row_labels)) > 1:  # 确保有多个聚类
    sil_score = silhouette_score(transformed_features, row_labels)
    ch_score = calinski_harabasz_score(transformed_features, row_labels)
else:
    sil_score = -1
    ch_score = 0

print(f"轮廓系数 (Silhouette Score): {sil_score:.3f}")
print(f"Calinski-Harabasz 指数: {ch_score:.3f}")


# 可视化双聚类结果 - 分成两个独立图形
def visualize_biclustering(data, row_labels, col_labels):
    """可视化双聚类结果"""
    # 对行和列进行排序以显示聚类结构
    row_order = np.argsort(row_labels)
    col_order = np.argsort(col_labels)

    # 重新排列数据
    sorted_data = data[row_order][:, col_order]

    # 图1: 双聚类热图
    plt.figure(figsize=(14, 6))
    sns.heatmap(
        sorted_data,
        cmap='viridis',
        yticklabels=False,
        xticklabels=False,
        cbar_kws={'label': '特征值'}
    )
    plt.title("双聚类热图 (Spectral Biclustering)", fontsize=14)
    plt.ylabel("歌曲 (按聚类排序)")
    plt.xlabel("特征 (按聚类排序)")
    plt.tight_layout()
    plt.show()

    # 图2: 双聚类结构图
    plt.figure(figsize=(14, 6))
    plt.imshow(sorted_data, cmap='viridis', aspect='auto')

    # 添加行聚类边界
    row_boundaries = np.where(np.diff(row_labels[row_order]))[0] + 1
    for boundary in row_boundaries:
        plt.axhline(y=boundary, color='red', linestyle='--', linewidth=1.5)

    # 添加列聚类边界
    col_boundaries = np.where(np.diff(col_labels[col_order]))[0] + 1
    for boundary in col_boundaries:
        plt.axvline(x=boundary, color='blue', linestyle='--', linewidth=1.5)

    plt.title("双聚类结构 (红色=歌曲聚类边界, 蓝色=特征聚类边界)", fontsize=14)
    plt.ylabel("歌曲索引")
    plt.xlabel("特征索引")
    plt.colorbar(label='特征值')
    plt.tight_layout()
    plt.show()


print("可视化双聚类结果...")
visualize_biclustering(transformed_features, row_labels, col_labels)


# 可视化降维结果
def visualize_clusters(features, labels, method='pca'):
    """可视化聚类结果"""
    fig = plt.figure(figsize=(10, 8))

    if method == 'pca':
        reducer = PCA(n_components=2)
        title = "PCA聚类可视化"
    else:
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        title = "t-SNE聚类可视化"

    # 降维
    reduced_data = reducer.fit_transform(features)

    # 绘制散点图
    scatter = plt.scatter(
        reduced_data[:, 0], reduced_data[:, 1],
        c=labels, cmap='viridis', alpha=0.7, s=50
    )

    plt.title(f"{title}\n轮廓系数: {sil_score:.3f} | CH指数: {ch_score:.3f}", fontsize=14)
    plt.colorbar(scatter, label='聚类标签')
    plt.xlabel(f"{method.upper()} 成分 1")
    plt.ylabel(f"{method.upper()} 成分 2")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


# 使用PCA可视化
print("使用PCA可视化...")
visualize_clusters(transformed_features, row_labels, method='pca')

# 使用t-SNE可视化
print("使用t-SNE可视化...")
visualize_clusters(transformed_features, row_labels, method='tsne')


# 分析歌曲聚类特征 - 修复雷达图显示
def analyze_clusters(df, labels, audio_features):
    """分析聚类特征"""
    # 添加聚类标签到原始数据
    df_clustered = df.copy()
    df_clustered['cluster'] = labels

    # 计算每个聚类的特征均值
    cluster_means = df_clustered.groupby('cluster')[audio_features].mean()

    # 1. 雷达图 - 单独显示
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True)

    angles = np.linspace(0, 2 * np.pi, len(audio_features), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    for cluster in sorted(df_clustered['cluster'].unique()):
        values = cluster_means.loc[cluster].tolist()
        values += values[:1]  # 闭合
        ax.plot(angles, values, linewidth=2, label=f'聚类 {cluster}')
        ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(audio_features, fontsize=10)
    ax.set_title('各聚类音频特征分析', size=14, pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()

    # 2. 其他分析图表
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)

    # 分组特征箱线图
    ax1 = fig.add_subplot(gs[0, 0])
    cluster_means_normalized = cluster_means.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
    sns.heatmap(
        cluster_means_normalized.T,
        annot=True,
        cmap='YlGnBu',
        ax=ax1,
        cbar_kws={'label': '标准化值'}
    )
    ax1.set_title('各聚类标准化特征值', fontsize=14)
    ax1.set_xlabel('聚类')
    ax1.set_ylabel('音频特征')

    # 聚类分布条形图
    ax2 = fig.add_subplot(gs[0, 1])
    cluster_counts = df_clustered['cluster'].value_counts().sort_index()
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='viridis', ax=ax2)
    ax2.set_title('聚类大小分布', fontsize=14)
    ax2.set_xlabel('聚类')
    ax2.set_ylabel('歌曲数量')

    # 流派分布热图
    ax3 = fig.add_subplot(gs[1, :])
    genre_cluster = pd.crosstab(df_clustered['artist_top_genre'], df_clustered['cluster'])
    # 只显示包含歌曲数量较多的流派
    genre_cluster = genre_cluster[genre_cluster.sum(axis=1) > 5]
    sns.heatmap(genre_cluster, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_title('各聚类中音乐流派分布', fontsize=14)
    ax3.set_xlabel('聚类')
    ax3.set_ylabel('音乐流派')

    plt.suptitle("双聚类分析结果", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留空间
    plt.show()

    return cluster_means


print("分析聚类特征...")
cluster_analysis = analyze_clusters(df, row_labels, audio_features)
print("\n聚类特征均值:")
print(cluster_analysis)

print("\n=== 分析结果总结 ===")
print(f"1. 轮廓系数: {sil_score:.3f} (值在-1到1之间，越接近1表示聚类效果越好)")
print(f"2. Calinski-Harabasz指数: {ch_score:.3f} (值越高表示聚类效果越好)")
print("3. 评价指标说明:")
print("   - 轮廓系数 > 0.5: 聚类结构良好")
print("   - 轮廓系数 < 0.2: 聚类结构不明显")
print(
    f"   - 当前值为 {sil_score:.3f}，表明聚类结构{'良好' if sil_score > 0.5 else '一般' if sil_score > 0.2 else '不明显'}")
print(f"4. 聚类大小分布:")
cluster_counts = df.assign(cluster=row_labels)['cluster'].value_counts().sort_index()
for cluster, count in cluster_counts.items():
    print(f"   聚类 {cluster}: {count} 首歌曲 ({count / len(df) * 100:.1f}%)")

print("\n分析完成!")