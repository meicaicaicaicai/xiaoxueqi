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
from matplotlib.gridspec import GridSpec
import pandas as pd

# 1. 解决中文显示问题
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (10, 6)




# 2. 加载数据
print("=== 加载数据 ===")
df = pd.read_csv("nigerian-songs.csv")
print(f"数据集包含 {df.shape[0]} 首歌曲, {df.shape[1]} 个特征")

# 3. 检查缺失值
print("\n每列缺失值统计：")
print(df.isnull().sum())

# 设置显示选项 - 确保完整输出
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# 查看所有列的完整统计摘要
print("\n所有列的完整统计摘要（无折叠）：")
print(df.describe(include='all'))



# 6. 查看唯一值数量
print("\n每列唯一值数量：")
print(df.nunique())

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

# 特征直方图
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, feature in enumerate(audio_features):
    sns.histplot(df[feature], bins=30, kde=True, ax=axes[i])
    axes[i].set_title(f'{feature.capitalize()} 分布', fontsize=12)
    axes[i].set_xlabel('')
    axes[i].grid(alpha=0.3)

# 隐藏多余的子图
for j in range(len(audio_features), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("音频特征分布直方图", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
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
    ('danceability', StandardScaler(), ['danceability']),
    ('speechiness', PowerTransformer(method='yeo-johnson'), ['speechiness']),
    ('liveness', QuantileTransformer(output_distribution='normal'), ['liveness']),
    ('energy', MinMaxScaler(feature_range=(-1, 1)), ['energy']),
    ('loudness', RobustScaler(), ['loudness']),
    ('acousticness', Pipeline(steps=[
        ('binarizer', Binarizer(threshold=0.5)),
        ('scaler', RobustScaler())
    ]), ['acousticness']),
    ('instrumentalness', Pipeline(steps=[
        ('binarizer', Binarizer(threshold=0.01)),
        ('scaler', PowerTransformer(method='yeo-johnson'))
    ]), ['instrumentalness']),
    ('tempo', StandardScaler(), ['tempo'])
])

# 创建完整处理管道
preprocess_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=0.95))
])

# 应用预处理
transformed_features = preprocess_pipe.fit_transform(df[audio_features])
print(f"预处理后特征维度: {transformed_features.shape}")

# ======================================
# 预处理后可视化
# ======================================

print("\n=== 预处理后数据可视化 ===")


def plot_post_processing(transformed_data):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 主成分散点图
    if transformed_data.shape[1] >= 2:
        axes[0].scatter(transformed_data[:, 0], transformed_data[:, 1], alpha=0.6)
        axes[0].set_title('前两个主成分分布', fontsize=14)
        axes[0].set_xlabel('第一主成分')
        axes[0].set_ylabel('第二主成分')
        axes[0].grid(True, alpha=0.3)

    # 箱线图展示特征分布
    max_features_to_show = min(8, transformed_data.shape[1])
    sns.boxplot(data=pd.DataFrame(transformed_data[:, :max_features_to_show]), ax=axes[1])
    axes[1].set_title('预处理后特征分布', fontsize=14)
    axes[1].set_xlabel('特征')
    axes[1].set_ylabel('值')

    plt.suptitle("预处理后数据分布", fontsize=16)
    plt.tight_layout()
    plt.show()


plot_post_processing(transformed_features)

# ======================================
# 肘部法则分析 - 确定最佳聚类数
# ======================================

print("\n=== 使用肘部法则确定最佳聚类数 ===")

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# 准备数据
X = transformed_features

# 设置测试的聚类范围
cluster_range = range(2, 11)
inertia_values = []  # 保存惯性值
silhouette_scores = []  # 保存轮廓系数
davies_bouldin_scores = []  # 保存DB指数

for k in cluster_range:
    # 创建并训练KMeans模型
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)

    # 获取聚类标签
    labels = kmeans.labels_

    # 计算评价指标
    inertia_values.append(kmeans.inertia_)

    # 轮廓系数需要至少2个聚类
    if k > 1:
        silhouette_scores.append(silhouette_score(X, labels))
        davies_bouldin_scores.append(davies_bouldin_score(X, labels))
    else:
        silhouette_scores.append(0)
        davies_bouldin_scores.append(0)

# 绘制肘部法则图
plt.figure(figsize=(15, 10))

# 1. 惯性图 - 肘部法则
plt.subplot(2, 2, 1)
plt.plot(cluster_range, inertia_values, 'bo-', markersize=8)
plt.xlabel('聚类数量 (K)', fontsize=12)
plt.ylabel('惯性 (Inertia)', fontsize=12)
plt.title('肘部法则 - KMeans', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# 标记可能的肘点
elbow_point = 4
for i in range(1, len(inertia_values) - 1):
    prev_diff = inertia_values[i - 1] - inertia_values[i]
    next_diff = inertia_values[i] - inertia_values[i + 1]

    if next_diff / prev_diff < 0.5:
        elbow_point = i + 2
        plt.scatter(elbow_point, inertia_values[elbow_point - 2], s=200,
                    facecolors='none', edgecolors='r', linewidths=2)
        plt.annotate(f'可能肘点 (K={elbow_point})',
                     xy=(elbow_point, inertia_values[elbow_point - 2]),
                     xytext=(elbow_point + 0.5, inertia_values[elbow_point - 2] + 100),
                     arrowprops=dict(facecolor='red', shrink=0.05))
        break

# 2. 轮廓系数图
plt.subplot(2, 2, 2)
plt.plot(cluster_range, silhouette_scores, 'go-', markersize=8)
plt.xlabel('聚类数量 (K)', fontsize=12)
plt.ylabel('轮廓系数', fontsize=12)
plt.title('轮廓系数 vs 聚类数量', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# 标记最高点
max_sil_idx = np.argmax(silhouette_scores)
max_sil_k = max_sil_idx + 2
plt.scatter(max_sil_k, silhouette_scores[max_sil_idx], s=200,
            facecolors='none', edgecolors='g', linewidths=2)
plt.annotate(f'最佳轮廓系数 (K={max_sil_k})',
             xy=(max_sil_k, silhouette_scores[max_sil_idx]),
             xytext=(max_sil_k + 0.5, silhouette_scores[max_sil_idx] - 0.02),
             arrowprops=dict(facecolor='green', shrink=0.05))

# 3. Davies-Bouldin指数图
plt.subplot(2, 2, 3)
plt.plot(cluster_range, davies_bouldin_scores, 'mo-', markersize=8)
plt.xlabel('聚类数量 (K)', fontsize=12)
plt.ylabel('Davies-Bouldin指数', fontsize=12)
plt.title('Davies-Bouldin指数 vs 聚类数量', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# 标记最低点
min_db_idx = np.argmin(davies_bouldin_scores)
min_db_k = min_db_idx + 2
plt.scatter(min_db_k, davies_bouldin_scores[min_db_idx], s=200,
            facecolors='none', edgecolors='m', linewidths=2)
plt.annotate(f'最佳DB指数 (K={min_db_k})',
             xy=(min_db_k, davies_bouldin_scores[min_db_idx]),
             xytext=(min_db_k + 0.5, davies_bouldin_scores[min_db_idx] + 0.05),
             arrowprops=dict(facecolor='purple', shrink=0.05))

# 4. 综合推荐
plt.subplot(2, 2, 4)
plt.axis('off')

# 计算综合得分
norm_inertia = (inertia_values - np.min(inertia_values)) / (np.max(inertia_values) - np.min(inertia_values))
norm_sil = (silhouette_scores - np.min(silhouette_scores)) / (np.max(silhouette_scores) - np.min(silhouette_scores))
norm_db = (davies_bouldin_scores - np.min(davies_bouldin_scores)) / (
            np.max(davies_bouldin_scores) - np.min(davies_bouldin_scores))

composite_scores = norm_sil - norm_db - norm_inertia
best_k_idx = np.argmax(composite_scores)
best_k = best_k_idx + 2

# 显示推荐结果
plt.text(0.1, 0.8, "聚类数量综合分析", fontsize=16, fontweight='bold')
plt.text(0.1, 0.6, f"肘部法则建议: K = {elbow_point}", fontsize=14)
plt.text(0.1, 0.5, f"轮廓系数建议: K = {max_sil_k}", fontsize=14)
plt.text(0.1, 0.4, f"DB指数建议: K = {min_db_k}", fontsize=14)
plt.text(0.1, 0.3, f"综合分析建议: K = {best_k}", fontsize=14, fontweight='bold', color='red')
plt.text(0.1, 0.1, "推荐使用综合分析建议的K值进行后续聚类", fontsize=12)

plt.tight_layout()
plt.show()

# 使用最佳K值重新聚类
print(f"\n使用最佳聚类数 K={best_k} 重新进行聚类...")
best_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
best_labels = best_kmeans.fit_predict(X)

# 评估新聚类质量
if len(np.unique(best_labels)) > 1:
    best_sil_score = silhouette_score(X, best_labels)
    best_ch_score = calinski_harabasz_score(X, best_labels)
else:
    best_sil_score = -1
    best_ch_score = 0

print(f"新聚类轮廓系数: {best_sil_score:.3f}")
print(f"新聚类Calinski-Harabasz指数: {best_ch_score:.3f}")

# ======================================
# 聚类可视化
# ======================================

print("\n=== 聚类可视化 ===")


def visualize_clusters(features, labels, method='pca'):
    fig = plt.figure(figsize=(10, 8))

    if method == 'pca':
        reducer = PCA(n_components=2)
        title = "PCA聚类可视化"
    else:
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        title = "t-SNE聚类可视化"

    reduced_data = reducer.fit_transform(features)

    scatter = plt.scatter(
        reduced_data[:, 0], reduced_data[:, 1],
        c=labels, cmap='viridis', alpha=0.7, s=50
    )

    plt.title(f"{title}\n轮廓系数: {best_sil_score:.3f} | CH指数: {best_ch_score:.3f}", fontsize=14)
    plt.colorbar(scatter, label='聚类标签')
    plt.xlabel(f"{method.upper()} 成分 1")
    plt.ylabel(f"{method.upper()} 成分 2")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


print("使用PCA可视化新聚类...")
visualize_clusters(X, best_labels, method='pca')

print("使用t-SNE可视化新聚类...")
visualize_clusters(X, best_labels, method='tsne')

# ======================================
# 聚类特征分析
# ======================================

print("\n=== 聚类特征分析 ===")


def analyze_clusters(df, labels, audio_features):
    # 添加聚类标签到原始数据
    df_clustered = df.copy()
    df_clustered['cluster'] = labels

    # 计算每个聚类的特征均值
    cluster_means = df_clustered.groupby('cluster')[audio_features].mean()

    # 1. 雷达图
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True)

    angles = np.linspace(0, 2 * np.pi, len(audio_features), endpoint=False).tolist()
    angles += angles[:1]

    for cluster in sorted(df_clustered['cluster'].unique()):
        # 归一化特征值
        values = (cluster_means.loc[cluster] - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
        values = values.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=f'聚类 {cluster}')
        ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(audio_features, fontsize=10)
    ax.set_title('各聚类音频特征分析 (归一化)', size=14, pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()

    # 2. 其他分析图表
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)

    # 特征热图
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

    # 聚类大小分布
    ax2 = fig.add_subplot(gs[0, 1])
    cluster_counts = df_clustered['cluster'].value_counts().sort_index()
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='viridis', ax=ax2)
    ax2.set_title('聚类大小分布', fontsize=14)
    ax2.set_xlabel('聚类')
    ax2.set_ylabel('歌曲数量')

    # 流派分布热图
    ax3 = fig.add_subplot(gs[1, :])
    genre_cluster = pd.crosstab(df_clustered['artist_top_genre'], df_clustered['cluster'])
    genre_cluster = genre_cluster[genre_cluster.sum(axis=1) > 5]  # 过滤小流派
    sns.heatmap(genre_cluster, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_title('各聚类中音乐流派分布', fontsize=14)
    ax3.set_xlabel('聚类')
    ax3.set_ylabel('音乐流派')

    plt.suptitle("聚类分析结果", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    return cluster_means


print("分析聚类特征...")
cluster_analysis = analyze_clusters(df, best_labels, audio_features)
print("\n聚类特征均值:")
print(cluster_analysis)

# ======================================
# 分析结果总结
# ======================================

print("\n=== 分析结果总结 ===")
print(f"1. 最佳聚类数: K = {best_k}")
print(f"2. 轮廓系数: {best_sil_score:.3f} (值在-1到1之间，越接近1表示聚类效果越好)")
print(f"3. Calinski-Harabasz指数: {best_ch_score:.3f} (值越高表示聚类效果越好)")

print("\n4. 聚类大小分布:")
cluster_counts = df.assign(cluster=best_labels)['cluster'].value_counts().sort_index()
for cluster, count in cluster_counts.items():
    print(f"   聚类 {cluster}: {count} 首歌曲 ({count / len(df) * 100:.1f}%)")

print("\n5. 聚类特征总结:")
for cluster in cluster_analysis.index:
    top_features = cluster_analysis.loc[cluster].nlargest(2).index.tolist()
    print(f"   聚类 {cluster} 最显著特征: {', '.join(top_features)}")

print("\n分析完成!")