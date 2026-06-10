import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import io
import base64


def preprocess_data(df, features):
    """Preprocess data for clustering."""
    data = df[features].copy()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler


def perform_kmeans(data, n_clusters=3, random_state=42):
    """Perform K-Means clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(data)
    return labels, kmeans


def perform_pca(data, n_components=2):
    """Perform PCA for visualization."""
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(data)
    return pca_data, pca


def get_cluster_insights(df, labels, features):
    """Generate insights for each cluster."""
    df_copy = df.copy()
    df_copy['Cluster'] = labels
    
    insights = []
    for cluster_id in sorted(df_copy['Cluster'].unique()):
        cluster_data = df_copy[df_copy['Cluster'] == cluster_id]
        
        n_customers = len(cluster_data)
        avg_income = cluster_data[features[0]].mean()
        avg_spending = cluster_data[features[1]].mean()
        
        if avg_spending >= 66:
            segment = "High Spenders"
            description = "High spending behavior - potential premium customers"
            suggestions = ["Premium offers", "VIP discounts", "Exclusive products", "Early access to sales"]
        elif avg_spending >= 33:
            segment = "Medium Spenders"
            description = "Moderate spending behavior - growth potential"
            suggestions = ["Product bundles", "Loyalty rewards", "Cross-sell recommendations", "Seasonal promotions"]
        else:
            segment = "Budget Customers"
            description = "Low spending behavior - price sensitive"
            suggestions = ["Discount campaigns", "Seasonal offers", "Bundle deals", "Free shipping thresholds"]
        
        insights.append({
            'cluster_id': int(cluster_id),
            'segment': segment,
            'n_customers': int(n_customers),
            'avg_income': round(float(avg_income), 2),
            'avg_spending': round(float(avg_spending), 2),
            'description': description,
            'suggestions': suggestions
        })
    
    return insights


def create_pca_plot(pca_data, labels, n_clusters):
    """Create PCA visualization plot."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = ['#4F46E5', '#06B6D4', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']
    
    for i in range(n_clusters):
        cluster_points = pca_data[labels == i]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   c=colors[i % len(colors)], label=f'Cluster {i}', 
                   alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
    
    ax.set_xlabel('Principal Component 1', fontsize=12, fontweight='500')
    ax.set_ylabel('Principal Component 2', fontsize=12, fontweight='500')
    ax.set_title('Customer Segments - PCA Visualization', fontsize=14, fontweight='600', pad=20)
    ax.legend(fontsize=10, framealpha=0.9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    
    return buf


def get_dataframe_download_link(df, filename="clustered_customers.csv"):
    """Generate CSV download link."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-link">Download CSV</a>'
    return href


def get_image_download_link(buf, filename="pca_visualization.png"):
    """Generate image download link."""
    b64 = base64.b64encode(buf.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}" class="download-link">Download Image</a>'
    return href


def calculate_kpis(df):
    """Calculate key performance indicators."""
    total_customers = len(df)
    avg_income = df['Annual Income (k$)'].mean()
    avg_spending = df['Spending Score (1-100)'].mean()
    highest_spender = df.loc[df['Spending Score (1-100)'].idxmax()]
    lowest_spender = df.loc[df['Spending Score (1-100)'].idxmin()]
    
    return {
        'total_customers': total_customers,
        'avg_income': round(avg_income, 1),
        'avg_spending': round(avg_spending, 1),
        'highest_spender': {
            'id': highest_spender['CustomerID'],
            'score': highest_spender['Spending Score (1-100)'],
            'income': highest_spender['Annual Income (k$)']
        },
        'lowest_spender': {
            'id': lowest_spender['CustomerID'],
            'score': lowest_spender['Spending Score (1-100)'],
            'income': lowest_spender['Annual Income (k$)']
        }
    }