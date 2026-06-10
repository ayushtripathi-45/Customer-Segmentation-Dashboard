import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import io
import base64
import sys
import os
import threading, time

# Fix: Suppress Windows ProactorEventLoop connection reset error
# This is a known issue with asyncio on Windows and does not affect the app
if sys.platform == "win32":
    import asyncio
    from asyncio.proactor_events import _ProactorBasePipeTransport
    from functools import wraps

    def _silence_connection_reset(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except (ConnectionResetError, OSError):
                pass
        return wrapper

    # Apply patch to suppress the error in server's connection handler
    _ProactorBasePipeTransport.__del__ = _silence_connection_reset(_ProactorBasePipeTransport.__del__)
    _call_connection_lost_orig = _ProactorBasePipeTransport._call_connection_lost
    _ProactorBasePipeTransport._call_connection_lost = _silence_connection_reset(_call_connection_lost_orig)

# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Custom CSS
# ============================================
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #F8FAFC;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        border-radius: 8px;
        border: none;
        background-color: transparent;
        color: #475569;
        font-weight: 500;
        font-size: 14px;
        padding: 10px 16px;
        text-align: left;
        transition: all 0.2s ease;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #F1F5F9;
        color: #4F46E5;
    }
    
    /* Cards */
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.03);
        transition: all 0.2s ease;
        border: 1px solid #E2E8F0;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transform: translateY(-2px);
    }
    
    .metric-card .title {
        font-size: 13px;
        color: #64748B;
        font-weight: 500;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-card .value {
        font-size: 28px;
        font-weight: 700;
        color: #1E293B;
        line-height: 1.2;
    }
    
    .metric-card .subtitle {
        font-size: 12px;
        color: #94A3B8;
        margin-top: 4px;
    }
    
    /* Section Headers */
    .section-header {
        font-family: 'Poppins', sans-serif;
        font-size: 22px;
        font-weight: 600;
        color: #1E293B;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid #E2E8F0;
    }
    
    /* Insights Panel */
    .insight-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
        border-left: 4px solid #4F46E5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .insight-card h4 {
        color: #4F46E5;
        margin-bottom: 8px;
        font-weight: 600;
    }
    
    .insight-card .description {
        color: #475569;
        font-size: 14px;
        margin-bottom: 12px;
    }
    
    .insight-card .stats {
        display: flex;
        gap: 24px;
        margin-bottom: 12px;
    }
    
    .insight-card .stat-item {
        text-align: center;
    }
    
    .insight-card .stat-value {
        font-size: 20px;
        font-weight: 700;
        color: #1E293B;
    }
    
    .insight-card .stat-label {
        font-size: 12px;
        color: #94A3B8;
    }
    
    .insight-card .suggestions {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 12px;
    }
    
    .insight-card .suggestions-title {
        font-size: 12px;
        font-weight: 600;
        color: #475569;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .insight-card .suggestion-tag {
        display: inline-block;
        background-color: #EEF2FF;
        color: #4F46E5;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
        margin: 4px 4px 4px 0;
    }
    
    /* Upload Area */
    .upload-section {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 32px;
        text-align: center;
        border: 2px dashed #CBD5E1;
        transition: all 0.2s ease;
    }
    
    .upload-section:hover {
        border-color: #4F46E5;
    }
    
    /* Buttons */
    .stButton > button[kind="primary"] {
        background-color: #4F46E5;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 24px;
        transition: all 0.2s ease;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #4338CA;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    }
    
    /* DataFrame styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Progress bars and other elements */
    .stProgress > div > div {
        background-color: #4F46E5;
    }
    
    /* Success messages */
    .stSuccess {
        background-color: #ECFDF5;
        border-left: 4px solid #10B981;
    }
    
    /* Info messages */
    .stInfo {
        background-color: #EEF2FF;
        border-left: 4px solid #4F46E5;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# ML Functions
# ============================================
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
            description = "High spending behavior - potential premium customers. These customers have the highest engagement and represent significant revenue potential."
            suggestions = ["Premium offers", "VIP discounts", "Exclusive products", "Early access to sales"]
        elif avg_spending >= 33:
            segment = "Medium Spenders"
            description = "Moderate spending behavior - significant growth potential. These customers can be nurtured to become high-value customers."
            suggestions = ["Product bundles", "Loyalty rewards", "Cross-sell recommendations", "Seasonal promotions"]
        else:
            segment = "Budget Customers"
            description = "Low spending behavior - price sensitive. These customers respond well to value-driven campaigns and deals."
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

# ============================================
# Session State Initialization
# ============================================
def init_session_state():
    """Initialize session state variables."""
    if 'page' not in st.session_state:
        st.session_state.page = 'Dashboard'
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'clustered_df' not in st.session_state:
        st.session_state.clustered_df = None
    if 'labels' not in st.session_state:
        st.session_state.labels = None
    if 'insights' not in st.session_state:
        st.session_state.insights = None
    if 'n_clusters' not in st.session_state:
        st.session_state.n_clusters = 3

# ============================================
# Sidebar Navigation
# ============================================
def sidebar_navigation():
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h2 style="color: #4F46E5; font-family: 'Poppins', sans-serif; font-weight: 700 Chern_and; font-size: 24px; margin-bottom: 4px;">🎯 Segmentation</h2>
        <p style="color: #94A3B8; font-size: 12px; margin-top: 0;">Customer Analytics Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    pages = ['Dashboard', 'Upload Data', 'Segmentation', 'Insights']
    
    for page in pages:
        if st.sidebar.button(page, key=f"nav_{page}", use_container_width=True):
            st.session_state.page = page
            st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="padding: 10px; text-align: center;">
        <p style="color: #94A3B8; font-size: 11px;">Built with ❤️ for ML learners</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# Dashboard Page
# ============================================
def dashboard_page():
    st.markdown('<h1 style="font-family: Poppins, sans-serif; color: #1E293B; font-weight: 700;">Dashboard</h1>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.info("📁 No data loaded. Please upload a dataset from the 'Upload Data' page.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div style="background: white; border-radius: 12px; padding: 24px; text-align: center; border: 1px solid #E2E8F0;">
                <div style="font-size: 40px; margin-bottom: 12px;">📤</div>
                <h4 style="color: #1E293B; margin-bottom: 8px;">Upload Data</h4>
                <p style="color: #64748B; font-size: 14px;">Import your customer dataset in CSV format</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style="background: white; border-radius: 12px; padding: 24px; text-align: center; border: 1px solid #E2E8F0;">
                <div style="font-size: 40px; margin-bottom: 12px;">🤖</div>
                <h4 style="color: #1E293B; margin-bottom: 8px;">Segment</h4>
                <p style="color: #64748B; font-size: 14px;">Apply K-Means clustering to group customers</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div style="background: white; border-radius: 12px; padding: 24px; text-align: center; border: 1px solid #E2E8F0;">
                <div style="font-size: 40px; margin-bottom: 12px;">💡</div>
                <h4 style="color: #1E293B; margin-bottom: 8px;">Get Insights</h4>
                <p style="color: #64748B; font-size: 14px;">View actionable customer insights</p>
            </div>
            """, unsafe_allow_html=True)
        return
    
    df = st.session_state.df
    kpis = calculate_kpis(df)
    
    # KPI Cards Row 1
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="title">Total Customers</div>
            <div class="value">{kpis['total_customers']:,}</div>
            <div class="subtitle">Unique customer records</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="title">Average Income</div>
            <div class="value">${kpis['avg_income']:,}k</div>
            <div class="subtitle">Annual income (thousands)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="title">Avg Spending Score</div>
            <div class="value">{kpis['avg_spending']}</div>
            <div class="subtitle">Out of 100</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # KPI Cards Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="title">Highest Spender</div>
            <div class="value" style="font-size: 22px;">{kpis['highest_spender']['id']}</div>
            <div class="subtitle">Score: {kpis['highest_spender']['score']} | Income: ${kpis['highest_spender']['income']}k</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="title">Lowest Spender</div>
            <div class="value" style="font-size: 22px;">{kpis['lowest_spender']['id']}</div>
            <div class="subtitle">Score: {kpis['lowest_spender']['score']} | Income: ${kpis['lowest_spender']['income']}k</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Data Preview
    st.markdown('<div class="section-header">Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)
    
    # Quick Stats
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Dataset Shape</div>', unsafe_allow_html=True)
        st.write(f"Rows: `{df.shape[0]}` | Columns: `{df.shape[1]}`")
    with col2:
        st.markdown('<div class="section-header">Column Names</div>', unsafe_allow_html=True)
        st.write(", ".join([f"`{col}`" for col in df.columns]))

# ============================================
# Upload Data Page
# ============================================
def upload_data_page():
    st.markdown('<h1 style="font-family: Poppins, sans-serif; color: #1E293B; font-weight: 700;">Upload Data</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #EEF2FF; border-left: 4px solid #4F46E5; padding: 16px; border-radius: 0 8px 8px 0; margin-bottom: 24px;">
        <p style="color: #4F46E5; margin: 0; font-weight: 500;">📋 Upload a CSV file with customer data</p>
        <p style="color: #475569; margin: 8px 0 0 0; font-size: 14px;">Expected columns: <strong>CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", label_visibility="collapsed")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("📁 Upload File", use_container_width=True, type="primary"):
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.df = df
                    st.session_state.clustered_df = None
                    st.session_state.labels = None
                    st.session_state.insights = None
                    st.success(f"✅ Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
            else:
                st.warning("Please select a file first.")
    
    with col2:
        if st.button("🎲 Load Sample Data", use_container_width=True):
            try:
                df = pd.read_csv("sample_customers.csv")
                st.session_state.df = df
                st.session_state.clustered_df = None
                st.session_state.labels = None
                st.session_state.insights = None
                st.success(f"✅ Successfully loaded sample data with {df.shape[0]} rows and {df.shape[1]} columns!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error loading sample data: {str(e)}")
    
    if st.session_state.df is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Current Dataset</div>', unsafe_allow_html=True)
        st.dataframe(st.session_state.df.head(20), use_container_width=True, hide_index=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", st.session_state.df.shape[0])
        with col2:
            st.metric("Total Columns", st.session_state.df.shape[1])
        with col3:
            st.metric("Memory Usage", f"{st.session_state.df.memory_usage(deep=True).sum() / 1024:.1f} KB")

# ============================================
# Segmentation Page
# ============================================
def segmentation_page():
    st.markdown('<h1 style="font-family: Poppins, sans-serif; color: #1E293B; font-weight: 700;">Customer Segmentation</h1>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first from the 'Upload Data' page.")
        return
    
    df = st.session_state.df
    
    # Cluster Configuration
    st.markdown('<div class="section-header">Cluster Configuration</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        n_clusters = st.slider("Number of Clusters", min_value=2, max_value=6, value=st.session_state.n_clusters, step=1)
        st.session_state.n_clusters = n_clusters
    
    with col2:
        features = ['Annual Income (k$)', 'Spending Score (1-100)']
        st.write("**Features used:**")
        st.write(", ".join([f"`{f}`" for f in features]))
    
    with col3:
        st.write(" ")
        st.write(" ")
        run_clustering = st.button("🚀 Run Clustering", use_container_width=True, type="primary")
    
    if run_clustering:
        with st.spinner("Running K-Means clustering..."):
            try:
                # Preprocess
                scaled_data, scaler = preprocess_data(df, features)
                
                # K-Means
                labels, kmeans = perform_kmeans(scaled_data, n_clusters=n_clusters)
                
                # PCA
                pca_data, pca = perform_pca(scaled_data)
                
                # Update session state
                st.session_state.labels = labels
                st.session_state.pca_data = pca_data
                
                # Create clustered dataframe
                clustered_df = df.copy()
                clustered_df['Cluster'] = labels
                st.session_state.clustered_df = clustered_df
                
                # Generate insights
                st.session_state.insights = get_cluster_insights(df, labels, features)
                
                st.success(f"✅ Successfully segmented customers into {n_clusters} clusters!")
                
            except Exception as e:
                st.error(f"❌ Error during clustering: {str(e)}")
    
    # Show results if available
    if st.session_state.clustered_df is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Cluster Distribution</div>', unsafe_allow_html=True)
        
        clustered_df = st.session_state.clustered_df
        cluster_counts = clustered_df['Cluster'].value_counts().sort_index()
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.bar_chart(cluster_counts, use_container_width=True)
        with col2:
            st.write("**Cluster Sizes:**")
            for cluster_id, count in cluster_counts.items():
                st.write(f"Cluster {cluster_id}: `{count}` customers")
        with col3:
            st.write("**Percentage:**")
            total = len(clustered_df)
            for cluster_id, count in cluster_counts.items():
                percentage = (count / total) * 100
                st.write(f"Cluster {cluster_id}: `{percentage:.1f}%`")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">PCA Visualization</div>', unsafe_allow_html=True)
        
        if hasattr(st.session_state, 'pca_data'):
            buf = create_pca_plot(st.session_state.pca_data, st.session_state.labels, st.session_state.n_clusters)
            st.image(buf, use_container_width=True)
            
            # Download button for PCA plot
            st.download_button(
                label="⬇️ Download PCA Plot",
                data=buf.getvalue(),
                file_name="pca_visualization.png",
                mime="image/png",
                use_container_width=True
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Clustered Data</div>', unsafe_allow_html=True)
        st.dataframe(clustered_df, use_container_width=True, hide_index=True)
        
        # Download button for clustered data
        csv = clustered_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Clustered Data (CSV)",
            data=csv,
            file_name="clustered_customers.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Auto shutdown after results displayed
        if st.session_state.clustered_df is not None and not st.session_state.get('shutdown_initiated', False):
            st.success("All results processed – the server will shut down shortly.")
            st.session_state.shutdown_initiated = True
            def _delayed_exit():
                time.sleep(2)
                sys.exit()
            threading.Thread(target=_delayed_exit, daemon=True).start()

# ============================================
# Insights Page
# ============================================
def insights_page():
    st.markdown('<h1 style="font-family: Poppins, sans-serif; color: #1E293B; font-weight: 700;">Customer Insights</h1>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first from the 'Upload Data' page.")
        return
    
    if st.session_state.insights is None:
        st.info("🔬 Run the segmentation on the 'Segmentation' page to generate insights.")
        return
    
    st.markdown("""
    <div style="background-color: #EEF2FF; border-left: 4px solid #4F46E5; padding: 16px; border-radius: 0 8px 8px 0; margin-bottom: 24px;">
        <p style="color: #4F46E5; margin: 0; font-weight: 500;">💡 Customer Segmentation Insights</p>
        <p style="color: #475569; margin: 8px 0 0 0; font-size: 14px;">Detailed analysis and marketing recommendations for each customer segment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    for insight in st.session_state.insights:
        st.markdown(f"""
        <div class="insight-card">
            <h4>🎯 Cluster {insight['cluster_id']} - {insight['segment']}</h4>
            <div class="description">{insight['description']}</div>
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-value">{insight['n_customers']}</div>
                    <div class="stat-label">Customers</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${insight['avg_income']}k</div>
                    <div class="stat-label">Avg Income</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{insight['avg_spending']}</div>
                    <div class="stat-label">Avg Spending</div>
                </div>
            </div>
            <div class="suggestions">
                <div class="suggestions-title">Marketing Suggestions</div>
                {''.join([f'<span class="suggestion-tag">{suggestion}</span>' for suggestion in insight['suggestions']])}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Marketing Summary
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Marketing Strategy Summary</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%); border-radius: 12px; padding: 24px; color: white;">
            <h4 style="margin-bottom: 12px; color: white;">High Spenders</h4>
            <p style="font-size: 14px; margin-bottom: 12px; opacity: 0.9;">Premium customers with high engagement. Focus on retention and exclusivity.</p>
            <ul style="font-size: 13px; padding-left: 16px; margin-bottom: 0;">
                <li>Premium offers</li>
                <li>VIP discounts</li>
                <li>Exclusive products</li>
                <li>Early access to sales</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #06B6D4 0%, #0891B2 100%); border-radius: 12px; padding: 24px; color: white;">
            <h4 style="margin-bottom: 12px; color: white;">Medium Spenders</h4>
            <p style="font-size: 14px; margin-bottom: 12px; opacity: 0.9;">Growth potential customers. Nurture with targeted campaigns.</p>
            <ul style="font-size: 13px; padding-left: 16px; margin-bottom: 0;">
                <li>Product bundles</li>
                <li>Loyalty rewards</li>
                <li>Cross-sell recommendations</li>
                <li>Seasonal promotions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #10B981 0%, #059669 100%); border-radius: 12px; padding: 24px; color: white;">
            <h4 style="margin-bottom: 12px; color: white;">Budget Customers</h4>
            <p style="font-size: 14px; margin-bottom: 12px; opacity: 0.9;">Price-sensitive segment. Focus on value and deals.</p>
            <ul style="font-size: 13px; padding-left: 16px; margin-bottom: 0;">
                <li>Discount campaigns</li>
                <li>Seasonal offers</li>
                <li>Bundle deals</li>
                <li>Free shipping thresholds</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Download Section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Download Results</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.clustered_df is not None:
            csv = st.session_state.clustered_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Clustered Data (CSV)",
                data=csv,
                file_name="clustered_customers.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        if hasattr(st.session_state, 'pca_data'):
            try:
                buf = create_pca_plot(st.session_state.pca_data, st.session_state.labels, st.session_state.n_clusters)
                st.download_button(
                    label="⬇️ Download PCA Plot (PNG)",
                    data=buf.getvalue(),
                    file_name="pca_visualization.png",
                    mime="image/png",
                    use_container_width=True
                )
            except Exception:
                st.button("⬇️ Download PCA Plot (PNG)", disabled=True, use_container_width=True)

    # Auto shutdown after results displayed
    if st.session_state.clustered_df is not None and not st.session_state.get('shutdown_initiated', False):
        st.success("All results processed – the server will shut down shortly.")
        st.session_state.shutdown_initiated = True
        def _delayed_exit():
            time.sleep(2)
            sys.exit()
        threading.Thread(target=_delayed_exit, daemon=True).start()

# ============================================
# Main App
# ============================================
def main():
    load_css()
    init_session_state()
    sidebar_navigation()
    
    page = st.session_state.page
    
    if page == 'Dashboard':
        dashboard_page()
    elif page == 'Upload Data':
        upload_data_page()
    elif page == 'Segmentation':
        segmentation_page()
    elif page == 'Insights':
        insights_page()

if __name__ == "__main__":
    main()
