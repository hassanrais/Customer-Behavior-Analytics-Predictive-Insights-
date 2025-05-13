import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

st.set_page_config(page_title="Online Shoppers Analysis", layout="wide")

# Project Title and Members
st.title("Online Shoppers Intention Analysis Dashboard")
st.markdown("---")
st.markdown("### A PROJECT BY")
st.markdown("""
- Hassan Rais (2022212)
- Hamza Zaidi (2022379)
""")
st.markdown("---")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=['csv'])

if uploaded_file is not None:
    # Load and preprocess data
    @st.cache_data
    def load_data(uploaded_file):
        df = pd.read_csv(uploaded_file)
        df['Month'] = LabelEncoder().fit_transform(df['Month'])
        df['VisitorType'] = LabelEncoder().fit_transform(df['VisitorType'])
        df['Weekend'] = df['Weekend'].astype(bool).astype(int)
        df['Revenue'] = df['Revenue'].astype(bool).astype(int)
        return df

    df = load_data(uploaded_file)

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["EDA", "Data Overview", "Classification Analysis", "Clustering Analysis", "Association Rules"])

    if page == "EDA":
        st.header("Exploratory Data Analysis")
        
        # 1. Revenue Distribution
        st.subheader("1. Revenue Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(data=df, x='Revenue', ax=ax)
        plt.title('Revenue Distribution (0 = No Purchase, 1 = Purchase)')
        st.pyplot(fig)
        st.markdown("""
        **Business Recommendations:**
        - If the non-purchase rate is high, consider implementing:
          - Better product recommendations
          - More attractive pricing strategies
          - Improved checkout process
          - Enhanced product descriptions and images
        """)
        
        # 2. Monthly Distribution
        st.subheader("2. Monthly Distribution")
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.countplot(data=df, x='Month', hue='Revenue', ax=ax)
        plt.title('Monthly Distribution by Revenue')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.markdown("""
        **Business Recommendations:**
        - Identify peak shopping months and:
          - Increase inventory during high-traffic months
          - Plan special promotions for low-traffic months
          - Adjust marketing budget allocation based on seasonal patterns
          - Consider seasonal product offerings
        """)
        
        # 3. Visitor Type Analysis
        st.subheader("3. Visitor Type Analysis")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(data=df, x='VisitorType', hue='Revenue', ax=ax)
        plt.title('Visitor Type Distribution by Revenue')
        st.pyplot(fig)
        st.markdown("""
        **Business Recommendations:**
        - For new visitors:
          - Implement welcome discounts
          - Create engaging onboarding experiences
          - Provide clear value propositions
        - For returning visitors:
          - Offer loyalty programs
          - Send personalized recommendations
          - Provide exclusive deals
        """)
        
        # 4. Weekend vs Weekday Analysis
        st.subheader("4. Weekend vs Weekday Analysis")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(data=df, x='Weekend', hue='Revenue', ax=ax)
        plt.title('Weekend vs Weekday Distribution by Revenue')
        st.pyplot(fig)
        st.markdown("""
        **Business Recommendations:**
        - Weekend shoppers:
          - Schedule weekend-specific promotions
          - Ensure weekend customer service availability
          - Offer weekend-only deals
        - Weekday shoppers:
          - Target office workers with lunch-hour specials
          - Implement B2B promotions
          - Focus on quick checkout options
        """)
        
        # 5. Duration Analysis
        st.subheader("5. Duration Analysis")
        duration_cols = ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for idx, col in enumerate(duration_cols):
            sns.boxplot(data=df, x='Revenue', y=col, ax=axes[idx])
            axes[idx].set_title(f'{col} by Revenue')
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("""
        **Business Recommendations:**
        - For high administrative duration:
          - Simplify navigation structure
          - Improve site search functionality
          - Add quick access menus
        - For high informational duration:
          - Enhance product information quality
          - Add video content
          - Implement live chat support
        - For high product-related duration:
          - Improve product filtering
          - Add comparison tools
          - Enhance product recommendations
        """)
        
        # 6. Page Value Analysis
        st.subheader("6. Page Value Analysis")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df, x='Revenue', y='PageValues', ax=ax)
        plt.title('Page Values Distribution by Revenue')
        st.pyplot(fig)
        st.markdown("""
        **Business Recommendations:**
        - High-value pages:
          - Optimize for conversion
          - Add trust signals
          - Implement A/B testing
        - Low-value pages:
          - Review and improve content
          - Add clear calls-to-action
          - Consider page redesign
        """)
        
        # 7. Bounce and Exit Rates
        st.subheader("7. Bounce and Exit Rates Analysis")
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        sns.boxplot(data=df, x='Revenue', y='BounceRates', ax=axes[0])
        axes[0].set_title('Bounce Rates by Revenue')
        sns.boxplot(data=df, x='Revenue', y='ExitRates', ax=axes[1])
        axes[1].set_title('Exit Rates by Revenue')
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("""
        **Business Recommendations:**
        - For high bounce rates:
          - Improve page load speed
          - Enhance mobile responsiveness
          - Make content more engaging
        - For high exit rates:
          - Add exit-intent popups
          - Implement retargeting strategies
          - Offer last-minute incentives
        """)
        
        # 8. Correlation Analysis
        st.subheader("8. Correlation Analysis")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
        plt.title('Correlation Matrix')
        st.pyplot(fig)
        st.markdown("""
        **Business Recommendations:**
        - Strong positive correlations:
          - Bundle related products
          - Create cross-selling opportunities
          - Develop targeted marketing campaigns
        - Strong negative correlations:
          - Address potential pain points
          - Improve user experience
          - Consider product/service adjustments
        """)
        
        # 9. Operating Systems and Browser Analysis
        st.subheader("9. Operating Systems and Browser Analysis")
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        sns.countplot(data=df, x='OperatingSystems', hue='Revenue', ax=axes[0])
        axes[0].set_title('Operating Systems Distribution by Revenue')
        sns.countplot(data=df, x='Browser', hue='Revenue', ax=axes[1])
        axes[1].set_title('Browser Distribution by Revenue')
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("""
        **Business Recommendations:**
        - For popular operating systems:
          - Optimize site performance
          - Ensure compatibility
          - Test on all major platforms
        - For popular browsers:
          - Ensure cross-browser compatibility
          - Optimize for each browser
          - Monitor browser-specific issues
        """)
        
        # 10. Region Analysis
        st.subheader("10. Region Analysis")
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.countplot(data=df, x='Region', hue='Revenue', ax=ax)
        plt.title('Region Distribution by Revenue')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.markdown("""
        **Business Recommendations:**
        - For high-performing regions:
          - Increase marketing investment
          - Expand product offerings
          - Consider local partnerships
        - For low-performing regions:
          - Conduct market research
          - Adjust pricing strategy
          - Consider cultural adaptations
          - Implement region-specific promotions
        """)

    elif page == "Data Overview":
        st.header("Data Overview")
        
        # Basic dataset info
        st.subheader("Dataset Information")
        st.write(f"Number of rows: {df.shape[0]}")
        st.write(f"Number of columns: {df.shape[1]}")
        
        # Display first few rows
        st.subheader("Sample Data")
        st.dataframe(df.head())
        
        # Basic statistics
        st.subheader("Basic Statistics")
        st.write(df.describe())
        
        # Revenue distribution
        st.subheader("Revenue Distribution")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(data=df, x='Revenue', ax=ax)
        plt.title('Revenue Distribution (0 = No Purchase, 1 = Purchase)')
        st.pyplot(fig)

    elif page == "Classification Analysis":
        st.header("Classification Analysis")
        
        # Prepare data for classification
        leak_features = ['PageValues', 'BounceRates', 'ExitRates']
        X = df.drop(columns=['Revenue'] + leak_features)
        y = df['Revenue']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model selection
        model = st.selectbox("Select Model", ["Decision Tree", "Naive Bayes", "K-Nearest Neighbors"])
        
        if model == "Decision Tree":
            clf = DecisionTreeClassifier(random_state=42)
        elif model == "Naive Bayes":
            clf = GaussianNB()
        else:
            k = st.slider("Select number of neighbors (K)", 1, 30, 7)
            clf = KNeighborsClassifier(n_neighbors=k)
        
        if st.button("Train and Evaluate Model"):
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            
            st.subheader("Classification Report")
            st.text(classification_report(y_test, y_pred, target_names=["No Purchase", "Purchase"]))

    elif page == "Clustering Analysis":
        st.header("Clustering Analysis")
        
        clustering_method = st.selectbox("Select Clustering Method", ["K-Means", "Hierarchical Clustering"])
        
        if clustering_method == "K-Means":
            n_clusters = st.slider("Select number of clusters", 2, 10, 3)
            
            if st.button("Perform K-Means Clustering"):
                X = df.drop(columns=['Revenue'])
                X_scaled = StandardScaler().fit_transform(X)
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                df['Cluster'] = kmeans.fit_predict(X_scaled)
                
                # PCA for visualization
                pca = PCA(n_components=2)
                components = pca.fit_transform(X_scaled)
                
                fig, ax = plt.subplots(figsize=(8,5))
                scatter = ax.scatter(components[:, 0], components[:, 1], c=df['Cluster'], cmap='Set2')
                plt.title('Customer Segmentation via KMeans Clustering')
                plt.xlabel('First Principal Component')
                plt.ylabel('Second Principal Component')
                plt.colorbar(scatter)
                st.pyplot(fig)
        
        else:  # Hierarchical Clustering
            if st.button("Perform Hierarchical Clustering"):
                X = df.drop(columns=['Revenue'])
                X_scaled = StandardScaler().fit_transform(X)
                
                Z = linkage(X_scaled, method='ward')
                
                fig, ax = plt.subplots(figsize=(12, 5))
                dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=45, leaf_font_size=10)
                plt.title('Hierarchical Clustering Dendrogram')
                plt.xlabel('Sample Index')
                plt.ylabel('Distance')
                st.pyplot(fig)

    elif page == "Association Rules":
        st.header("Association Rules Analysis")
        
        method = st.selectbox("Select Method", ["Apriori", "FP-Growth"])
        min_support = st.slider("Minimum Support", 0.01, 0.1, 0.05)
        min_confidence = st.slider("Minimum Confidence", 0.1, 0.9, 0.5)
        
        if st.button("Generate Rules"):
            # Prepare data for association rules
            # Convert categorical variables to binary format
            df_rules = df[['VisitorType', 'Weekend', 'Revenue']].copy()
            
            # Convert to binary format
            df_rules['VisitorType'] = df_rules['VisitorType'].astype(str)
            df_rules['Weekend'] = df_rules['Weekend'].map({0: 'Weekday', 1: 'Weekend'})
            df_rules['Revenue'] = df_rules['Revenue'].map({0: 'NoPurchase', 1: 'Purchase'})
            
            # Create dummy variables
            df_trans = pd.get_dummies(df_rules)
            
            if method == "Apriori":
                frequent_itemsets = apriori(df_trans, min_support=min_support, use_colnames=True)
            else:
                frequent_itemsets = fpgrowth(df_trans, min_support=min_support, use_colnames=True)
            
            rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)
            
            if not rules.empty:
                st.subheader("Top 5 Rules")
                st.dataframe(rules.head())
                
                # Plot confidence vs support
                fig, ax = plt.subplots(figsize=(8, 5))
                scatter = ax.scatter(rules['support'], rules['confidence'], alpha=0.5)
                plt.title('Support vs Confidence')
                plt.xlabel('Support')
                plt.ylabel('Confidence')
                st.pyplot(fig)
            else:
                st.warning("No rules found with the current parameters. Try lowering the minimum support or confidence.")

else:
    st.info("Please upload a CSV file to begin analysis.")
    st.markdown("""
    ### Expected CSV Format
    The CSV file should contain the following columns:
    - Administrative_Duration
    - Informational_Duration
    - ProductRelated_Duration
    - BounceRates
    - ExitRates
    - PageValues
    - Month
    - VisitorType
    - Weekend
    - Revenue
    """) 