import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, r2_score, classification_report, mean_squared_error, 
                            precision_score, recall_score, f1_score, confusion_matrix)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SETUP - Page Configuration and Title
# =============================================================================

st.set_page_config(
    page_title="ML Analytics Dashboard", 
    page_icon="📊", 
    layout="wide"
)

st.title("🤖 Machine Learning Analytics Dashboard")
st.markdown("Upload your CSV file and build predictive models with ease!")

# =============================================================================
# DATA UPLOAD SECTION
# =============================================================================

st.header("📁 Data Upload")
uploaded_file = st.file_uploader(
    "Choose a CSV file", 
    type="csv", 
    help="Upload a CSV file to get started with analysis and modeling"
)

if uploaded_file is not None:
    # Load the dataset
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns!")
        
        # Store dataset in session state
        st.session_state.df = df
        
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        st.stop()

    # =============================================================================
    # EXPLORATORY DATA ANALYSIS (EDA) SECTION
    # =============================================================================
    
    st.header("📊 Exploratory Data Analysis")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Overview")
        st.write("**Shape:**", df.shape)
        st.write("**Data Types:**")
        st.write(df.dtypes)
    
    with col2:
        st.subheader("Missing Values")
        missing_values = df.isnull().sum()
        if missing_values.sum() == 0:
            st.write("✅ No missing values found!")
        else:
            st.write(missing_values[missing_values > 0])
    
    # Dataset Head
    st.subheader("Dataset Head")
    st.dataframe(df.head())
    
    # Summary Statistics
    st.subheader("Summary Statistics")
    st.dataframe(df.describe())
    
    # Visualization Section
    st.subheader("Data Visualization")
    
    # Get numeric columns for plotting
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_columns:
        selected_column = st.selectbox(
            "Select a column to visualize its distribution:",
            numeric_columns,
            help="Choose a numeric column to display its histogram"
        )
        
        if selected_column:
            col1, col2 = st.columns(2)
            
            with col1:
                # Matplotlib histogram
                st.write("**Matplotlib Histogram**")
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(df[selected_column].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_xlabel(selected_column)
                ax.set_ylabel('Frequency')
                ax.set_title(f'Distribution of {selected_column}')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                # Plotly histogram
                st.write("**Interactive Plotly Histogram**")
                fig = px.histogram(
                    df, 
                    x=selected_column, 
                    title=f'Distribution of {selected_column}',
                    nbins=30,
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No numeric columns found for visualization.")
    
    # =============================================================================
    # MACHINE LEARNING SECTION - SIDEBAR CONTROLS
    # =============================================================================
    
    st.sidebar.header("🤖 Machine Learning Controls")
    st.sidebar.markdown("Configure your predictive model below:")
    
    # Target Variable Selection
    st.sidebar.subheader("🎯 Target Variable")
    all_columns = df.columns.tolist()
    target_variable = st.sidebar.selectbox(
        "Select Target Variable (y):",
        all_columns,
        help="Choose the column you want to predict"
    )
    
    # Determine if target is suitable for classification or regression
    if target_variable:
        unique_values = df[target_variable].nunique()
        is_numeric = df[target_variable].dtype in ['int64', 'float64']
        
        st.sidebar.write(f"**Target Info:**")
        st.sidebar.write(f"- Unique values: {unique_values}")
        st.sidebar.write(f"- Data type: {df[target_variable].dtype}")
        
        # Prediction Type Selection
        st.sidebar.subheader("📈 Prediction Type")
        
        # Auto-suggest based on target characteristics
        if unique_values <= 10 and is_numeric:
            suggested_type = "Classification"
            suggestion_reason = f"(Suggested: {unique_values} unique values)"
        elif unique_values > 10 and is_numeric:
            suggested_type = "Regression"
            suggestion_reason = f"(Suggested: {unique_values} continuous values)"
        else:
            suggested_type = "Classification"
            suggestion_reason = "(Suggested: Non-numeric target)"
        
        prediction_type = st.sidebar.radio(
            f"Choose prediction type {suggestion_reason}:",
            ["Classification", "Regression"]
        )
        
        # Feature Selection
        st.sidebar.subheader("⚙️ Feature Selection")
        available_features = [col for col in all_columns if col != target_variable]
        
        selected_features = st.sidebar.multiselect(
            "Select Feature Columns (X):",
            available_features,
            default=available_features[:min(5, len(available_features))],  # Default to first 5 features
            help="Choose the columns to use as input features for the model"
        )
        
        # Advanced Options for Classification
        if prediction_type == "Classification":
            st.sidebar.subheader("🔧 Advanced Options")
            
            # Class balancing options
            class_balance_method = st.sidebar.selectbox(
                "Class Balancing Method:",
                ["None", "SMOTE", "SMOTE + ENN"],
                help="Choose a method to handle class imbalance"
            )
            
            # Stratification option
            use_stratification = st.sidebar.checkbox(
                "Use Stratified Sampling",
                value=True,
                help="Maintain class distribution in train-test split"
            )
            
            # Display class distribution
            if target_variable:
                st.sidebar.write("**Current Class Distribution:**")
                class_dist = df[target_variable].value_counts()
                for class_name, count in class_dist.items():
                    percentage = (count / len(df)) * 100
                    st.sidebar.write(f"- {class_name}: {count} ({percentage:.1f}%)")
        
        # Test size selection
        test_size = st.sidebar.slider(
            "Test Set Size:",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Proportion of dataset to use for testing"
        )
        
        # Model Training Button
        if selected_features:
            train_model = st.sidebar.button("🚀 Train Model & Predict", type="primary")
            
            # =============================================================================
            # MODEL EXECUTION AND RESULTS
            # =============================================================================
            
            if train_model:
                st.header("🎯 Model Training & Results")
                
                try:
                    # Data preprocessing
                    with st.spinner("⚙️ Preprocessing data..."):
                        # Prepare feature matrix X and target vector y
                        X = df[selected_features].copy()
                        y = df[target_variable].copy()
                        
                        # Handle missing values in features
                        # For numeric columns, fill with mean
                        numeric_cols = X.select_dtypes(include=[np.number]).columns
                        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
                        
                        # For categorical columns, fill with mode
                        categorical_cols = X.select_dtypes(include=['object']).columns
                        for col in categorical_cols:
                            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')
                        
                        # Handle missing values in target
                        if y.isnull().sum() > 0:
                            if prediction_type == "Regression":
                                y = y.fillna(y.mean())
                            else:
                                y = y.fillna(y.mode()[0] if not y.mode().empty else y.unique()[0])
                        
                        # One-hot encoding for categorical features
                        X_encoded = pd.get_dummies(X, drop_first=True)
                        
                        # For classification, encode target if it's categorical
                        if prediction_type == "Classification" and y.dtype == 'object':
                            le = LabelEncoder()
                            y_encoded = le.fit_transform(y)
                            target_classes = le.classes_
                        else:
                            y_encoded = y
                            target_classes = None
                    
                    # Train-test split
                    with st.spinner("🔀 Splitting data..."):
                        if prediction_type == "Classification" and use_stratification:
                            # Use stratified split to maintain class distribution
                            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
                            train_idx, test_idx = next(sss.split(X_encoded, y_encoded))
                            X_train, X_test = X_encoded.iloc[train_idx], X_encoded.iloc[test_idx]
                            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
                        else:
                            # Regular train-test split
                            X_train, X_test, y_train, y_test = train_test_split(
                                X_encoded, y_encoded, test_size=test_size, random_state=42
                            )
                    
                    # Apply SMOTE if selected (only for classification)
                    if prediction_type == "Classification" and class_balance_method != "None":
                        with st.spinner(f"⚖️ Applying {class_balance_method} for class balancing..."):
                            original_train_dist = pd.Series(y_train).value_counts().sort_index()
                            
                            if class_balance_method == "SMOTE":
                                smote = SMOTE(random_state=42)
                                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                            elif class_balance_method == "SMOTE + ENN":
                                smote_enn = SMOTEENN(random_state=42)
                                X_train_balanced, y_train_balanced = smote_enn.fit_resample(X_train, y_train)
                            
                            balanced_train_dist = pd.Series(y_train_balanced).value_counts().sort_index()
                            
                            # Show balancing results
                            st.info(f"✅ Applied {class_balance_method}: {len(X_train)} → {len(X_train_balanced)} samples")
                            
                            X_train, y_train = X_train_balanced, y_train_balanced
                    
                    # Display data split information
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.info(f"**Training Set:** {len(X_train)} samples")
                    with col2:
                        st.info(f"**Test Set:** {len(X_test)} samples")
                    with col3:
                        if prediction_type == "Classification":
                            st.info(f"**Classes:** {len(np.unique(y_encoded))}")
                        else:
                            st.info(f"**Target Range:** {y_encoded.min():.2f} - {y_encoded.max():.2f}")
                    
                    # Show class distribution after balancing
                    if prediction_type == "Classification" and class_balance_method != "None":
                        st.subheader("📊 Class Distribution Comparison")
                        
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                        
                        # Original distribution
                        original_train_dist.plot(kind='bar', ax=ax1, color='lightcoral', alpha=0.7)
                        ax1.set_title('Original Training Set Distribution')
                        ax1.set_xlabel('Class')
                        ax1.set_ylabel('Count')
                        ax1.tick_params(axis='x', rotation=45)
                        
                        # Balanced distribution
                        balanced_train_dist.plot(kind='bar', ax=ax2, color='lightgreen', alpha=0.7)
                        ax2.set_title(f'After {class_balance_method} Distribution')
                        ax2.set_xlabel('Class')
                        ax2.set_ylabel('Count')
                        ax2.tick_params(axis='x', rotation=45)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Model selection and training
                    with st.spinner("🤖 Training model..."):
                        if prediction_type == "Classification":
                            if len(np.unique(y_encoded)) == 2:
                                model = LogisticRegression(random_state=42, max_iter=1000)
                                model_name = "Logistic Regression"
                            else:
                                model = RandomForestClassifier(random_state=42, n_estimators=100)
                                model_name = "Random Forest Classifier"
                        else:
                            model = RandomForestRegressor(random_state=42, n_estimators=100)
                            model_name = "Random Forest Regressor"
                        
                        # Train the model
                        model.fit(X_train, y_train)
                        
                        # Make predictions
                        y_pred = model.predict(X_test)
                    
                    # Display results
                    st.success("✅ Model trained successfully!")
                    
                    # Model information
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.info(f"**Model Type:** {model_name}")
                    with col2:
                        st.info(f"**Prediction Type:** {prediction_type}")
                    with col3:
                        st.info(f"**Features Used:** {len(selected_features)}")
                    
                    # Evaluation metrics
                    st.subheader("📊 Model Performance")
                    
                    if prediction_type == "Classification":
                        # Calculate all classification metrics
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        # Handle multi-class vs binary classification for averaging
                        avg_method = 'binary' if len(np.unique(y_test)) == 2 else 'weighted'
                        
                        precision = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
                        recall = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
                        f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
                        
                        # Display main metrics in columns
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Accuracy", f"{accuracy:.4f}", f"{accuracy*100:.2f}%")
                        with col2:
                            st.metric("Precision", f"{precision:.4f}", f"{precision*100:.2f}%")
                        with col3:
                            st.metric("Recall", f"{recall:.4f}", f"{recall*100:.2f}%")
                        with col4:
                            st.metric("F1-Score", f"{f1:.4f}", f"{f1*100:.2f}%")
                        
                        # Detailed classification report with support
                        st.subheader("📋 Detailed Classification Report")
                        
                        if target_classes is not None:
                            y_test_labels = le.inverse_transform(y_test)
                            y_pred_labels = le.inverse_transform(y_pred)
                            report_dict = classification_report(y_test_labels, y_pred_labels, output_dict=True)
                        else:
                            report_dict = classification_report(y_test, y_pred, output_dict=True)
                        
                        # Convert to DataFrame and format
                        report_df = pd.DataFrame(report_dict).transpose()
                        
                        # Round values for better display
                        numeric_columns = ['precision', 'recall', 'f1-score']
                        for col in numeric_columns:
                            if col in report_df.columns:
                                report_df[col] = report_df[col].round(4)
                        
                        # Format support as integer
                        if 'support' in report_df.columns:
                            report_df['support'] = report_df['support'].astype('Int64')
                        
                        st.dataframe(report_df)
                        
                        # Confusion Matrix
                        st.subheader("🔍 Confusion Matrix")
                        
                        cm = confusion_matrix(y_test, y_pred)
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        
                        # Use class labels if available
                        if target_classes is not None:
                            labels = target_classes
                        else:
                            labels = np.unique(y_test)
                        
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                   xticklabels=labels, yticklabels=labels, ax=ax)
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        ax.set_title('Confusion Matrix')
                        
                        st.pyplot(fig)
                        
                        # Per-class metrics breakdown
                        if len(np.unique(y_test)) > 2:
                            st.subheader("📊 Per-Class Metrics")
                            
                            # Calculate per-class metrics
                            precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
                            recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
                            f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
                            
                            # Create DataFrame for per-class metrics
                            class_labels = target_classes if target_classes is not None else np.unique(y_test)
                            
                            per_class_df = pd.DataFrame({
                                'Class': class_labels,
                                'Precision': precision_per_class,
                                'Recall': recall_per_class,
                                'F1-Score': f1_per_class
                            })
                            
                            # Bar chart for per-class metrics
                            fig, ax = plt.subplots(figsize=(10, 6))
                            x = np.arange(len(class_labels))
                            width = 0.25
                            
                            ax.bar(x - width, precision_per_class, width, label='Precision', alpha=0.8)
                            ax.bar(x, recall_per_class, width, label='Recall', alpha=0.8)
                            ax.bar(x + width, f1_per_class, width, label='F1-Score', alpha=0.8)
                            
                            ax.set_xlabel('Classes')
                            ax.set_ylabel('Score')
                            ax.set_title('Per-Class Performance Metrics')
                            ax.set_xticks(x)
                            ax.set_xticklabels(class_labels, rotation=45)
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            st.dataframe(per_class_df.round(4))
                        
                    else:  # Regression
                        r2 = r2_score(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        mae = np.mean(np.abs(y_test - y_pred))
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("R² Score", f"{r2:.4f}", f"{r2*100:.2f}%")
                        with col2:
                            st.metric("RMSE", f"{rmse:.4f}")
                        with col3:
                            st.metric("MAE", f"{mae:.4f}")
                    
                    # Feature Importance Plot
                    st.subheader("📈 Feature Importance")
                    
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = pd.DataFrame({
                            'feature': X_encoded.columns,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        
                        # Create feature importance plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        top_features = feature_importance.head(10)  # Show top 10 features
                        
                        bars = ax.barh(top_features['feature'], top_features['importance'])
                        ax.set_xlabel('Importance')
                        ax.set_title('Top 10 Feature Importance')
                        ax.grid(True, alpha=0.3)
                        
                        # Color bars with gradient
                        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
                        for bar, color in zip(bars, colors):
                            bar.set_color(color)
                        
                        plt.gca().invert_yaxis()
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Show feature importance table
                        st.write("**Feature Importance Values:**")
                        st.dataframe(feature_importance.head(10))
                        
                    elif hasattr(model, 'coef_'):
                        # For linear models, show coefficients
                        if prediction_type == "Classification" and len(model.coef_.shape) > 1:
                            coef_values = np.abs(model.coef_[0])  # Take first class for binary classification
                        else:
                            coef_values = np.abs(model.coef_)
                        
                        feature_importance = pd.DataFrame({
                            'feature': X_encoded.columns,
                            'importance': coef_values
                        }).sort_values('importance', ascending=False)
                        
                        # Create coefficient plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        top_features = feature_importance.head(10)
                        
                        bars = ax.barh(top_features['feature'], top_features['importance'])
                        ax.set_xlabel('Absolute Coefficient Value')
                        ax.set_title('Top 10 Feature Coefficients (Absolute Values)')
                        ax.grid(True, alpha=0.3)
                        
                        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
                        for bar, color in zip(bars, colors):
                            bar.set_color(color)
                        
                        plt.gca().invert_yaxis()
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        st.write("**Feature Coefficient Values:**")
                        st.dataframe(feature_importance.head(10))
                    
                    # Prediction vs Actual (for regression)
                    if prediction_type == "Regression":
                        st.subheader("🎯 Predictions vs Actual Values")
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.scatter(y_test, y_pred, alpha=0.6)
                        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                        ax.set_xlabel('Actual Values')
                        ax.set_ylabel('Predicted Values')
                        ax.set_title('Predictions vs Actual Values')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"❌ Error during model training: {str(e)}")
                    st.write("**Troubleshooting tips:**")
                    st.write("- Ensure your target variable has valid values")
                    st.write("- Check that selected features contain relevant data")
                    st.write("- Try selecting different features or target variable")
        
        else:
            st.sidebar.warning("⚠️ Please select at least one feature column.")

else:
    # Instructions when no file is uploaded
    st.info("👆 Please upload a CSV file to get started!")
    
    st.markdown("""
    ### How to use this app:
    
    1. **Upload Data**: Click 'Browse files' above and select your CSV file
    2. **Explore Data**: View dataset statistics and visualizations
    3. **Configure Model**: Use the sidebar to:
       - Select your target variable (what you want to predict)
       - Choose prediction type (Classification or Regression)  
       - Select feature columns (input variables)
    4. **Train Model**: Click 'Train Model & Predict' to build and evaluate your model
    5. **View Results**: Examine model performance and feature importance
    
    ### Supported Features:
    - ✅ Automatic data preprocessing
    - ✅ Missing value handling
    - ✅ Categorical variable encoding
    - ✅ Model evaluation metrics
    - ✅ Feature importance analysis
    - ✅ Interactive visualizations
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | 🤖 Machine Learning Analytics Dashboard")
