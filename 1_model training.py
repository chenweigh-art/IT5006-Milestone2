"""
Chicago Crime Arrest Prediction - End-to-End Pipeline
Training Period: 2015-2024
Testing Period: 2025
Models: Logistic Regression, Random Forest, Decision Tree, XGBoost
Evaluation: Standard metrics, Spatial analysis, Temporal analysis, Robustness, Cross-validation
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score,
    roc_curve, accuracy_score, f1_score, precision_score, recall_score,
    average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed")

try:
    from imblearn.over_sampling import SMOTE

    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("Chicago Arrest Prediction Pipeline")
print("=" * 80)

# ============================================================================
# Stage 1: Data Loading and Preprocessing
# ============================================================================
print("\n" + "=" * 80)
print("Stage 1: Data Loading and Preprocessing")
print("=" * 80)

filepath = r'C:\Users\Lenovo\Desktop\5006\proj\Crimes_2001_to_Present.csv'

print(f"\nLoading data: {filepath}")
df = pd.read_csv(filepath, low_memory=False)
print(f"Original data: {df.shape[0]:,} rows x {df.shape[1]} columns")

# Verify required columns
required_cols = ['Date', 'Arrest', 'Primary Type']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"Error: Missing required columns: {missing_cols}")
    exit()

print("\nPreprocessing steps:")

# Remove rows with missing critical values
print("  1. Removing rows with missing values...")
initial_count = len(df)
df = df.dropna(subset=['Date', 'Arrest', 'Primary Type'])
print(f"     Removed {initial_count - len(df):,} rows")

# Parse dates
print("  2. Parsing date column...")
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
df = df.dropna(subset=['Date'])

# Extract year
df['Year'] = df['Date'].dt.year
print(f"     Date range: {df['Date'].min()} to {df['Date'].max()}")

# Convert target to boolean
print("  3. Converting target variable...")
df['Arrest'] = df['Arrest'].astype(bool)

# Remove future dates if any
future_count = df[df['Date'] > pd.Timestamp.now()].shape[0]
if future_count > 0:
    print(f"  4. Removing {future_count:,} future records")
    df = df[df['Date'] <= pd.Timestamp.now()]

# Create train/test split
print("\n  5. Creating train/test split...")
train_df = df[(df['Year'] >= 2015) & (df['Year'] <= 2024)].copy()
test_df = df[df['Year'] == 2025].copy()

print(f"     Training set (2015-2024): {len(train_df):,} records")
print(f"     Test set (2025): {len(test_df):,} records")

# Fallback if 2025 data unavailable
if len(test_df) == 0:
    print("\n     Note: Using Oct-Dec 2024 as test set")
    train_df = df[(df['Year'] >= 2015) &
                  ((df['Year'] < 2024) |
                   ((df['Year'] == 2024) & (df['Date'].dt.month < 10)))].copy()
    test_df = df[(df['Year'] == 2024) & (df['Date'].dt.month >= 10)].copy()

    print(f"     Training set (2015-Sep 2024): {len(train_df):,} records")
    print(f"     Test set (Oct-Dec 2024): {len(test_df):,} records")

# Display arrest rate
arrest_rate = train_df['Arrest'].sum() / len(train_df) * 100
print(f"\n  6. Training set arrest rate: {arrest_rate:.2f}%")

# ============================================================================
# Stage 2: Feature Engineering
# ============================================================================
print("\n" + "=" * 80)
print("Stage 2: Feature Engineering")
print("=" * 80)


def create_features(df):
    """Generate temporal and categorical features"""
    # Temporal features
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Hour'] = df['Date'].dt.hour
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['Quarter'] = df['Date'].dt.quarter

    # Time period categorization
    def categorize_time(hour):
        if pd.isna(hour):
            return 'Unknown'
        if 0 <= hour < 6:
            return 'Night'
        elif 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 18:
            return 'Afternoon'
        else:
            return 'Evening'

    df['Time_Period'] = df['Hour'].apply(categorize_time)

    # Season categorization
    def categorize_season(month):
        if pd.isna(month):
            return 'Unknown'
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    df['Season'] = df['Month'].apply(categorize_season)

    # Cyclical encoding
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    # Location indicator
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        df['Has_Location'] = (~df['Latitude'].isna()).astype(int)

    # Domestic violence indicator
    if 'Domestic' in df.columns:
        df['Is_Domestic'] = df['Domestic'].fillna(False).astype(int)

    return df


print("\nApplying feature engineering...")
train_df = create_features(train_df)
test_df = create_features(test_df)

print("New features created:")
feature_list = ['Month', 'Hour', 'DayOfWeek', 'Is_Weekend', 'Quarter',
                'Time_Period', 'Season', 'Hour_Sin', 'Hour_Cos',
                'Month_Sin', 'Month_Cos']
for feat in feature_list:
    if feat in train_df.columns:
        print(f"  - {feat}")

# ============================================================================
# Stage 3: Prepare Model Inputs
# ============================================================================
print("\n" + "=" * 80)
print("Stage 3: Prepare Model Inputs")
print("=" * 80)

print("\nSelecting features...")

# Numerical features
numerical_features = ['Month', 'Day', 'Hour', 'DayOfWeek', 'Is_Weekend',
                      'Quarter', 'Hour_Sin', 'Hour_Cos', 'Month_Sin', 'Month_Cos']

# Add available geographic features
if 'District' in train_df.columns:
    numerical_features.append('District')
if 'Ward' in train_df.columns:
    numerical_features.append('Ward')
if 'Has_Location' in train_df.columns:
    numerical_features.append('Has_Location')
if 'Is_Domestic' in train_df.columns:
    numerical_features.append('Is_Domestic')

# Categorical features
categorical_features = []
if 'Primary Type' in train_df.columns:
    categorical_features.append('Primary Type')
if 'Time_Period' in train_df.columns:
    categorical_features.append('Time_Period')
if 'Season' in train_df.columns:
    categorical_features.append('Season')
if 'Location Description' in train_df.columns:
    categorical_features.append('Location Description')

print(f"\nNumerical features ({len(numerical_features)}): {numerical_features[:5]}...")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

# One-hot encoding
print("\nApplying one-hot encoding...")

train_df['dataset'] = 'train'
test_df['dataset'] = 'test'
combined = pd.concat([train_df, test_df], ignore_index=True)

for cat_feat in categorical_features:
    if cat_feat in combined.columns:
        # Keep top 20 categories
        top_cats = combined[cat_feat].value_counts().head(20).index
        combined[cat_feat] = combined[cat_feat].apply(
            lambda x: x if x in top_cats else 'Other'
        )

        # Create dummy variables
        dummies = pd.get_dummies(combined[cat_feat], prefix=cat_feat, drop_first=True)
        combined = pd.concat([combined, dummies], axis=1)

        print(f"  {cat_feat}: {len(dummies.columns)} categories")

# Separate back to train and test
train_encoded = combined[combined['dataset'] == 'train'].copy()
test_encoded = combined[combined['dataset'] == 'test'].copy()

# Collect feature columns
feature_columns = numerical_features.copy()
for cat_feat in categorical_features:
    cat_cols = [col for col in combined.columns if col.startswith(f'{cat_feat}_')]
    feature_columns.extend(cat_cols)

# Prepare matrices
X_train = train_encoded[feature_columns].fillna(0)
y_train = train_encoded['Arrest'].values

X_test = test_encoded[feature_columns].fillna(0)
y_test = test_encoded['Arrest'].values

print(f"\nTraining matrix: {X_train.shape}")
print(f"Test matrix: {X_test.shape}")

# Display class distribution
unique, counts = np.unique(y_train, return_counts=True)
print(f"\nTarget distribution:")
for val, count in zip(unique, counts):
    label = 'Arrest' if val else 'No Arrest'
    print(f"  {label}: {count:,} ({count / len(y_train) * 100:.2f}%)")

# Standardize features
print("\nStandardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# Stage 4: Handle Class Imbalance
# ============================================================================
print("\n" + "=" * 80)
print("Stage 4: Handle Class Imbalance")
print("=" * 80)

# Compute class weights
class_weights = compute_class_weight('balanced',
                                     classes=np.unique(y_train),
                                     y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

print(f"\nClass weights: {class_weight_dict}")

# Use class weights
X_train_final = X_train_scaled
y_train_final = y_train

# ============================================================================
# Stage 5: Model Training
# ============================================================================
print("\n" + "=" * 80)
print("Stage 5: Model Training")
print("=" * 80)

models = {}

# Model 1: Logistic Regression
print("\n[1/4] Training Logistic Regression...")
lr = LogisticRegression(
    max_iter=1000,
    class_weight=class_weight_dict,
    random_state=42,
    n_jobs=-1
)
lr.fit(X_train_final, y_train_final)
models['Logistic Regression'] = lr
print("  Training complete")

# Model 2: Random Forest
print("\n[2/4] Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    class_weight=class_weight_dict,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_final, y_train_final)
models['Random Forest'] = rf
print("  Training complete")

# Model 3: Decision Tree
print("\n[3/4] Training Decision Tree...")
dt = DecisionTreeClassifier(
    max_depth=15,
    min_samples_split=20,
    class_weight=class_weight_dict,
    random_state=42
)
dt.fit(X_train_final, y_train_final)
models['Decision Tree'] = dt
print("  Training complete")

# Model 4: XGBoost
print("\n[4/4] Training XGBoost...")
if HAS_XGBOOST:
    scale_pos_weight = (y_train_final == 0).sum() / (y_train_final == 1).sum()

    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train_final, y_train_final)
    models['XGBoost'] = xgb_model
    print("  Training complete")
else:
    print("  XGBoost not available, skipping")

print(f"\nTrained {len(models)} models")

# ============================================================================
# Stage 6: Comprehensive Evaluation
# ============================================================================
print("\n" + "=" * 80)
print("Stage 6: Comprehensive Evaluation")
print("=" * 80)

output_dir = r'C:\Users\Lenovo\Desktop\5006\proj'
all_results = {}

for model_name, model in models.items():
    print(f"\n{'=' * 80}")
    print(f"Evaluating: {model_name}")
    print(f"{'=' * 80}")

    all_results[model_name] = {}

    # Predictions
    y_pred = model.predict(X_test_scaled)

    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred_proba = None

    # ========================================================================
    # 1. Standard Classification Metrics
    # ========================================================================
    print("\n" + "-" * 80)
    print("1. STANDARD CLASSIFICATION METRICS")
    print("-" * 80)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    all_results[model_name]['standard_metrics'] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    # AUC-ROC
    if y_pred_proba is not None:
        try:
            auc_roc = roc_auc_score(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
            print(f"AUC-ROC:   {auc_roc:.4f}")
            print(f"PR-AUC:    {avg_precision:.4f}")
            all_results[model_name]['standard_metrics']['auc_roc'] = auc_roc
            all_results[model_name]['standard_metrics']['pr_auc'] = avg_precision
            all_results[model_name]['y_pred_proba'] = y_pred_proba
        except:
            pass

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"               No    Yes")
    print(f"Actual No  {cm[0, 0]:6d} {cm[0, 1]:6d}")
    print(f"       Yes {cm[1, 0]:6d} {cm[1, 1]:6d}")

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    print(f"\nSpecificity: {specificity:.4f}")
    print(f"NPV:         {npv:.4f}")

    all_results[model_name]['standard_metrics']['specificity'] = specificity
    all_results[model_name]['standard_metrics']['npv'] = npv

    # ========================================================================
    # 2. Spatial Accuracy Evaluation
    # ========================================================================
    print("\n" + "-" * 80)
    print("2. SPATIAL ACCURACY EVALUATION")
    print("-" * 80)

    if 'District' in test_encoded.columns:
        print("\nPerformance by District (Top 10):")
        print(f"{'District':<12} {'Count':>8} {'Accuracy':>10} {'F1-Score':>10}")
        print("-" * 48)

        districts = test_encoded['District'].value_counts().head(10).index
        district_f1_scores = []

        for district in districts:
            district_mask = test_encoded['District'] == district
            if district_mask.sum() > 10:
                y_true_dist = y_test[district_mask]
                y_pred_dist = y_pred[district_mask]

                dist_acc = accuracy_score(y_true_dist, y_pred_dist)
                dist_f1 = f1_score(y_true_dist, y_pred_dist, average='binary', zero_division=0)
                district_f1_scores.append(dist_f1)

                print(f"District {int(district):<4} {district_mask.sum():>8} {dist_acc:>10.4f} {dist_f1:>10.4f}")

        if len(district_f1_scores) > 0:
            spatial_variance = np.var(district_f1_scores)
            print(f"\nSpatial Variance (F1): {spatial_variance:.6f}")
            all_results[model_name]['spatial_variance'] = spatial_variance

    # ========================================================================
    # 3. Temporal Accuracy Measures
    # ========================================================================
    print("\n" + "-" * 80)
    print("3. TEMPORAL ACCURACY MEASURES")
    print("-" * 80)

    if 'Month' in test_encoded.columns:
        print("\nPerformance by Month:")
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        monthly_f1 = []
        for month in range(1, 13):
            month_mask = test_encoded['Month'] == month
            if month_mask.sum() > 10:
                month_f1 = f1_score(y_test[month_mask], y_pred[month_mask],
                                    average='binary', zero_division=0)
                monthly_f1.append(month_f1)
                print(f"  {month_names[month - 1]}: F1 = {month_f1:.4f}")

        if len(monthly_f1) > 0:
            temporal_variance = np.var(monthly_f1)
            print(f"\nTemporal Variance (F1): {temporal_variance:.6f}")
            all_results[model_name]['temporal_variance'] = temporal_variance

    # ========================================================================
    # 4. Model Robustness Analysis
    # ========================================================================
    print("\n" + "-" * 80)
    print("4. MODEL ROBUSTNESS ANALYSIS")
    print("-" * 80)

    if 'Primary Type' in test_encoded.columns:
        print("\nPerformance by Crime Type (Top 5):")
        print(f"{'Crime Type':<25} {'Count':>8} {'F1-Score':>10}")
        print("-" * 50)

        top_crimes = test_encoded['Primary Type'].value_counts().head(5).index
        crime_f1_scores = []

        for crime in top_crimes:
            crime_mask = test_encoded['Primary Type'] == crime
            if crime_mask.sum() > 10:
                crime_f1 = f1_score(y_test[crime_mask], y_pred[crime_mask],
                                    average='binary', zero_division=0)
                crime_f1_scores.append(crime_f1)
                print(f"{str(crime)[:24]:<25} {crime_mask.sum():>8} {crime_f1:>10.4f}")

        if len(crime_f1_scores) > 0:
            crime_variance = np.var(crime_f1_scores)
            print(f"\nCrime Type Variance (F1): {crime_variance:.6f}")

    print(f"\nClass Imbalance Handling:")
    print(f"  Minority Class Recall:    {recall:.4f}")
    print(f"  Majority Class Specificity: {specificity:.4f}")

    # ========================================================================
    # 5. Cross-Validation Results
    # ========================================================================
    print("\n" + "-" * 80)
    print("5. CROSS-VALIDATION RESULTS")
    print("-" * 80)

    print("\nPerforming 3-Fold Stratified CV...")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    cv_f1 = cross_val_score(model, X_train_final, y_train_final,
                            cv=cv, scoring='f1', n_jobs=-1)

    print(f"\n3-Fold CV F1-Score: {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")
    print(f"  Min: {cv_f1.min():.4f}")
    print(f"  Max: {cv_f1.max():.4f}")

    cv_stability = cv_f1.std()
    if cv_stability < 0.02:
        stability = "Excellent"
    elif cv_stability < 0.05:
        stability = "Good"
    else:
        stability = "Fair"

    print(f"\nStability: {stability}")

    all_results[model_name]['cv_f1_mean'] = cv_f1.mean()
    all_results[model_name]['cv_f1_std'] = cv_f1.std()
    all_results[model_name]['stability'] = stability

# ============================================================================
# Generate Summary and Visualizations
# ============================================================================
print("\n" + "=" * 80)
print("Generating Summary and Visualizations")
print("=" * 80)

# Summary table
summary_data = []
for model_name in models.keys():
    row = {
        'Model': model_name,
        'Accuracy': all_results[model_name]['standard_metrics']['accuracy'],
        'Precision': all_results[model_name]['standard_metrics']['precision'],
        'Recall': all_results[model_name]['standard_metrics']['recall'],
        'F1-Score': all_results[model_name]['standard_metrics']['f1_score'],
        'AUC-ROC': all_results[model_name]['standard_metrics'].get('auc_roc', np.nan),
        'CV_F1': all_results[model_name]['cv_f1_mean'],
        'CV_Std': all_results[model_name]['cv_f1_std'],
        'Stability': all_results[model_name]['stability']
    }
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data).sort_values('F1-Score', ascending=False)

print("\n" + "=" * 80)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 80)
print("\n" + summary_df.to_string(index=False))

# Save summary
summary_df.to_csv(f'{output_dir}\\model_performance_summary.csv', index=False)
print(f"\nSaved: model_performance_summary.csv")

# Best model
best_model_name = summary_df.iloc[0]['Model']
print(f"\nBest Model: {best_model_name}")
print(f"   F1-Score: {summary_df.iloc[0]['F1-Score']:.4f}")

# Visualizations
print("\nGenerating visualizations...")

# ROC curves
plt.figure(figsize=(10, 8))
for model_name in models.keys():
    if 'y_pred_proba' in all_results[model_name]:
        y_prob = all_results[model_name]['y_pred_proba']
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = all_results[model_name]['standard_metrics'].get('auc_roc', 0)
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC={auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}\\roc_curves.png', dpi=300)
print("Saved: roc_curves.png")
plt.close()

# Metrics comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    values = summary_df[metric].values
    bars = ax.bar(range(len(summary_df)), values, color='steelblue', edgecolor='black')
    ax.set_xticks(range(len(summary_df)))
    ax.set_xticklabels(summary_df['Model'].values, rotation=45, ha='right')
    ax.set_ylabel(metric, fontsize=11)
    ax.set_ylim([0, 1])
    ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f'{output_dir}\\metrics_comparison.png', dpi=300)
print("Saved: metrics_comparison.png")
plt.close()

# Feature importance (Random Forest)
if 'Random Forest' in models:
    print("\nGenerating feature importance...")
    importances = models['Random Forest'].feature_importances_
    feature_imp_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    top_20 = feature_imp_df.head(20)
    plt.barh(range(len(top_20)), top_20['Importance'].values, color='darkgreen')
    plt.yticks(range(len(top_20)), top_20['Feature'].values)
    plt.xlabel('Importance', fontsize=12)
    plt.title('Top 20 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}\\feature_importance.png', dpi=300)
    print("Saved: feature_importance.png")
    plt.close()

    feature_imp_df.to_csv(f'{output_dir}\\feature_importance.csv', index=False)
    print("Saved: feature_importance.csv")

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "=" * 80)
print("PIPELINE COMPLETE!")
print("=" * 80)

print("\nGenerated Files:")
print("  1. model_performance_summary.csv - All models comparison")
print("  2. roc_curves.png - ROC curves comparison")
print("  3. metrics_comparison.png - 4-panel metrics comparison")
print("  4. feature_importance.png - Feature importance chart")
print("  5. feature_importance.csv - Feature importance table")

print("\nEvaluation Coverage:")
print("  - Standard Classification Metrics (Precision, Recall, F1, AUC-ROC)")
print("  - Spatial Accuracy Evaluation (by District)")
print("  - Temporal Accuracy Measures (by Month)")
print("  - Model Robustness Analysis (by Crime Type)")
print("  - Cross-Validation Results (3-Fold Stratified)")

print("\n" + "=" * 80)
print(f"Best Model: {best_model_name}")
print(f"F1-Score: {summary_df.iloc[0]['F1-Score']:.4f}")
print(f"Accuracy: {summary_df.iloc[0]['Accuracy']:.4f}")
print(f"AUC-ROC: {summary_df.iloc[0]['AUC-ROC']:.4f}")
print("=" * 80)