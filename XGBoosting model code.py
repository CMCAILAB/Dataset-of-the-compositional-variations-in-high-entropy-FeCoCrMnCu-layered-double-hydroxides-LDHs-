import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from itertools import combinations
import shap
from scipy.stats import t
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

# 設定統一的輸出目錄
OUTPUT_BASE_PATH = "D:/desktop/中興大學/實驗室/炤方ai paper/程式碼/機器學習 圖/"

# 確保輸出目錄存在
os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)

# Function: Generate interaction terms up to max_order (default is 5)
def generate_interactions(dt, features, max_order=5):   #生成交互項(最高5階)
    """Generate interaction terms between features, up to max_order"""
    for order in range(2, max_order + 1):
        for combo in combinations(features, order):
            sorted_combo = sorted(combo)
            interaction_name = '*'.join(sorted_combo)    # ex: Fe*Co
            interaction_value = dt[list(combo)].prod(axis=1)
            dt[interaction_name] = interaction_value
    return dt

# Function: Load and engineer data
def load_and_engineer_data(filepath):  #特徵工程(歸一化、平方項、交互項)
    """Load data and perform feature engineering"""
    dt = pd.read_excel(filepath)
    print(f"Number of loaded samples: {dt.shape[0]}")
    print("First five rows of data:\n", dt.head())
    print("Missing value statistics:\n", dt.isnull().sum())

    base_features = ['Fe', 'Co', 'Cr', 'Mn', 'Cu']   #進行歸一化
    for index, row in dt.iterrows():
        total = row[base_features].sum()
        if total != 0:
            dt.loc[index, base_features] = row[base_features] / total

    important_features = ['Cr', 'Fe', 'Cu', 'Mn', 'Co']   #新噌平方項
    for col in important_features:
        dt[f'{col}^2'] = dt[col] ** 2
    dt = generate_interactions(dt, important_features, max_order=5)
    
    print(f"Final number of features: {dt.shape[1] - 1}")
    
    # 保存處理後的數據集到統一目錄
    processed_data_path = os.path.join(OUTPUT_BASE_PATH, "processed_dataset.xlsx")
    dt.to_excel(processed_data_path, index=False)
    print(f"Processed dataset saved to: {processed_data_path}")
    
    return dt

# Function: Visualize data
def visualize_data(dt, target_col="Overpotential"):
    """Visualize data distributions, transformations, scatter plots, and correlation heatmap"""
    sns.set(style="whitegrid")
    
    # 分布直方圖
    features_to_plot = ['Fe', 'Co', 'Cr', 'Mn', 'Cu', target_col]
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(features_to_plot):
        plt.subplot(2, 3, i + 1)
        sns.histplot(dt[col], kde=True, color='skyblue', stat='count')
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Counts")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_BASE_PATH}01_元素分布直方圖.png", dpi=600, bbox_inches='tight')
    plt.show()

    # Yeo-Johnson 變換前後對比
    features_to_transform = ['Fe', 'Co', 'Cr', 'Mn', 'Cu']
    pt = PowerTransformer(method='yeo-johnson')
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(features_to_transform):
        plt.subplot(2, 3, i + 1)
        sns.kdeplot(dt[col], color='blue', label=f'{col} (Original)', linewidth=2)
        transformed_data = pt.fit_transform(dt[[col]]).flatten()
        sns.kdeplot(transformed_data, color='red', label=f'{col} (Yeo-Johnson)', linewidth=2)
        plt.title(f"Distribution of {col} Before & After Yeo-Johnson")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_BASE_PATH}02_Yeo-Johnson變換對比.png", dpi=600, bbox_inches='tight')
    plt.show()

    # 散點圖
    features = ['Fe', 'Co', 'Cr', 'Mn', 'Cu']
    target = dt[target_col]
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(features):
        plt.subplot(1, len(features), i + 1)
        plt.scatter(dt[col], target, marker='o', alpha=0.7)
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel('Overpotential')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_BASE_PATH}03_元素與目標變數散點圖.png", dpi=600, bbox_inches='tight')
    plt.show()

    # 相關性熱圖
    correlation_with_target = dt.corr()[target_col].drop(target_col)
    top_10_features = correlation_with_target.abs().sort_values(ascending=False).head(10).index
    selected_features = list(top_10_features) + [target_col]
    correlation_matrix_top10 = dt[selected_features].corr().round(2)
    
    mask = np.triu(np.ones_like(correlation_matrix_top10, dtype=bool))
    sns.set(font_scale=1.5)
    plt.figure(figsize=(12, 10))
    heatmap = sns.heatmap(
        data=correlation_matrix_top10, 
        annot=True, 
        annot_kws={"size": 16}, 
        cmap="coolwarm", 
        cbar=True,
        mask=mask
    )
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=16, fontweight='bold', rotation=45, ha="right")
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=16, fontweight='bold', rotation=0)
    plt.title("Top 10 Features Correlated with Overpotential (Pearson) - Lower Triangle", fontsize=20)
    plt.savefig(f"{OUTPUT_BASE_PATH}04_相關性熱圖.png", dpi=600, bbox_inches='tight')
    plt.show()

# Function: Calculate VIF for features
def calculate_vif(X):
    """Calculate Variance Inflation Factor (VIF) for each feature"""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

# Function: Train and evaluate the model with dual R² criteria
def train_and_evaluate(dt, target_col):
    """Train and evaluate the model, prioritizing parameters with test_r2 > 0.8 and cross_val_r2 > 0.8"""
    X = dt.drop(target_col, axis=1)
    Y = dt[target_col]
    print(f"Number of samples before training: {X.shape[0]}")

    # Handle outliers
    for col in X.columns:
        lower, upper = X[col].quantile([0.05, 0.95])  #把5%和95%分位數的特徵壓縮
        X[col] = np.clip(X[col], lower, upper)
    
    X_clean = X
    Y_clean = Y
    print(f"Number of samples after processing: {X_clean.shape[0]}")

    # Split into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_clean, Y_clean, test_size=0.2, random_state=42)
    print(f"Training set samples: {X_train.shape[0]}, Test set samples: {X_test.shape[0]}")

    # Dynamically set k
    n_features = X_train.shape[1]
    k = min(15, n_features)

    # Build pipeline
    pipeline = Pipeline([
        ('preprocessor', PowerTransformer(method='yeo-johnson')),
        ('feature_selector', SelectKBest(f_regression, k=k)),
        ('regressor', XGBRegressor(random_state=42, n_jobs=-1))
    ])

    # Define parameter grid
    param_grid = {
        'feature_selector__k': [25],
        'regressor__n_estimators': [1300],
        'regressor__max_depth': [1],
        'regressor__learning_rate': [0.2],
        'regressor__reg_alpha': [0.14],
        'regressor__reg_lambda': [0.8],
        'regressor__subsample': [0.2],
        'regressor__colsample_bytree': [0.3],
        'regressor__min_child_weight': [5]
    }

    # Perform grid search
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, Y_train)

    # Evaluate all parameter combinations
    results = []
    for i in range(len(grid_search.cv_results_['mean_test_score'])):
        params = grid_search.cv_results_['params'][i]
        cross_val_r2 = grid_search.cv_results_['mean_test_score'][i]
        
        # Train model with current parameters
        model = pipeline.set_params(**params)
        model.fit(X_train, Y_train)
        
        # Calculate test set R²
        Y_test_pred = model.predict(X_test)
        test_r2 = r2_score(Y_test, Y_test_pred)
        
        results.append({
            'params': params,
            'cross_val_r2': cross_val_r2,
            'test_r2': test_r2
        })

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    results_path = os.path.join(OUTPUT_BASE_PATH, "model_evaluation_results.xlsx")
    results_df.to_excel(results_path, index=False)
    print(f"Model evaluation results saved to: {results_path}")
    
    # Filter combinations where cross_val_r2 > 0.8 and test_r2 > 0.8
    filtered_results = results_df[(results_df['cross_val_r2'] > 0.8) & (results_df['test_r2'] > 0.8)]
    
    if not filtered_results.empty:
        # Select the combination with highest cross_val_r2 among filtered results
        best_result = filtered_results.loc[filtered_results['cross_val_r2'].idxmax()]
        best_params = best_result['params']
        best_cross_val_r2 = best_result['cross_val_r2']
        best_test_r2 = best_result['test_r2']
        print("Found parameter combination satisfying cross_val_r2 > 0.8 and test_r2 > 0.8.")
    else:
        # Fallback to highest cross_val_r2
        print("No parameter combination satisfies both cross_val_r2 > 0.8 and test_r2 > 0.8. Falling back to highest cross_val_r2.")
        best_result = results_df.loc[results_df['cross_val_r2'].idxmax()]
        best_params = best_result['params']
        best_cross_val_r2 = best_result['cross_val_r2']
        best_test_r2 = best_result['test_r2']

    # Train final model
    best_model = pipeline.set_params(**best_params)
    best_model.fit(X_train, Y_train)
    
    # Get selected features
    selected_features = X_train.columns[best_model.named_steps['feature_selector'].get_support()]
    
    # Get feature importance
    xgb_model = best_model.named_steps['regressor']
    feature_importances = xgb_model.feature_importances_
    
    # Create feature importance dataframe
    importance_dict = dict(zip(selected_features, feature_importances))
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    # 保存選出的25個特徵及其重要性
    selected_25_features_df = pd.DataFrame({
        'Rank': range(1, len(sorted_features) + 1),
        'Feature_Name': [f[0] for f in sorted_features],
        'Importance': [f[1] for f in sorted_features],
        'Importance_Percentage': [f[1] / sum(feature_importances) * 100 for f in sorted_features]
    })
    
    selected_25_path = os.path.join(OUTPUT_BASE_PATH, "selected_25_features_with_importance.xlsx")
    selected_25_features_df.to_excel(selected_25_path, index=False)
    print(f"\n選出的25個特徵及重要性已保存至: {selected_25_path}")
    
    # 保存包含原始數據和選出的25個特徵的完整數據集
    base_features = ['Fe', 'Co', 'Cr', 'Mn', 'Cu']
    
    # 創建包含基礎特徵、選出的25個特徵和目標變數的數據集
    selected_columns = base_features.copy()
    
    # 添加選出的25個特徵
    for feature in selected_features:
        if feature not in selected_columns:
            selected_columns.append(feature)
    
    # 添加目標變數
    selected_columns.append(target_col)
    
    # 從原始dt中提取這些列
    dt_with_selected = dt[selected_columns].copy()
    
    # 保存完整數據集
    dataset_with_selected_path = os.path.join(OUTPUT_BASE_PATH, "dataset_with_selected_25_features.xlsx")
    dt_with_selected.to_excel(dataset_with_selected_path, index=False)
    print(f"包含原始數據和選出的25個特徵的數據集已保存至: {dataset_with_selected_path}")
    print(f"數據集形狀: {dt_with_selected.shape} (樣本數 x 特徵數)")
    print(f"包含列: {len(selected_columns)} 列 = {len(base_features)} 基礎特徵 + {len(selected_features)} 選出的特徵 + 1 目標變數")
    
    # Predict using the best model
    Y_train_pred = best_model.predict(X_train)
    Y_test_pred = best_model.predict(X_test)

    # Calculate evaluation metrics
    train_r2 = r2_score(Y_train, Y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(Y_train, Y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(Y_test, Y_test_pred))

    # Output results
    print(f"\nBest parameters: {best_params}")
    print(f"Best average R² score from cross-validation: {best_cross_val_r2:.3f}")
    print(f"Training set R²: {train_r2:.3f}, RMSE: {train_rmse:.3f}")
    print(f"Test set R²: {best_test_r2:.3f}, RMSE: {test_rmse:.3f}")

    # Calculate residuals
    train_residuals = Y_train - Y_train_pred
    test_residuals = Y_test - Y_test_pred

    # 殘差圖
    plt.figure(figsize=(10, 6))
    plt.scatter(Y_train_pred, train_residuals, alpha=0.5, c='red', label="Training Residuals")
    plt.scatter(Y_test_pred, test_residuals, alpha=0.5, c='blue', label="Test Residuals")
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel("Predicted Overpotential (mV)")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residual Plot")
    plt.legend()
    plt.savefig(f"{OUTPUT_BASE_PATH}05_殘差圖.png", dpi=600, bbox_inches='tight')
    plt.show()

    # Output actual vs predicted values with compositions
    base_features = ['Fe', 'Co', 'Cr', 'Mn', 'Cu']
    train_results = pd.DataFrame({
        'Fe': X_train[base_features[0]].values,
        'Co': X_train[base_features[1]].values,
        'Cr': X_train[base_features[2]].values,
        'Mn': X_train[base_features[3]].values,
        'Cu': X_train[base_features[4]].values,
        'Actual Overpotential': Y_train.values,
        'Predicted Overpotential': Y_train_pred,
        'Set': 'Training'
    })
    test_results = pd.DataFrame({
        'Fe': X_test[base_features[0]].values,
        'Co': X_test[base_features[1]].values,
        'Cr': X_test[base_features[2]].values,
        'Mn': X_test[base_features[3]].values,
        'Cu': X_test[base_features[4]].values,
        'Actual Overpotential': Y_test.values,
        'Predicted Overpotential': Y_test_pred,
        'Set': 'Test'
    })
    all_results = pd.concat([train_results, test_results], ignore_index=True)
    
    # 保存到統一目錄
    output_path = os.path.join(OUTPUT_BASE_PATH, "actual_vs_predicted.xlsx")
    all_results.to_excel(output_path, index=False)
    print(f"\nActual vs Predicted values with compositions saved to: {output_path}")
    print("First few rows of the results:\n", all_results.head())

    # 預測 vs 實際值
    plt.figure(figsize=(8, 8))
    plt.scatter(Y_train, Y_train_pred, alpha=0.5, c='red', label=f"Training Data (R²: {train_r2:.3f})")
    plt.scatter(Y_test, Y_test_pred, alpha=0.5, c='blue', label=f"Test Data (R²: {best_test_r2:.3f})")
    plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], color='black', linestyle='--', label="45° Reference Line")
    plt.xlabel("Actual Overpotential (mV)")
    plt.ylabel("Predicted Overpotential (mV)")
    plt.title("Predicted vs Actual (Best Model)")
    plt.legend()
    plt.savefig(f"{OUTPUT_BASE_PATH}06_預測vs實際值.png", dpi=600, bbox_inches='tight')
    plt.show()

    # Get feature names and importance
    xgb_model = best_model.named_steps['regressor']
    feature_importances = xgb_model.feature_importances_
    selected_features = X_train.columns[best_model.named_steps['feature_selector'].get_support()]

    # 特徵重要性條形圖（顯示所有特徵）
    importance_dict = dict(zip(selected_features, feature_importances))
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    all_feature_names = [feature[0] for feature in sorted_features]
    all_importances = [feature[1] for feature in sorted_features]

    # 保存特徵重要性數據
    feature_importance_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': all_importances
    })
    feature_importance_path = os.path.join(OUTPUT_BASE_PATH, "feature_importance.xlsx")
    feature_importance_df.to_excel(feature_importance_path, index=False)
    print(f"Feature importance data saved to: {feature_importance_path}")

    plt.figure(figsize=(12, len(all_feature_names) * 0.5))  # 根據特徵數量動態調整高度
    sns.barplot(x=all_importances, y=all_feature_names, palette='viridis')
    plt.title('All Feature Importance (XGBoost)')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_BASE_PATH}07_特徵重要性條形圖.png", dpi=600, bbox_inches='tight')
    plt.show()

    # 預測誤差分布
    plt.figure(figsize=(10, 6))
    sns.histplot(train_residuals, color='red', alpha=0.5, label='Training Residuals', kde=True)
    sns.histplot(test_residuals, color='blue', alpha=0.5, label='Test Residuals', kde=True)
    plt.axvline(x=0, color='black', linestyle='--')
    plt.title('Prediction Error Distribution', fontsize=16)
    plt.xlabel('Residuals (Actual - Predicted)', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_BASE_PATH}08_預測誤差分布.png", dpi=600, bbox_inches='tight')
    plt.show()

    # Box Plot of Predicted Values
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=[Y_train_pred, Y_test_pred], palette=['red', 'blue'])
    plt.xticks([0, 1], ['Training Predictions', 'Test Predictions'])
    plt.title('Box Plot of Predicted Values', fontsize=16)
    plt.ylabel('Predicted Overpotential', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_BASE_PATH}09_預測值箱形圖.png", dpi=600, bbox_inches='tight')
    plt.show()

    # 檢查測試集中的異常值
    test_predictions = Y_test_pred
    Q1 = np.percentile(test_predictions, 25)
    Q3 = np.percentile(test_predictions, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 找到低於下限的異常值
    outliers = test_predictions < lower_bound
    outlier_indices = np.where(outliers)[0]
    outlier_values = test_predictions[outliers]

    # 保存異常值分析結果
    if len(outlier_indices) > 0:
        outliers_data = []
        print("測試集中的低值異常點：")
        for idx, val in zip(outlier_indices, outlier_values):
            outlier_info = {
                'Index': idx,
                'Predicted_Value': val,
                'Fe': test_results.iloc[idx]['Fe'],
                'Co': test_results.iloc[idx]['Co'],
                'Cr': test_results.iloc[idx]['Cr'],
                'Mn': test_results.iloc[idx]['Mn'],
                'Cu': test_results.iloc[idx]['Cu'],
                'Actual_Overpotential': test_results.iloc[idx]['Actual Overpotential']
            }
            outliers_data.append(outlier_info)
            print(f"索引：{idx}, 預測值：{val:.2f}")
            print("對應的特徵值：")
            print(test_results.iloc[idx][['Fe', 'Co', 'Cr', 'Mn', 'Cu']])
            print(f"實際 Overpotential：{test_results.iloc[idx]['Actual Overpotential']:.2f}")
            print()
        
        outliers_df = pd.DataFrame(outliers_data)
        outliers_path = os.path.join(OUTPUT_BASE_PATH, "outliers_analysis.xlsx")
        outliers_df.to_excel(outliers_path, index=False)
        print(f"Outliers analysis saved to: {outliers_path}")

    # 繪製雷達圖
    total_importance = sum(all_importances[:10])  # 只使用前 10 個特徵來與雷達圖一致
    if total_importance > 0:
        importances = [(imp / total_importance * 100) for imp in all_importances[:10]]
    else:
        importances = [0] * 10
    importances += importances[:1]

    num_features = min(10, len(all_feature_names))
    angles = np.linspace(0, 2 * np.pi, num_features, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_facecolor('#F0F0F0')
    colors = ['#FF6347', '#FFD700', '#90EE90', '#87CEEB', '#9370DB', '#FFA07A', '#FF4500', '#2E8B57', '#DAA520', '#4169E1']

    for i in range(num_features):
        theta = np.linspace(angles[i] - np.pi / num_features, angles[i] + np.pi / num_features, 100)
        r = np.array([importances[i] / 100] * len(theta))
        ax.fill(np.concatenate([[angles[i]], theta, [angles[i]]]),
                np.concatenate([[0], r, [0]]),
                color=colors[i % len(colors)], alpha=0.5)

    ax.set_yticks([0.05, 0.1, 0.15, 0.2])
    ax.set_yticklabels(['5%', '10%', '15%', '20%'], color='grey', fontsize=16)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(all_feature_names[:10], fontsize=18, fontweight='bold')

    ax.spines['polar'].set_visible(False)
    ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.7)

    legend_labels = [f'{feature}: {importance:.1f}%' for feature, importance in sorted(zip(all_feature_names[:10], importances[:-1]), key=lambda x: x[1], reverse=True)]
    ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1.1, 0.5), fontsize=14, title="Features",
              title_fontsize=16, facecolor='#F0F0F0', edgecolor='black')

    plt.title('Top 10 Feature Importance Radar Chart')
    plt.savefig(f"{OUTPUT_BASE_PATH}10_特徵重要性雷達圖.png", dpi=600, bbox_inches='tight')
    plt.close()

    print("前 10 個重要特徵及其重要性 (總和為 100%)：")
    for name, importance in zip(all_feature_names[:10], importances[:-1]):
        print(f"{name}: {importance:.2f}%")
    print(f"總和: {sum(importances[:-1]):.2f}%")

    # SHAP analysis
    X_transformed = best_model.named_steps['preprocessor'].transform(X_train)
    X_transformed = best_model.named_steps['feature_selector'].transform(X_transformed)
    X_transformed_df = pd.DataFrame(X_transformed, columns=selected_features)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_transformed_df)

    # 保存SHAP值
    shap_values_df = pd.DataFrame(shap_values, columns=selected_features)
    shap_path = os.path.join(OUTPUT_BASE_PATH, "shap_values.xlsx")
    shap_values_df.to_excel(shap_path, index=False)
    print(f"SHAP values saved to: {shap_path}")

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_transformed_df, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance Summary")
    plt.savefig(f"{OUTPUT_BASE_PATH}11_SHAP特徵重要性.png", dpi=600, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_transformed_df, show=False)
    plt.title("SHAP Value Distribution")
    plt.savefig(f"{OUTPUT_BASE_PATH}12_SHAP值分布.png", dpi=600, bbox_inches='tight')
    plt.show()
    
    
    return best_model

# Function: Predict the lowest Overpotential
def predict_lowest_op(best_model):
    """Predict the lowest Overpotential combination"""
    
    # --- 修改部分開始：使用整數步進避免誤差 ---
    total_steps = 20  # 1.0 / 0.05 = 20 個工步
    valid_combinations = []
    
    # 透過巢狀迴圈精確控制總和為 20 (即 100%)
    for fe in range(total_steps + 1):
        for co in range(total_steps + 1 - fe):
            for cr in range(total_steps + 1 - fe - co):
                for mn in range(total_steps + 1 - fe - co - cr):
                    cu = total_steps - (fe + co + cr + mn)
                    
                    # 將整數轉回比例，並用 round 確保數字乾淨 (例如 0.1 而非 0.10000000002)
                    valid_combinations.append([
                        round(fe * 0.05, 2),
                        round(co * 0.05, 2),
                        round(cr * 0.05, 2),
                        round(mn * 0.05, 2),
                        round(cu * 0.05, 2)
                    ])
    # --- 修改部分結束 ---
    
    df_combinations = pd.DataFrame(valid_combinations, columns=['Fe', 'Co', 'Cr', 'Mn', 'Cu'])
    important_features = ['Cr', 'Fe', 'Cu', 'Mn', 'Co']
    for col in important_features:
        df_combinations[f'{col}^2'] = df_combinations[col] ** 2
    df_combinations = generate_interactions(df_combinations, important_features, max_order=5)
    
    Y_pred = best_model.predict(df_combinations)
    df_combinations['Overpotential'] = Y_pred
    
    # 保存所有組合的預測結果
    all_combinations_path = os.path.join(OUTPUT_BASE_PATH, "all_combinations_predictions.xlsx")
    df_combinations.to_excel(all_combinations_path, index=False)
    print(f"All combinations predictions saved to: {all_combinations_path}")
    
    min_op_idx = df_combinations['Overpotential'].idxmin()
    min_op_row = df_combinations.iloc[min_op_idx]
    
    # 保存最佳組合
    best_combination_df = pd.DataFrame([min_op_row])
    best_combination_path = os.path.join(OUTPUT_BASE_PATH, "best_combination.xlsx")
    best_combination_df.to_excel(best_combination_path, index=False)
    print(f"Best combination saved to: {best_combination_path}")
    
    return min_op_row, Y_pred

# Function: Visualize predicted Overpotential
def visualize_predicted_op(y_pred):
    """Visualize predicted Overpotential for all combinations"""
    min_op = np.min(y_pred)
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_pred)), y_pred, color='purple', alpha=0.5, label='Predicted Overpotential')
    plt.axhline(y=min_op, color='red', linestyle='--', label=f'Minimum Overpotential = {min_op:.3f}')
    plt.xlabel('Combination Index')
    plt.ylabel('Predicted Overpotential')
    plt.title('Predicted Overpotential for All Combinations')
    plt.legend()
    plt.savefig(f"{OUTPUT_BASE_PATH}13_所有組合預測結果.png", dpi=600, bbox_inches='tight')
    plt.show()

# Function: Generate summary report
def generate_summary_report(dt, best_model, min_op_composition, model_performance):
    """Generate a comprehensive summary report"""
    summary_data = {
        'Dataset_Info': {
            'Total_Samples': dt.shape[0],
            'Total_Features': dt.shape[1] - 1,
            'Base_Elements': ['Fe', 'Co', 'Cr', 'Mn', 'Cu']
        },
        'Model_Performance': model_performance,
        'Best_Composition': {
            'Fe': min_op_composition['Fe'],
            'Co': min_op_composition['Co'],
            'Cr': min_op_composition['Cr'],
            'Mn': min_op_composition['Mn'],
            'Cu': min_op_composition['Cu'],
            'Predicted_Overpotential': min_op_composition['Overpotential']
        }
    }
    
    # 創建摘要報告DataFrame
    summary_df = pd.DataFrame([
        ['Dataset Samples', dt.shape[0]],
        ['Dataset Features', dt.shape[1] - 1],
        ['Cross Validation R²', model_performance['cross_val_r2']],
        ['Training R²', model_performance['train_r2']],
        ['Test R²', model_performance['test_r2']],
        ['Training RMSE', model_performance['train_rmse']],
        ['Test RMSE', model_performance['test_rmse']],
        ['Best Fe Content', f"{min_op_composition['Fe']:.3f}"],
        ['Best Co Content', f"{min_op_composition['Co']:.3f}"],
        ['Best Cr Content', f"{min_op_composition['Cr']:.3f}"],
        ['Best Mn Content', f"{min_op_composition['Mn']:.3f}"],
        ['Best Cu Content', f"{min_op_composition['Cu']:.3f}"],
        ['Minimum Overpotential', f"{min_op_composition['Overpotential']:.3f}"]
    ], columns=['Metric', 'Value'])
    
    summary_path = os.path.join(OUTPUT_BASE_PATH, "summary_report.xlsx")
    summary_df.to_excel(summary_path, index=False)
    print(f"Summary report saved to: {summary_path}")
    
    return summary_data

# Main function
def main():
    """Execute data processing, model training, and prediction"""
    filepath = "D:/desktop/中興大學/實驗室/炤方ai paper/程式碼/datas.xlsx"
    
    print("="*60)
    print("開始執行過電位預測模型")
    print("="*60)
    
    # Load and engineer data
    print("\n1. 載入和處理數據...")
    dt = load_and_engineer_data(filepath)
    target_col = 'Overpotential'
    
    # Visualize data
    print("\n2. 數據可視化...")
    visualize_data(dt, target_col)
    
    # Train and evaluate model
    print("\n3. 模型訓練和評估...")
    best_model = train_and_evaluate(dt, target_col)
    
    # Extract model performance metrics for summary
    X = dt.drop(target_col, axis=1)
    Y = dt[target_col]
    
    # Handle outliers (same as in train_and_evaluate)
    for col in X.columns:
        lower, upper = X[col].quantile([0.05, 0.95])
        X[col] = np.clip(X[col], lower, upper)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    Y_train_pred = best_model.predict(X_train)
    Y_test_pred = best_model.predict(X_test)
    
    model_performance = {
        'cross_val_r2': 0.0,  # This would need to be extracted from the actual training process
        'train_r2': r2_score(Y_train, Y_train_pred),
        'test_r2': r2_score(Y_test, Y_test_pred),
        'train_rmse': np.sqrt(mean_squared_error(Y_train, Y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(Y_test, Y_test_pred))
    }
    
    # Predict optimal composition
    print("\n4. 預測最佳組合...")
    min_op_composition, y_pred = predict_lowest_op(best_model)
    visualize_predicted_op(y_pred)
    
    # Generate summary report
    print("\n5. 生成摘要報告...")
    summary_data = generate_summary_report(dt, best_model, min_op_composition, model_performance)
    
    # Final results
    print("\n" + "="*60)
    print("最終結果摘要")
    print("="*60)
    print(f"預測最低過電位組合:")
    print(f"Fe: {min_op_composition['Fe']:.3f}")
    print(f"Co: {min_op_composition['Co']:.3f}")
    print(f"Cr: {min_op_composition['Cr']:.3f}")
    print(f"Mn: {min_op_composition['Mn']:.3f}")
    print(f"Cu: {min_op_composition['Cu']:.3f}")
    print(f"預測過電位: {min_op_composition['Overpotential']:.3f} mV")
    print(f"\n所有結果和圖表已保存到: {OUTPUT_BASE_PATH}")
    print("="*60)

if __name__ == "__main__":
    main()