import pandas as pd
import numpy as np
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
DATA_DIR = Path('input')

# ============================================================
# 1. LOAD DATA
# ============================================================
logger.info("=" * 60)
logger.info("STEP 1: LOADING DATA")
logger.info("=" * 60)

train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')

logger.info(f"Train shape: {train.shape}")
logger.info(f"Test shape: {test.shape}")
logger.info(f"Train columns: {list(train.columns)}")
logger.info(f"Test columns: {list(test.columns)}")

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================
logger.info("=" * 60)
logger.info("STEP 2: EXPLORATORY DATA ANALYSIS")
logger.info("=" * 60)

# ---- 2a. Target distribution ----
logger.info(f"\nTarget distribution:\n{train['health_condition'].value_counts()}")
logger.info(f"\nTarget distribution (%):\n{train['health_condition'].value_counts(normalize=True).mul(100).round(2)}")

# ---- 2b. Missing values ----
logger.info("\n--- Missing Values ---")
for name, df in [('Train', train), ('Test', test)]:
    miss = df.isnull().sum()
    miss_df = pd.DataFrame({'count': miss, 'pct': df.isnull().mean().mul(100).round(2)})
    miss_df = miss_df[miss_df['count'] > 0].sort_values('count', ascending=False)
    logger.info(f"{name} missing:\n{miss_df}")

# ---- 2c. Numerical features ----
num_cols = ['sleep_duration', 'heart_rate', 'bmi', 'calorie_expenditure',
            'step_count', 'exercise_duration', 'water_intake']
cat_cols = ['diet_type', 'stress_level', 'sleep_quality',
            'physical_activity_level', 'smoking_alcohol', 'gender']

for col in num_cols:
    train[col] = pd.to_numeric(train[col], errors='coerce')
    test[col] = pd.to_numeric(test[col], errors='coerce')

logger.info(f"\nNumerical stats:\n{train[num_cols].describe().round(2)}")

# ---- 2d. Target vs numerical (mean) ----
target_mean = train.groupby('health_condition')[num_cols].mean().round(2)
logger.info(f"\nTarget vs Numerical (mean):\n{target_mean}")

# ---- 2e. Categorical distributions ----
logger.info("\n--- Categorical Feature Distributions ---")
for col in cat_cols:
    vc = train[col].value_counts(dropna=False)
    logger.info(f"\n{col}:\n{vc}")

# ---- 2f. Target vs categorical ----
logger.info("\n--- Target vs Categorical (% by row) ---")
for col in cat_cols:
    ct = pd.crosstab(train[col], train['health_condition'], normalize='index').mul(100).round(1)
    logger.info(f"\n{col}:\n{ct}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
# 设计依据（EDA 结论）:
#   1. sleep_duration 是最强预测因子: fit(7.95h) > at-risk(7.09h) > unhealthy(5.37h)
#   2. stress_level 几乎决定类别: high→97.5% unhealthy, low→20% fit, medium→99.4% at-risk
#   3. physical_activity: active→96.8% fit, sedentary→91.7% at-risk
#   4. bmi: unhealthy(24.1) > at-risk(23.0) > fit(21.8)
#   5. smoking_alcohol: yes→45% unhealthy, no→45% fit
#   6. sleep_quality: poor→54% unhealthy, good→46% fit
#   7. 数值特征间几乎独立（corr ~ 0）
#   8. 缺失率在各类间一致（MCAR）
# ============================================================
logger.info("=" * 60)
logger.info("STEP 3: FEATURE ENGINEERING")
logger.info("=" * 60)

def engineer_features(df, is_train=True):
    df = df.copy()

    # ---- 3a. 分箱特征（连续值离散化为医学标准区间）----
    # BMI 分类（WHO 标准）
    # EDA: obese+any activity→~83% unhealthy; normal+active→17.6% fit
    # 离散化后模型可直接捕捉 BMI>25(超重) 和 BMI>30(肥胖) 的阈值效应
    df['bmi_category'] = pd.cut(df['bmi'],
                                 bins=[0, 18.5, 25, 30, 100],
                                 labels=['underweight', 'normal', 'overweight', 'obese'])
    logger.info(f"  [3a] bmi_category: 将 bmi 离散化为 underweight/normal/overweight/obese")

    # 心率区间（临床标准）
    # EDA: heart_rate 均值三类接近(74.8~75.3)，但极端值分布不同
    # 离散化后模型可捕捉 hr>100(心动过速) 和 hr<60(心动过缓) 的风险信号
    df['heart_rate_zone'] = pd.cut(df['heart_rate'],
                                    bins=[0, 60, 80, 100, 300],
                                    labels=['low', 'normal', 'elevated', 'high'])
    logger.info(f"  [3a] heart_rate_zone: 将 heart_rate 离散化为 low/normal/elevated/high")

    # ---- 3b. 简单衍生特征（加减乘除四则运算）----
    # sleep_debt = 8 - sleep_duration
    # EDA: fit 中位睡眠 7.95h(~0 debt), unhealthy 中位 5.37h(debt=2.63)
    # 比原始 sleep_duration 更直接反映"与理想睡眠的差距"
    df['sleep_debt'] = 8 - df['sleep_duration']
    logger.info(f"  [3b] sleep_debt: 8 - sleep_duration, 与理想睡眠的偏差（减法）")

    # bmi_x_heartrate = bmi × heart_rate / 100
    # EDA: unhealthy 的 bmi 和 heart_rate 均偏高，乘积放大差异
    # 捕获"高 BMI + 高心率"的双重风险协同效应（乘法交互）
    df['bmi_x_heartrate'] = df['bmi'] * df['heart_rate'] / 100
    logger.info(f"  [3b] bmi_x_heartrate: bmi × heart_rate / 100, 体态×心负荷的协同风险（乘法）")

    # calorie_per_step = calorie / (step_count + 1)
    # EDA: fit 步数(12342)显著高于 at-risk(8483)和 unhealthy(8941)
    # 每步能耗反映代谢效率，不受步数绝对值影响（除法 ratio）
    df['calorie_per_step'] = df['calorie_expenditure'] / (df['step_count'] + 1)
    logger.info(f"  [3b] calorie_per_step: calorie_expenditure / step_count, 每步能耗（除法 ratio）")

    # exercise_intensity = calorie / (exercise_duration + 0.1)
    # EDA: fit 运动时长(50min)高于其他(38~39min)，但卡路里差异更小
    # 单位时间能耗反映运动强度，比绝对值更稳定（除法 ratio）
    df['exercise_intensity'] = df['calorie_expenditure'] / (df['exercise_duration'] + 0.1)
    logger.info(f"  [3b] exercise_intensity: calorie / exercise_duration, 运动强度（除法 ratio）")

    # water_per_kg = water / (bmi + 0.1)
    # EDA: 饮水量绝对值三类几乎相同(~2.18L)，但 bmi 不同
    # 单位 BMI 的饮水量反映相对补水状态（除法 ratio）
    df['water_per_kg'] = df['water_intake'] / (df['bmi'] + 0.1)
    logger.info(f"  [3b] water_per_kg: water_intake / bmi, 单位 BMI 饮水量（除法 ratio）")

    # ---- 3c. 有序编码（保留类别间的顺序关系）----
    # OHE 将类别变为独立二元列，丢失了 low<medium<high 的顺序
    # 有序编码让 CatBoost 能直接做 "stress_level_ord >= 1" 这样的 split，
    # 比 OHE 的三个独立列更高效
    # EDA: stress_level 几乎完全决定类别，顺序至关重要
    ord_maps = {
        'stress_level_ord': ('stress_level', {'low': 0, 'medium': 1, 'high': 2}),
        'sleep_quality_ord': ('sleep_quality', {'good': 2, 'average': 1, 'poor': 0}),
        'physical_activity_level_ord': ('physical_activity_level', {'sedentary': 0, 'moderate': 1, 'active': 2}),
        'smoking_alcohol_ord': ('smoking_alcohol', {'no': 0, 'occasional': 1, 'yes': 2}),
        'diet_type_ord': ('diet_type', {'veg': 0, 'balanced': 1, 'non-veg': 2}),
        'gender_ord': ('gender', {'female': 0, 'male': 1, 'other': 2}),
    }
    for new_col, (src, m) in ord_maps.items():
        df[new_col] = df[src].map(m)
    logger.info(f"  [3c] 有序编码: 将 6 个有序类别映射为 0/1/2 数值，保留顺序关系")

    # ---- 3d. 缺失值填充 ----
    # EDA: 缺失率在各类间一致（MCAR），用中位数/众数填充不会引入偏见
    for col in num_cols:
        if is_train:
            fill_val = df[col].median()
            setattr(engineer_features, f'{col}_fill', fill_val)
        else:
            fill_val = getattr(engineer_features, f'{col}_fill', df[col].median())
        df[col] = df[col].fillna(fill_val)

    for col in cat_cols:
        if is_train:
            fill_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'unknown'
            setattr(engineer_features, f'{col}_fill', fill_val)
        else:
            fill_val = getattr(engineer_features, f'{col}_fill', 'unknown')
        df[col] = df[col].fillna(fill_val)

    # 分箱后的 NaN（原始值缺失导致）填充为 'unknown'
    for col in ['bmi_category', 'heart_rate_zone']:
        if hasattr(df[col], 'cat'):
            df[col] = df[col].cat.add_categories('unknown')
        df[col] = df[col].fillna('unknown')

    logger.info(f"  [3d] 缺失值填充完成")
    return df

train_fe = engineer_features(train, is_train=True)
test_fe = engineer_features(test, is_train=False)

# ---- 特征组合说明 ----
# 最终的 46 个特征由以下部分组成:
#   7 原始数值      — 原始数据直接提供
#   6 有序编码      — 类别变量保留顺序(3c)
#   5 简单衍生      — 加减乘除组合(3b)
#  18 OHE          — 6 个原始类别展开为 dummy(3e)
#  10 分箱 OHE     — 2 个连续值离散化(3a)
logger.info(f"\n{'='*60}")
logger.info("FEATURE COMPOSITION")
logger.info(f"{'='*60}")
logger.info(f"  7 原始数值: {num_cols}")
logger.info(f"  6 有序编码: stress_level_ord, sleep_quality_ord, physical_activity_level_ord, smoking_alcohol_ord, diet_type_ord, gender_ord")
logger.info(f"  5 简单衍生: sleep_debt, bmi_x_heartrate, calorie_per_step, exercise_intensity, water_per_kg")
logger.info(f" 18 OHE: 6 个原始类别 × 3 取值")
logger.info(f" 10 分箱 OHE: bmi_category(5) + heart_rate_zone(5)")

# ---- 3e. One-Hot Encoding ----
logger.info(f"\n  OHE 类别: {cat_cols + ['bmi_category', 'heart_rate_zone']}")
cat_to_ohe = cat_cols + ['bmi_category', 'heart_rate_zone']

train_fe = pd.get_dummies(train_fe, columns=cat_to_ohe, drop_first=False, dummy_na=False)
test_fe = pd.get_dummies(test_fe, columns=cat_to_ohe, drop_first=False, dummy_na=False)

# 对齐训练集和测试集的列（确保 OHE 列一致）
train_fe, test_fe = train_fe.align(test_fe, join='inner', axis=1)
logger.info(f"\nAfter OHE — Train: {train_fe.shape}, Test: {test_fe.shape}")

# 删除 id 列
id_cols = [c for c in train_fe.columns if c == 'id']
if id_cols:
    train_fe = train_fe.drop(columns=id_cols)
    test_fe = test_fe.drop(columns=id_cols)
    logger.info(f"  已删除 id 列（无预测信息）")

logger.info(f"Final features — Train: {train_fe.shape}, Test: {test_fe.shape}")
final_cols = list(train_fe.columns)
logger.info(f"Final feature list:\n{final_cols}")

X = train_fe.values
y = train['health_condition'].values
X_test = test_fe.values
ids_test = test['id'].values

logger.info(f"\nX shape: {X.shape}")
logger.info(f"X_test shape: {X_test.shape}")
logger.info(f"Target classes: {np.unique(y)}")

# ============================================================
# 4. CATBOOST 5-FOLD CROSS-VALIDATION
# ============================================================
logger.info("=" * 60)
logger.info("STEP 4: CATBOOST 5-FOLD CROSS-VALIDATION")
logger.info("=" * 60)

from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_enc = le.fit_transform(y)
logger.info(f"Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

cv_scores = []
oof_preds = np.zeros(len(y_enc))
test_preds = np.zeros((len(X_test), len(le.classes_)))

# CatBoost 参数说明:
#   loss_function=MultiClass — 三分类任务
#   auto_class_weights=Balanced — 处理85.9%/8.4%/5.8%的类别不平衡
#   early_stopping_rounds=50 — 验证集50轮无改善则停止
#   use_best_model=True — 回滚到最佳迭代（在 .fit() 中传入）
#   其余参数保持默认：depth=6, lr=auto, iterations=1000
cb_params = {
    'loss_function': 'MultiClass',
    'auto_class_weights': 'Balanced',
    'random_seed': SEED,
    'task_type': 'GPU',
    'thread_count': -1,
    'verbose': 50,
    'early_stopping_rounds': 50,
}

logger.info(f"CatBoost params: {cb_params}")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_enc)):
    logger.info(f"\n{'='*40}")
    logger.info(f"FOLD {fold + 1}/{n_splits}")
    logger.info(f"{'='*40}")

    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y_enc[train_idx], y_enc[val_idx]

    logger.info(f"Train size: {X_tr.shape[0]}, Val size: {X_val.shape[0]}")
    logger.info(f"Train class dist: {np.bincount(y_tr)}")
    logger.info(f"Val class dist:   {np.bincount(y_val)}")

    model = CatBoostClassifier(**cb_params)

    start_time = datetime.now()
    model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        use_best_model=True,
        logging_level='Verbose',
    )
    elapsed = (datetime.now() - start_time).total_seconds()
    best_iter = model.best_iteration_ if hasattr(model, 'best_iteration_') else 'N/A'
    logger.info(f"Training time: {elapsed:.1f}s (best iteration: {best_iter})")

    val_prob = model.predict_proba(X_val)
    val_pred = val_prob.argmax(axis=1)
    oof_preds[val_idx] = val_pred

    bal_acc = balanced_accuracy_score(y_val, val_pred)
    cv_scores.append(bal_acc)
    logger.info(f"Fold {fold + 1} Balanced Accuracy: {bal_acc:.6f}")

    test_prob = model.predict_proba(X_test)
    test_preds += test_prob / n_splits

    # 输出每折 Top 10 特征重要性
    imp = pd.DataFrame({
        'feature': train_fe.columns,
        f'importance_fold{fold+1}': model.feature_importances_
    }).sort_values(f'importance_fold{fold+1}', ascending=False)
    logger.info(f"\nTop 10 features fold {fold+1}:\n{imp.head(10).to_string(index=False)}")

logger.info(f"\n{'='*60}")
logger.info(f"CV RESULTS — {n_splits}-FOLD STRATIFIED")
logger.info(f"{'='*60}")
logger.info(f"Per-fold balanced accuracies: {[f'{s:.6f}' for s in cv_scores]}")
logger.info(f"Mean balanced accuracy: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")

oof_bal_acc = balanced_accuracy_score(y_enc, oof_preds.astype(int))
logger.info(f"OOF Balanced Accuracy: {oof_bal_acc:.6f}")

from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_enc, oof_preds.astype(int))
logger.info(f"\nOOF Confusion Matrix:\n{cm}")
logger.info(f"\nOOF Classification Report:\n{classification_report(y_enc, oof_preds.astype(int), target_names=le.classes_)}")

# ============================================================
# 5. SUBMISSION
# ============================================================
logger.info("=" * 60)
logger.info("STEP 5: GENERATING SUBMISSION")
logger.info("=" * 60)

test_labels = le.inverse_transform(test_preds.argmax(axis=1))
sub = pd.DataFrame({'id': ids_test, 'health_condition': test_labels})
sub.to_csv('submission.csv', index=False)
logger.info(f"\nSubmission saved to submission.csv")
logger.info(f"Shape: {sub.shape}")
logger.info(f"Submission head:\n{sub.head(10)}")
logger.info(f"Submission target distribution:\n{sub['health_condition'].value_counts()}")

logger.info("\n" + "=" * 60)
logger.info("DONE")
logger.info("=" * 60)
