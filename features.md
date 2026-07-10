# 最终特征列表（46 个）

## 特征构成

| 类别 | 数量 | 说明 |
|------|------|------|
| 原始数值 | 7 | 直接来自 CSV |
| 有序编码 | 6 | 类别→0/1/2，保留顺序 |
| 简单衍生 | 5 | 加减乘除组合 |
| 类别 OHE | 18 | 6 个原始类别 × 3 取值 |
| 分箱 OHE | 10 | BMI×5 + 心率×5 |
| **合计** | **46** | |

---

## 原始数值（7 个）

直接来自输入数据，缺失值用中位数填充。

| # | 特征 | 类型 |
|---|------|------|
| 1 | sleep_duration | float |
| 2 | heart_rate | float |
| 3 | bmi | float |
| 4 | calorie_expenditure | float |
| 5 | step_count | float |
| 6 | exercise_duration | float |
| 7 | water_intake | float |

---

## 分箱 OHE（10 个）

将连续值按医学标准离散化为区间，再 One-Hot 编码。

### bmi_category（5 列）

| # | 特征 | 条件 |
|---|------|------|
| 8 | bmi_category_underweight | bmi < 18.5 |
| 9 | bmi_category_normal | 18.5 ≤ bmi < 25 |
| 10 | bmi_category_overweight | 25 ≤ bmi < 30 |
| 11 | bmi_category_obese | bmi ≥ 30 |
| 12 | bmi_category_unknown | 缺失 |

**添加理由**：WHO 标准 BMI 分类，肥胖→~83% unhealthy（EDA 验证），正常+活跃→17.6% fit。离散化让模型直接捕捉 BMI>25/30 的阈值效应。

### heart_rate_zone（5 列）

| # | 特征 | 条件 |
|---|------|------|
| 13 | heart_rate_zone_low | hr < 60 |
| 14 | heart_rate_zone_normal | 60 ≤ hr < 80 |
| 15 | heart_rate_zone_elevated | 80 ≤ hr < 100 |
| 16 | heart_rate_zone_high | hr ≥ 100 |
| 17 | heart_rate_zone_unknown | 缺失 |

**添加理由**：临床标准心率区间，捕捉心动过速/过缓的风险信号。

---

## 简单衍生（5 个）

加减乘除四则运算组合原始特征。

| # | 特征 | 公式 | 运算 | 添加理由 |
|---|------|------|------|----------|
| 18 | sleep_debt | 8 - sleep_duration | **减法** | fit 中位 7.95h(debt≈0)，unhealthy 中位 5.37h(debt=2.63)。与理想睡眠的偏差比绝对值更有信息量 |
| 19 | bmi_x_heartrate | bmi × heart_rate / 100 | **乘法** | unhealthy 的 BMI 和心率均偏高，乘积放大差异，捕获双重风险协同 |
| 20 | calorie_per_step | calorie / (step_count + 1) | **除法 ratio** | 每步能耗反映代谢效率，不受步数绝对值影响 |
| 21 | exercise_intensity | calorie / (exercise_duration + 0.1) | **除法 ratio** | 单位时间能耗反映运动强度，比绝对值更稳定 |
| 22 | water_per_kg | water / (bmi + 0.1) | **除法 ratio** | 饮水量绝对值三类几乎相同，相对 BMI 后反映真实补水状态 |

---

## 有序编码（6 个）

将有序类别映射为 0/1/2 数值，保留顺序关系。相比 OHE（三个独立列），CatBoost 能直接做阈值 split。

| # | 特征 | 来源 | 映射 |
|---|------|------|------|
| 23 | stress_level_ord | stress_level | low=0, medium=1, high=2 |
| 24 | sleep_quality_ord | sleep_quality | poor=0, average=1, good=2 |
| 25 | physical_activity_level_ord | physical_activity_level | sedentary=0, moderate=1, active=2 |
| 26 | smoking_alcohol_ord | smoking_alcohol | no=0, occasional=1, yes=2 |
| 27 | diet_type_ord | diet_type | veg=0, balanced=1, non-veg=2 |
| 28 | gender_ord | gender | female=0, male=1, other=2 |

**添加理由**：EDA 显示 stress_level 几乎完全决定类别（high→97.5% unhealthy），顺序至关重要。OHE 丢失了 low<medium<high 的次序。

---

## 类别 OHE（18 个）

原始类别变量 One-Hot 编码（6 类 × 3 取值 = 18 列）。

| # | 特征 |
|---|------|
| 29 | diet_type_balanced |
| 30 | diet_type_non-veg |
| 31 | diet_type_veg |
| 32 | stress_level_high |
| 33 | stress_level_low |
| 34 | stress_level_medium |
| 35 | sleep_quality_average |
| 36 | sleep_quality_good |
| 37 | sleep_quality_poor |
| 38 | physical_activity_level_active |
| 39 | physical_activity_level_moderate |
| 40 | physical_activity_level_sedentary |
| 41 | smoking_alcohol_no |
| 42 | smoking_alcohol_occasional |
| 43 | smoking_alcohol_yes |
| 44 | gender_female |
| 45 | gender_male |
| 46 | gender_other |

---

## 训练参数

```python
loss_function: 'MultiClass'
auto_class_weights: 'Balanced'    # 处理 85.9%/8.4%/5.8% 不平衡
task_type: 'GPU'
early_stopping_rounds: 50
use_best_model: True              # 回滚到最佳迭代
# 其余默认: depth=6, lr=auto, iterations=1000
```

## 结果

Balanced Accuracy: **0.9493** （5折CV）
Overall Accuracy: **94%**
