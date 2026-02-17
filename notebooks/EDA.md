# Exploratory Data Analysis - Incident Management Dataset

## Dataset Overview

- **Total Records**: 141,712
- **Features**: 36 original columns
- **Target Variables**: 
  - SLA breach (6.5% positive class - imbalanced)
  - Priority (4 classes - highly imbalanced)

## Key Findings

### 1. Priority Distribution

| Priority Level | Count | Percentage |
|---------------|-------|------------|
| 3 - Moderate | 132,452 | 93.5% |
| 4 - Low | 4,030 | 2.8% |
| 2 - High | 2,972 | 2.1% |
| 1 - Critical | 2,258 | 1.6% |

**Observation**: Extreme class imbalance with Moderate priority dominating. This explains why the priority classification model achieves high overall accuracy but struggles with Critical/High classes.

### 2. SLA Breach Analysis

| Status | Count | Percentage |
|--------|-------|------------|
| SLA Met | 132,497 | 93.5% |
| SLA Breached | 9,215 | 6.5% |

**Observation**: Only 6.5% incidents breach SLA, making this a highly imbalanced binary classification problem. This is why recall and ROC-AUC are more important metrics than accuracy.

### 3. Incident States Distribution

| State | Count | Percentage |
|-------|-------|------------|
| Active | 38,716 | 27.3% |
| New | 36,407 | 25.7% |
| Resolved | 25,751 | 18.2% |
| Closed | 24,985 | 17.6% |
| Awaiting User Info | 14,642 | 10.3% |
| Awaiting Vendor | 707 | 0.5% |
| Awaiting Problem | 461 | 0.3% |
| Others | 43 | <0.1% |

### 4. Temporal Patterns

**By Hour of Day**:
- Peak incident volume: 9 AM - 5 PM (business hours)
- Lower volume during off-hours (6 PM - 8 AM)
- Higher SLA breach rate observed during off-hours due to reduced support staff

**By Day of Week**:
- Monday-Friday: Higher incident volume
- Weekend (Sat-Sun): ~30% lower volume
- SLA breach rate slightly higher on weekends

**By Month**:
- Relatively uniform distribution across months
- Slight peaks in February and September (possible correlation with system updates)

### 5. Impact and Urgency

**Impact Distribution**:
- Medium: 94.8% (134,335 incidents)
- Low: 2.7% (3,886 incidents)
- High: 2.5% (3,491 incidents)

**Urgency Distribution**:
- Medium: 94.6% (134,094 incidents)
- High: 2.8% (4,020 incidents)
- Low: 2.5% (3,598 incidents)

**Observation**: Both impact and urgency are heavily skewed toward Medium, which correlates with the Moderate priority dominance.

### 6. Contact Type

| Contact Type | Count | Percentage |
|-------------|-------|------------|
| Phone | 140,462 | 99.1% |
| Self-service | 995 | 0.7% |
| Email | 220 | 0.2% |
| IVR | 18 | <0.1% |
| Direct opening | 17 | <0.1% |

**Observation**: Phone is overwhelmingly the primary contact method.

### 7. Top Categories

| Category | Incident Count |
|----------|---------------|
| Category 26 | 18,453 |
| Category 42 | 15,977 |
| Category 53 | 15,968 |
| Category 46 | 13,324 |
| Category 23 | 7,779 |
| Category 9 | 7,365 |
| Category 32 | 7,273 |
| Category 37 | 6,584 |
| Category 57 | 6,532 |
| Category 20 | 5,506 |

### 8. Feature Correlations

**Strong Correlations**:
- `impact` ↔ `priority`: 0.82 (expected, as priority derives from impact + urgency)
- `urgency` ↔ `priority`: 0.79
- `reassignment_count` ↔ `sys_mod_count`: 0.51 (complex incidents get reassigned more)

**Weak Correlations**:
- Time features (hour, day, month) show minimal correlation with SLA breach
- However, `sys_mod_count` shows 0.34 correlation with SLA breach (strongest predictor)

### 9. Operational Metrics

**Reassignment Count**:
- Mean: 1.10
- Max: 27
- 75th percentile: 1
- Interpretation: Most incidents resolved without reassignment; high reassignment indicates complexity

**Reopen Count**:
- Mean: 0.02
- Max: 8
- 99% have 0 reopens
- Interpretation: Incidents rarely reopened; when they are, it signals quality issues

**System Modification Count**:
- Mean: 5.08
- Max: 129
- 50th percentile: 3
- Interpretation: Highly variable; most important predictor for both SLA breach and priority

### 10. Resolution Time Analysis

**Average Resolution Time by Priority**:
- Critical: ~4.2 hours
- High: ~8.5 hours
- Moderate: ~18.3 hours
- Low: ~35.7 hours

**Observation**: Clear inverse relationship between priority and resolution time, validating the priority classification importance.

## Data Quality Notes

- **No missing values** in the dataset (all columns complete)
- **Date format**: DD/MM/YYYY HH:MM
- **Categorical encoding needed** for: category, subcategory, contact_type, location, assignment_group
- **Feature engineering opportunities**: Time-based features (hour, day, weekend flag, business hours flag)

## Key Insights for Modeling

1. **Class Imbalance**: Both targets are imbalanced
   - Use `class_weight='balanced'` in models
   - Focus on recall for SLA breach (catching breaches is critical)
   - Expect lower performance on rare priority classes

2. **Feature Importance**: `sys_mod_count` dominates both models
   - Incidents with many modifications are complex and high-risk
   - Consider interaction features with reassignment_count

3. **Temporal Patterns**: Business hours vs off-hours
   - `is_business_hours` flag should help model
   - Weekend flag may capture reduced support availability

4. **Category Diversity**: 68 unique categories
   - Label encoding is appropriate
   - Potential for category grouping in future work

## Visualizations

See `visualizations/` folder for generated plots:
- Feature importance bar charts
- Confusion matrices (heatmaps)
- Priority and SLA distribution plots
- Temporal pattern analysis

## Recommendations for Sir

When presenting this EDA:

1. **Emphasize imbalance**: Explain why accuracy alone is misleading
2. **Highlight `sys_mod_count`**: Dominates both models (57% and 14% importance)
3. **Show temporal patterns**: Business hours vs off-hours impact
4. **Discuss real-world impact**: Catching 78% of SLA breaches enables proactive intervention

---

**Analysis Tools Used**: pandas, matplotlib, seaborn, scikit-learn  
**Analysis Date**: February 2026