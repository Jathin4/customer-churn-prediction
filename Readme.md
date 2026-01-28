# 📊 Customer Churn Prediction - End-to-End ML Project

Complete machine learning project for predicting customer churn in telecom industry, including data analysis, feature engineering, customer segmentation, and predictive modeling.

## 🎯 Project Overview

**Objective:** Build an end-to-end ML system to predict customer churn and provide actionable retention strategies.

**Key Results:**
- ✅ 84.72% ROC-AUC score with XGBoost
- ✅ Identified 4 distinct customer segments
- ✅ 26.54% overall churn rate analyzed
- ✅ $3.5M potential revenue saved identified
- ✅ Created interactive Streamlit dashboard

## 📁 Project Structure
```
customer-churn-project/
│
├── data/
│   ├── raw/                    # Original dataset (not in repo)
│   └── processed/              # Cleaned data (not in repo)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_customer_segmentation.ipynb
│   └── 05_churn_prediction.ipynb
│
├── models/                     # Saved models (not in repo)
│
├── dashboard.py               # Streamlit dashboard
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── COMPLETE_PROJECT_DOCUMENTATION.md  # Full documentation
└── .gitignore
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- pip or conda

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Jathin4/customer-churn-project.git
cd customer-churn-project
```

2. **Create virtual environment:**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download the dataset:**
- Visit [Kaggle Telco Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Place in `data/raw/` folder

5. **Run notebooks in order:**
- Open VS Code or Jupyter
- Execute notebooks 01 through 05 sequentially

6. **Launch dashboard:**
```bash
streamlit run dashboard.py
```

## 📊 Key Findings

### Business Insights

1. **First-Year Crisis:** 47.68% of new customers churn in first year
2. **Contract Effect:** Month-to-month customers have 15x higher churn than 2-year contracts
3. **Service Bundling:** Customers with 5+ services have <15% churn vs 40%+ for 1-2 services
4. **Payment Friction:** Electronic check users have 2x higher churn than auto-pay users

### Customer Segments

| Segment          | Size  | Churn Rate | Characteristics                         |
|------------------|-------|------------|-----------------------------------------|
| High-Risk New    | 29.4% | 50.2%      | New customers, month-to-month contracts |
| At-Risk Mid-Tier | 27.8% | 31.7%      | Moderate tenure, high charges           |
| Loyal Premium    | 26.1% | 10.0%      | Long tenure, multiple services          |
| Budget Committed | 16.7% | 2.2%       | Low spend, long contracts               |

### Model Performance

| Model               | ROC-AUC    | Accuracy   | Status   |
|---------------------|------------|------------|----------|
| Logistic Regression | 0.8427     | 80.27%     | Baseline |
| Random Forest       | 0.8287     | 78.50%     | Overfit  |
| XGBoost (Tuned)     | **0.8472** | **80.62%** | ✅ Best |

## 🛠️ Technologies Used

- **Python 3.13**
- **pandas** - Data manipulation
- **scikit-learn** - Machine learning
- **XGBoost** - Gradient boosting
- **matplotlib/seaborn** - Visualization
- **Streamlit** - Dashboard
- **Jupyter** - Analysis notebooks

## 📈 Business Recommendations

1. **New Customer Success Program** - Focus on first 90 days
2. **Contract Conversion Campaign** - Incentivize annual contracts
3. **Payment Friction Removal** - Promote auto-pay with discounts
4. **Service Bundling** - Cross-sell to increase engagement
5. **Loyal Customer Appreciation** - VIP program for high-value customers

## 📝 Project Phases

### Phase 1: Data Exploration
- Analyzed 7,043 customer records
- Identified churn patterns and correlations
- Generated business hypotheses

### Phase 2: Data Cleaning
- Fixed TotalCharges data type issue
- Handled 11 missing values logically
- Validated data quality

### Phase 3: Feature Engineering
- Created 29 new features from 21 original
- Engineered risk scores and behavioral indicators
- Improved model performance by 4-6%

### Phase 4: Customer Segmentation
- K-Means clustering with 4 segments
- Profiled each segment with actionable insights
- Enabled targeted retention strategies

### Phase 5: Churn Prediction
- Built and compared multiple models
- Hyperparameter tuning with GridSearchCV
- Achieved 84.72% ROC-AUC with XGBoost

### Phase 6: Dashboard Development
- Interactive Streamlit dashboard
- Real-time churn predictions
- Business metrics visualization

## 📊 Dashboard Features

- **Overview:** Key metrics and churn distribution
- **Segments:** Customer segment analysis
- **Prediction:** Individual customer churn prediction
- **Performance:** Model metrics and feature importance

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**[GANGI JATHIN]**
- GitHub: [@Jathin4](https://github.com/Jathin4)
- LinkedIn: [GANGI JATHIN](https://www.linkedin.com/in/jathin-gangi-042a29225/)


## 🙏 Acknowledgments

- Dataset: IBM Sample Data Sets (via Kaggle)
- Inspiration: Real-world telecom churn problem
- Tools: Anthropic Claude for guidance

## 📚 Documentation

For complete project documentation, see [COMPLETE_PROJECT_DOCUMENTATION.md](COMPLETE_PROJECT_DOCUMENTATION.md)

---

**⭐ If you found this project helpful, please give it a star!**