from pycaret.anomaly import *
from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load the dataset
df = pd.read_csv(
    'https://www.kaggle.com/code/danushkumarv/credit-card-fraud-analysis-98-45/data?select=creditcard.csv')
df_copy = df.copy()
df_train = df_copy.drop(['Class', 'Time'])
# df_train = df_copy.iloc[:1000]  ## try out on smaller batch

# Set up Pycaret - Pre-processing data
anom = setup(data=df_train,
             silent=True,
             )

# Create Isolation Forest Model
anom_model = create_model(model='iforest', fraction=0.04)  # 0.04 indicates that the dataset has 4% of outliers.

# Evaluation (Cook's distance)
results = assign_model(anom_model)

# Save labeled file with two additional columns
'''
column 'Anomaly': is binary where 1 indicates that the record is anomalous and 0 indicates that it is normal
column 'Anomaly_Score': gives the raw score for the record, where negative indicates that the record is normal
'''
results.to_csv('./data/results_fraud.csv')

# Plotting
plot_model(anom_model, plot='tsne')
plot_model(anom_model, plot='umap')

# Save model
save_model(model=anom_model, model_name='iforest_model')
