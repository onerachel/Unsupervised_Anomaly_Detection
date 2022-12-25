from pycaret.anomaly import *
import pandas as pd

# Load the dataset
df = pd.read_csv(
    'https://hr-projects-assets-prod.s3.amazonaws.com/9omh71m21sj/a6cca10ee275c7a89fc3f4f2a257717b/unsupervisedLearningData.csv')
df_train = df.copy()

# Set up Pycaret - Pre-processing data
''' 
missing values: PyCaret by default imputes the missing value in the dataset by mean for numeric features and constant for categorical features
feature engineering: one-hot encoding and ordinal encoding for the categorical features
transformation: By default, the feature transformation method is set to yeo-johnson
ignore_features: we drop not influenceable columns
'''
anom = setup(data=df_train,
             transformation=True,
             categorical_features=['b2', 'c18'],
             ordinal_features={'c3': ['7e8b1406d903', '2f169f9b4e6a'],
                               'c9': ['7e8b1406d903', '2f169f9b4e6a'],
                               'c15': ['7e8b1406d903', '2f169f9b4e6a'],
                               'c29': ['7e8b1406d903', '2f169f9b4e6a'],
                               'c20': ['7e8b1406d903', '2f169f9b4e6a', 'a5a4f007abc4'],
                               'c26': ['af4b5bc5ca2f', '685dc962baea', 'af2f2ec57ada']},
             ignore_features=['idnr', 'c6', 'c14', 'c19'],
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
results.to_csv('./data/results.csv')

# Plotting
plot_model(anom_model, plot='tsne')
plot_model(anom_model, plot='umap')

# Save model
save_model(model=anom_model, model_name='iforest_model')

# Make submission file
results.rename(columns={'Anomaly': 'is_outlier'}, inplace=True)
submission = results[['idnr', 'is_outlier']].set_index('idnr')
submission.to_csv('./data/solution.csv')
