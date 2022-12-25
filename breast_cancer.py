from pycaret.anomaly import *
from sklearn.datasets import load_breast_cancer

# Load the dataset
df = load_breast_cancer(as_frame=True)['data']
df_train = df.iloc[:-10]
df_unseen = df.tail(10)

# Set up Pycaret
anom = setup(data=df_train,
             silent=True)

# Create Isolation Forest Model
anom_model = create_model(model='iforest', fraction=0.05)  # 0.05 indicates that the dataset has 5% of outliers.
results = assign_model(anom_model)

# 3D and 2D Plotting
plot_model(anom_model, plot='tsne')
plot_model(anom_model, plot='umap')

# Save model
save_model(model=anom_model, model_name='iforest_model')

# Load model
loaded_model = load_model('iforest_model')  # type(loaded_model) >> sklearn.pipeline.Pipeline

# Predict unseen data
print(loaded_model.predict(df_unseen))
print(loaded_model.predict_proba(df_unseen))

# Return the anomaly score
print(loaded_model.decision_function(df_unseen))
