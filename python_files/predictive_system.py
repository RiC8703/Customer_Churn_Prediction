from pathlib import Path
import pandas as pd
import pickle

def predict(X):
    with open(str(Path(__file__).parents[1] / 'model/model.pickle'), 'rb') as f:
        rf_classifier, one_hot_encoder, scaler = pickle.load(f)

    #onehot encoding
    for column, encoder in one_hot_encoder.items():
        encoded_features=encoder.transform(X[[column]])
        encoded_data=pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out([column]))
        X=pd.concat([X.drop(column, axis=1), encoded_data], axis=1)

    #normalisation
    continuos_columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    X[continuos_columns]=scaler.transform(X[continuos_columns])

    pred=rf_classifier.predict(X)
    return pred
