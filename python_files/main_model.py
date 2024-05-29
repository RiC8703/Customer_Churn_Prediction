from pathlib import Path
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report


def one_hot_coding(df):
    categorical_columns=df.select_dtypes(include='object').columns
    one_hot_encoders={}

    #iterate for each categorical cloumn and fit one hot encoder
    for column in categorical_columns:
        encoder=OneHotEncoder(sparse=False)
        encoder.fit(df[[column]])
        encoded_features=encoder.transform(df[[column]])
        encoded_data=pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out([column]))
        df=pd.concat([df.drop(column, axis=1), encoded_data], axis=1)
        one_hot_encoders[column]=encoder
    return df, one_hot_encoders


def data_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def normalization(X):
    continuos_columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    scaler=MinMaxScaler()

    #fit scaler on training data
    scaler.fit(X[continuos_columns])

    #normalization
    X[continuos_columns]=scaler.transform(X[continuos_columns])
    return X, scaler


def perform_oversampling(X, y):
    smote = SMOTE()
    X_oversampled, y_oversampled = smote.fit_resample(X, y)
    return X_oversampled, y_oversampled


if __name__ == "__main__":
    df=pd.read_csv(str(Path(__file__).parents[1] / 'data/Churn_Modelling.csv'))

    #drop fields not used for training
    drop_columns = ['RowNumber', 'CustomerId', 'Surname']
    df.drop(drop_columns, axis=1, inplace=True)

    #splitting features and label
    X=df.drop('Exited', axis=1) #Features
    y=df['Exited'] #target variable

    X, one_hot_encoder = one_hot_coding(X) #onehot encoding
    X, scaler = normalization(X) #normalization
    X_oversampled, y_oversampled = perform_oversampling(X, y) #oversampling
    
    X_train, X_test, y_train, y_test = data_split(X_oversampled, y_oversampled) #train test split

    #model creation
    rf_classifier=RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    #evaluate model
    y_pred=rf_classifier.predict(X_test)
    print(classification_report(y_test, y_pred))

    #save model
    with open(str(Path(__file__).parents[1] / 'model/model.pickle'), 'wb') as f:
        pickle.dump((rf_classifier, one_hot_encoder, scaler), f)