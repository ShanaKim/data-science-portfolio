import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, recall_score
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from clean import process
from sklearn.metrics import log_loss


def fit_logistic_regression(df, target_col, test_size=0.3, random_state=42, max_iter=1000):
    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # identify categorical columns
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    # One-Hot Encoding
    encoder = OneHotEncoder(handle_unknown="ignore", drop = 'first', sparse_output=False)
    encoder.fit(X_train[categorical_cols])  # Fit on training data only

    # tranform train Test Sets
    X_train_encoded = encoder.transform(X_train[categorical_cols])
    X_test_encoded = encoder.transform(X_test[categorical_cols])

    # conv to df
    X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(), index=X_train.index)
    X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(), index=X_test.index)

    # drop cate col & merge Encoded Features
    X_train = X_train.drop(columns=categorical_cols).reset_index(drop=True)
    X_test = X_test.drop(columns=categorical_cols).reset_index(drop=True)

    X_train_encoded = X_train_encoded.reset_index(drop=True)
    X_test_encoded = X_test_encoded.reset_index(drop=True)

    X_train = pd.concat([X_train, X_train_encoded], axis=1)
    X_test = pd.concat([X_test, X_test_encoded], axis=1)


    # fit logistic reg model
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)

    # make predictions
    y_pred = model.predict(X_test)

    # eval model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.2f}")
    print(f"Model Recall: {sensitivity:.2f}")

    print("\nClassification Report:\n", report)

    return model, X_train, accuracy, sensitivity, report



def fit_logistic_regression_with_bootstrapping(df, target_col, test_size=0.3, n_bootstrap=10, random_state=42, max_iter=1000):
    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    encoder = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    encoder.fit(X_train[categorical_cols])

    # tranform train and test sets
    X_train_encoded = encoder.transform(X_train[categorical_cols])
    X_test_encoded = encoder.transform(X_test[categorical_cols])

    # con to df
    X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(), index=X_train.index)
    X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(), index=X_test.index)

    # drop categorical columns and merge encoded features
    X_train = X_train.drop(columns=categorical_cols).reset_index(drop=True)
    X_test = X_test.drop(columns=categorical_cols).reset_index(drop=True)

    X_train_encoded = X_train_encoded.reset_index(drop=True)
    X_test_encoded = X_test_encoded.reset_index(drop=True)

    X_train = pd.concat([X_train, X_train_encoded], axis=1)
    X_test = pd.concat([X_test, X_test_encoded], axis=1)

    log_loss_scores = []

    for i in range(n_bootstrap):
        # bootstrap resampling (sampling with replacement)
        boot_indices = np.random.choice(X_train.index, size=len(X_train), replace=True)
        X_bootstrap = X_train.iloc[boot_indices].reset_index(drop=True)
        y_bootstrap = y_train.iloc[boot_indices].reset_index(drop=True)

        # fit logistic regression model
        model = LogisticRegression(max_iter=max_iter)
        model.fit(X_bootstrap, y_bootstrap)

        # predict probabilities on the validation set
        y_prob = model.predict_proba(X_test)[:, 1]

        # compute log-loss
        loss = log_loss(y_test, y_prob)
        log_loss_scores.append(loss)

        print(f"Bootstrap {i+1}: Log-Loss = {loss:.4f}")

    # calculate log-loss statistics
    mean_log_loss = np.mean(log_loss_scores)
    std_log_loss = np.std(log_loss_scores)

    print("\nBootstrap Log-Loss Summary:")
    print(f"Mean Log-Loss: {mean_log_loss:.4f}")
    print(f"Standard Deviation of Log-Loss: {std_log_loss:.4f}")

    return log_loss_scores, mean_log_loss, std_log_loss

def modeling(data):
    data = process(data)
    data_age_2 = data[data['AgeTwoPlus'] == 2]
    data_age_1 = data[data['AgeTwoPlus'] == 1]

    model_1 = fit_logistic_regression(data_age_1, 'PosIntFinal')

    model_2 = fit_logistic_regression(data_age_2, 'PosIntFinal')
    return model_1, model_2

#     # Retrieve coefficients
# coefficients = model_2[0].coef_[0]  # Logistic regression coefficients

# # Create DataFrame for better visualization
# coef_df = pd.DataFrame({'Feature': model_2[1].columns, 'Coefficient': coefficients})

# # Sort by absolute value of coefficient (most impactful features first)
# coef_df = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=False).index)

# # Display the coefficients
# print(coef_df)

if __name__ == "__main__":
    dir_path = "TBI PUD 10-08-2013.csv"  
    data = pd.read_csv(dir_path)  
    data = process(data)  
    model_1, model_2 = modeling(data)
