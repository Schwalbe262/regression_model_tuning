import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd


class Pre_processing:

    def __init__(self):
        self.a = 1

    def drop_na(self, data, input_cols, output_cols):
        print(f"drop_na 전 데이터 수: {len(data)}")

        data = data.dropna(subset=input_cols + output_cols)

        print(f"drop_na 후 데이터 수: {len(data)}")
        
        return data

    def normalize_data(self, data, input_cols, path, scaler_name):
        input_data = data[input_cols]

        scaler = StandardScaler()
        X = scaler.fit_transform(input_data)
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/{scaler_name}.pkl", "wb") as f:
            pickle.dump(scaler, f)

        return X, scaler

    def outlier_filtering(self, data, input_cols, output_cols, method="IQR", **kwargs):
        input_data = data[input_cols]
        output_data = data[output_cols]
        data = pd.concat([input_data, output_data], axis=1)

        print(f"outlier_filtering 전 데이터 수: {len(data)}")

        if method == "IQR":
            outlier_constant = kwargs.get("outlier_constant", 1.5)
            Q1 = data[input_cols].quantile(0.25)
            Q3 = data[input_cols].quantile(0.75)
            IQR = Q3 - Q1
            # 하나라도 이상치면 True
            mask = ((data[input_cols] < (Q1 - outlier_constant * IQR)) | (data[input_cols] > (Q3 + outlier_constant * IQR))).any(axis=1)
            # 이상치가 아닌 데이터만 선택 (mask가 False인 데이터)
            data = data[~mask]

        print(f"outlier_filtering 후 데이터 수: {len(data)}")
        
        return data

    def split_data(self, data, input_cols, output_cols):
        X = data[input_cols]
        Y = data[output_cols]
        return X, Y
    

    def split_train_val_test(self, X, Y, test_size=0.2, val_size=0.2, random_state=42):
        X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
        relative_val_size = val_size / (1 - test_size)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=relative_val_size, random_state=random_state)
        return X_train, Y_train, X_val, Y_val, X_test, Y_test
