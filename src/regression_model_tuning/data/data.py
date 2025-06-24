from .pre_processing import Pre_processing

import pandas as pd

class Data():
    def __init__(self, path, input_cols=None, output_cols=None):
        self.path = path
        self.raw_data = self.load_data(path=path)
        self.data = self.raw_data
        if input_cols is not None:
            self.set_input_cols(input_cols)
        if output_cols is not None:
            self.set_output_cols(output_cols)
        self.preprocessor = Pre_processing()

    def set_input_cols(self, input_cols):
        self.input_cols = input_cols
        self.input_dim = len(input_cols)

    def set_output_cols(self, output_cols):
        self.output_cols = output_cols

    def load_data(self, path):
        raw_data = pd.read_csv(path)
        print("Data loaded successfully!")
        print(f"Data Shape: {raw_data.shape}")
        print(f"Columns: {raw_data.columns.tolist()}")
        self.raw_data = raw_data
        return raw_data



    def _check_default_params(self, data=None, input_cols=None, output_cols=None):
        if data is None:
            data = self.data
        if input_cols is None:
            input_cols = self.input_cols
        if output_cols is None:
            output_cols = self.output_cols
        return data, input_cols, output_cols

    def drop_na(self, data=None, input_cols=None, output_cols=None):
        data, input_cols, output_cols = self._check_default_params(data, input_cols, output_cols)
        self.data = self.preprocessor.drop_na(data, input_cols, output_cols)
        return self.data

    def outlier_filtering(self, data=None, input_cols=None, output_cols=None, method="IQR", **kwargs):
        data, input_cols, output_cols = self._check_default_params(data, input_cols, output_cols)
        self.data = self.preprocessor.outlier_filtering(data, input_cols, output_cols, method, **kwargs)
        return self.data

    def split_data(self, data=None, input_cols=None, output_cols=None):
        data, input_cols, output_cols = self._check_default_params(data, input_cols, output_cols)
        self.X, self.Y = self.preprocessor.split_data(data, input_cols, output_cols)
        return self.X, self.Y

    def normalize_data(self, data=None, path=None, scaler_name=None):
        data, input_cols, output_cols = self._check_default_params(data)
        self.X, self.scaler = self.preprocessor.normalize_data(data, self.input_cols, path, scaler_name)
        return self.X, self.scaler
    
    def split_train_val_test(self, test_size=0.2, val_size=0.2, random_state=42):
        self.train_X, self.train_Y, self.val_X, self.val_Y, self.test_X, self.test_Y = self.preprocessor.split_train_val_test(self.X, self.Y, test_size, val_size, random_state)
        return self.train_X, self.train_Y, self.val_X, self.val_Y, self.test_X, self.test_Y

