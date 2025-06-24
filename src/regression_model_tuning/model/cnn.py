import torch
import torch.nn as nn

import ast
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score     
import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    def build_model(self, input_dim, n_layers=3, n_units=128, activation='relu', dropout_rate=0.3):
        """신경망 모델을 생성하는 메서드
        
        Args:
            input_dim (int): 입력 차원 수
            n_layers (int): 은닉층 개수 (default: 3)
            n_units (int|list): 각 층의 유닛 수 (default: 128)
            activation (str|list): 활성화 함수 (default: 'relu')
            dropout_rate (float): 드롭아웃 비율 (default: 0.3)
            
        Returns:
            nn.Module: 생성된 신경망 모델
        """
        def _get_layer_configs():
            return (
                [n_units] * n_layers if not isinstance(n_units, list) else n_units,
                [activation] * n_layers if not isinstance(activation, list) else activation
            )
            
        def _create_layer_block(in_dim, out_dim, act_name, dropout):
            act_fn = {
                'relu': nn.ReLU,
                'tanh': nn.Tanh, 
                'leaky_relu': nn.LeakyReLU
            }.get(act_name, nn.ReLU)()
            
            return [
                nn.Linear(in_dim, out_dim),
                act_fn,
                nn.BatchNorm1d(out_dim),
                nn.Dropout(dropout)
            ]

        units, activations = _get_layer_configs()
        layers = []
        in_features = input_dim
        
        # 은닉층 생성
        for i in range(n_layers):
            layers.extend(_create_layer_block(
                in_features, units[i], 
                activations[i], dropout_rate
            ))
            in_features = units[i]
            
        # 출력층 추가
        layers.append(nn.Linear(in_features, 1))
        
        # 모델 생성 및 디바이스 설정
        model = nn.Sequential(*layers).to(self.device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            
        self.model = model
        return model

    def forward(self, x):
        return self.model(x)

    def convert_data_to_tensor(self, X, y, view_y=True):
        """데이터를 PyTorch tensor로 변환하는 메서드
        
        Args:
            X: 입력 데이터 (DataFrame, Series, ndarray)
            y: 출력 데이터 (DataFrame, Series, ndarray) 
            device: 텐서를 할당할 디바이스 (default: None)
            view_y: y텐서를 2차원으로 변환할지 여부 (default: True)
            
        Returns:
            tuple: (X_tensor, y_tensor) 변환된 텐서들
        """
        X_tensor = self._convert_to_tensor(self._to_numpy(X))
        y_tensor = self._convert_to_tensor(self._process_target(y), view=view_y)
        
        if self.device:
            X_tensor = X_tensor.to(self.device)
            y_tensor = y_tensor.to(self.device)

        self.X_tensor = X_tensor
        self.y_tensor = y_tensor
            
        return X_tensor, y_tensor
    
    
    def _to_numpy(self, data):
        """데이터를 numpy array로 변환"""
        return data.values if isinstance(data, (pd.DataFrame, pd.Series)) else data
    
    def _convert_to_tensor(self, data, view=False):
        """numpy array를 tensor로 변환"""
        tensor = torch.tensor(data, dtype=torch.float32)
        return tensor.view(-1, 1) if view else tensor
        
    def _process_target(self, y):
        """타겟 데이터 전처리"""
        y = self._to_numpy(y)
        
        if y.dtype == np.dtype('O') or y.dtype.kind in 'SU':
            return np.array([self._process_item(item) for item in y])
        return y.astype(np.float32)
    
    def _process_item(self, item):
        """단일 타겟 값 처리"""
        if isinstance(item, bytes):
            item = item.decode()
            
        if isinstance(item, list):
            return float(item[0])
            
        if isinstance(item, str):
            try:
                parsed = ast.literal_eval(item)
                if isinstance(parsed, list):
                    return float(parsed[0])
                return float(parsed)
            except:
                try:
                    return float(item)
                except Exception as e:
                    raise ValueError(f"Cannot convert item: {item}") from e
                    
        try:
            return float(item)
        except Exception as e:
            raise ValueError(f"Cannot convert item: {item}") from e
        


    def train_one_epoch(self, model, optimizer, train_loader, device, criterion):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)
    

    def validate_epoch(self, model, X_val_tensor, y_val_tensor, device, criterion):
        model.eval()
        with torch.no_grad():
            outputs = model(X_val_tensor)
            loss = criterion(outputs, y_val_tensor).item()
        return loss


    def evaluate_split(self, model, X, y):
        X_tensor, _ = self.convert_data_to_tensor(X, y, view_y=True)
        model.eval()
        with torch.no_grad():
            y_pred = model(X_tensor).detach().cpu().numpy()
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_true = y.values.flatten()
        else:
            y_true = np.array(y).flatten()
        mae = nn.L1Loss()(torch.tensor(y_true, dtype=torch.float32).view(-1,1),
                          torch.tensor(y_pred, dtype=torch.float32)).item()
        mape = mean_absolute_percentage_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return {"MAE": mae, "MAPE": mape, "MSE": mse, "RMSE": rmse, "R2": r2}, y_pred



    def plot_scatter(self, model, output_col, X_data, y_data, save_path=None, metrics=None):
        """
        output_col: 출력 변수 이름 (모델 딕셔너리 키)
        X_data, y_data: 실제 데이터 (예측에 사용)
        save_path: 결과 이미지 저장 경로 (선택)
        metrics: 미리 계산된 평가 지표 딕셔너리, 예:
                 {"R2": 0.95, "MAE": 0.1, "MSE": 0.02, "RMSE": 0.14, "MAPE": 0.05}
                 만약 None이면 내부에서 sklearn을 사용해 재계산함.
        """
        import seaborn as sns

        if model is None:
            print(f"No trained model found.")
            return

        # 데이터 변환
        X_tensor, _ = self.convert_data_to_tensor(X_data, y_data, view_y=False)
        with torch.no_grad():
            y_pred_tensor = model(X_tensor)
        y_pred = y_pred_tensor.cpu().numpy().flatten()
        if isinstance(y_data, (pd.Series, pd.DataFrame)):
            y_true = y_data.values.flatten()
        else:
            y_true = np.array(y_data).flatten()
        
        # metrics가 제공되지 않으면 내부에서 계산
        if metrics is None:
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
            r2 = r2_score(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(y_true, y_pred)
            metrics = {
                "R2": r2,
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "MAPE": mape
            }
        
        # 플롯 스타일 설정
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 8))
        
        # 산점도
        plt.scatter(y_true, y_pred, alpha=0.6, edgecolor='k', label="Data points")
        
        # 1:1 이상적인 선 (대각선)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Ideal Fit")
        
        # 회귀선 추가: 1차 선형 회귀선 계산
        slope, intercept = np.polyfit(y_true, y_pred, 1)
        reg_line = slope * np.array([min_val, max_val]) + intercept
        plt.plot([min_val, max_val], reg_line, 'b-', lw=2, label="Regression Line")
        
        # 축, 제목, 범례 설정
        plt.xlabel("Actual Values", fontsize=12)
        plt.ylabel("Predicted Values", fontsize=12)
        plt.title(f"Scatter Plot for {output_col}", fontsize=14)
        plt.legend(fontsize=10)
        
        # 평가 지표 텍스트 박스 추가
        metrics_text = "\n".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
        plt.gca().text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Scatter plot saved to {save_path}")
        else :
            plt.show()
        plt.close() 