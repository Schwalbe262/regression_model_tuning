from regression_model_tuning.data.data import Data
from regression_model_tuning.model.cnn import CNN

class Trainer:
    def __init__(self, data_path, input_cols, output_cols, model_type="cnn", **model_kwargs):
        
        self.data = Data(data_path, input_cols, output_cols)

        if model_type == "cnn":
            self.model = CNN()
        # 다른 모델도 필요시 추가

    def train(self):
        # 학습 코드 (예시)
        print("학습 시작!")
        # X, y 준비, optimizer, loss 등 세팅 후 학습 루프 작성