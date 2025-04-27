from DrugPredictorApp import DrugPredictorApp
from model_training import prepare_data, train_models


if __name__ == "__main__":
    demo = DrugPredictorApp().create_app()
    demo.launch()
