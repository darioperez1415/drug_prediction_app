from app.app import DrugPredictorApp

if __name__ == "__main__":
    app = DrugPredictorApp().create_app()
    app.launch()
