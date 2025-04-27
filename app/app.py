import gradio as gr
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import numpy as np
from model_training import prepare_data, train_models
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

class DrugPredictorApp:
    def __init__(self, data_path="data/drug_consumption_combined.csv"):
        # Prepare the data and train models
        self.df, self.features, self.drug_targets = prepare_data(data_path)
        self.results = train_models(self.df, self.features, self.drug_targets)
        self.metrics_df = self._prepare_metrics_df()
        
    def _prepare_metrics_df(self):
        """Calculate and format metrics for all trained models"""
        metrics = []
        for res in self.results:
            model = res['model']
            X_test = res['X_test']
            y_test = res['y_test']
            
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            metrics.append({
                'Drug': res['Drug'],
                'ROC-AUC': roc_auc_score(y_test, y_proba),
                'F1': f1_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'Minority Samples': y_test.sum()
            })
        
        # Sort metrics by ROC-AUC descending
        return pd.DataFrame(metrics).sort_values('ROC-AUC', ascending=False)
    
    def predict(self, drug, features_to_show, show_metrics):
        """Generate feature importance plot and optionally show performance metrics"""
        try:
            res = next(r for r in self.results if r['Drug'] == drug)
            model = res['model']
            
            importances = pd.Series(
                model.best_estimator_.named_steps['model'].feature_importances_,
                index=self.features
            )[features_to_show].sort_values()
            
            # Create feature importance plot
            fig, ax = plt.subplots(figsize=(10, 6))
            importances.plot.barh(color='skyblue', edgecolor='navy', ax=ax)
            ax.set_title(f"Feature Importance for {drug}", pad=20)
            ax.set_xlabel("Importance Score")
            ax.xaxis.set_major_formatter(PercentFormatter(1.0))
            plt.tight_layout()
            
            # Prepare metrics text
            metrics = ""
            if show_metrics:
                stats = self.metrics_df[self.metrics_df['Drug'] == drug].iloc[0]
                metrics = f"""
                **Performance Metrics:**
                - ROC-AUC: {stats['ROC-AUC']:.3f}
                - F1 Score: {stats['F1']:.3f}
                - Precision: {stats['Precision']:.3f}
                - Recall: {stats['Recall']:.3f}
                - Minority Samples: {stats['Minority Samples']}
                """
            
            return fig, metrics
        
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def create_app(self):
        """Build the Gradio interface"""
        with gr.Blocks(title="Drug Use Predictor", theme=gr.themes.Soft()) as app:
            gr.Markdown("# 🧪 Drug Use Predictor Explorer")
            
            with gr.Tab("Feature Analysis"):
                with gr.Row():
                    with gr.Column():
                        drug_dropdown = gr.Dropdown(
                            choices=list(self.metrics_df['Drug']),
                            label="Select Drug",
                            value=self.metrics_df.iloc[0]['Drug']
                        )
                        feature_checkboxes = gr.CheckboxGroup(
                            choices=self.features,
                            label="Select Features",
                            value=self.features[:5]  # preselect top 5 features
                        )
                        metrics_toggle = gr.Checkbox(value=True, label="Show Metrics")
                        submit_btn = gr.Button("Analyze", variant="primary")
                    
                    with gr.Column():
                        plot_output = gr.Plot()
                        metrics_output = gr.Markdown()
            
            with gr.Tab("Model Comparison"):
                gr.Markdown("### 🏆 Top 5 Models Performance")
                
                # Only show the Top 5 models
                top5_metrics = self.metrics_df.nlargest(5, 'ROC-AUC')
                
                gr.Dataframe(top5_metrics)
                
                with gr.Row():
                    gr.ScatterPlot(
                        top5_metrics,
                        x="ROC-AUC",
                        y="F1",
                        color="Drug",
                        title="Top 5 Models - ROC-AUC vs F1 Score"
                    )
                    gr.BarPlot(
                        top5_metrics,
                        x="Drug",
                        y="ROC-AUC",
                        title="Top 5 Models by ROC-AUC"
                    )
            
            submit_btn.click(
                self.predict,
                inputs=[drug_dropdown, feature_checkboxes, metrics_toggle],
                outputs=[plot_output, metrics_output]
            )
        
        return app

# ========== Entry Point ==========
if __name__ == "__main__":
    app = DrugPredictorApp().create_app()
    app.launch()
