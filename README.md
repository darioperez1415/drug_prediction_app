#Read Me

🧪 Drug Use Predictor Explorer
An interactive machine learning application predicting drug use likelihood based on demographic and psychological traits.
Built using Random Forest models and visualized through an interactive Gradio app.

🚀 Features
Predict likelihood of drug usage across 18 substances.

Explore feature importance dynamically through interactive plots.

Visualize model performance (ROC-AUC vs F1 Score).

Focus analysis on Top 5 performing models.

Organized, modular project structure for easy extension.

📚 Dataset
Demographics: age, gender, country, ethnicity

Psychological Traits: Big Five (nscore, escore, oscore, ascore, cscore), impulsiveness, sensation seeking

Drug Usage: Self-reported behavior, binary classified as user (1) or non-user (0).

🛠️ Machine Learning Model
Algorithm: Random Forest Classifier

Balancing: SMOTE applied to handle class imbalance

Evaluation Metrics:

ROC-AUC Score

F1 Score

Precision

Recall

📊 Top 5 Models (Performance Summary)

Drug	ROC-AUC	F1 Score	Precision	Recall
Legal Highs	0.876	0.748	0.717	0.783
Cannabis	0.860	0.840	0.876	0.806
LSD	0.857	0.700	0.644	0.766
Methadone	0.820	0.562	0.455	0.735
Mushrooms	0.810	0.665	0.617	0.719
