import pandas as pd
import gradio as gr
import joblib
import shap
import xgboost as xgb

selected_features =  ['AFP', 'AG', 'AST', 'CA72-4', 'CEA', 'CO2CP', 'CREA', 'GLO','HE4','HGB']
model = joblib.load('xg.pkl')
model.get_booster().save_model("model.json")

try:
    model = xgb.Booster()
    model.load_model("model.json")
except Exception as e:
    print(f"Error loading model: {e}")
    traceback.print_exc()

def infer(input_dataframe):
    # Select the relevant features
    input_dataframe = input_dataframe[selected_features]
    dmatrix = xgb.DMatrix(input_dataframe)

    # Make predictions
    predictions = model.predict(dmatrix)

    # Calculate SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer(input_dataframe)

    # Convert SHAP values to a DataFrame
    shap_df = pd.DataFrame(shap_values.values, columns=selected_features)

    # Combine predictions and SHAP values
    results = pd.concat([pd.DataFrame(predictions, columns=['Prediction']), shap_df], axis=1)

    return results

inputs = gr.File(label="Upload Excel file with patient data")
def process_file(file):
    df = pd.read_excel(file.name)
    return infer(df)

outputs = gr.Dataframe(row_count=(2, "dynamic"), col_count=(len(selected_features) + 1, "dynamic"), label="Predictions and SHAP Values")

model_gradio = gr.Interface(fn=process_file, inputs=inputs, outputs=outputs).launch(debug=True)