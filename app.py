from flask import Flask, request, render_template
from src.config import TARGET_COLUMN, mongo_client ,database_name
import pandas as pd
import dill
from src.utils import get_relevant_past_df, store_prediction_records_to_database , load_object
from src.predictor import ModelResolver
from src.feature_extractor import generate_features
import warnings
import os
warnings.filterwarnings('ignore')

app = Flask(__name__)

# -------------------------
# Load trained ML model
# -------------------------
resolver=ModelResolver()
model=load_object(file_path=resolver.get_latest_model_path())

# -------------------------
# Home route
# -------------------------
@app.route('/')
def home():
    return render_template("home.html")

# -------------------------
# Prediction route
# -------------------------
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Step 1: Get user input from form
            transaction_id = int(request.form['transaction_id'])
            customer_id = int(request.form['customer_id'])
            terminal_id = int(request.form['terminal_id'])
            amount = float(request.form['amount'])
            timestamp_str = pd.to_datetime(request.form['timestamp'])

            # Step 2: Query past transactions by customer or terminal
            query = {
                "$or": [
                    {"CUSTOMER_ID": customer_id},
                    {"TERMINAL_ID": terminal_id}
                ]
            }
            past_df = get_relevant_past_df(
                query=query,
                database_name=database_name,
                collection_name="transactions"
            )

            # Step 3: Create DataFrame for current transaction
            input_data = pd.DataFrame([{
                'TRANSACTION_ID': transaction_id,
                'CUSTOMER_ID': customer_id,
                'TERMINAL_ID': terminal_id,
                'TX_AMOUNT': amount,
                'TX_DATETIME': timestamp_str
            }])

            # Step 4: Generate features
            features = generate_features(current_df=input_data, past_df=past_df, mode="prediction")
            features_required = model.feature_names_in_
            final_features = features[features_required]

            # Step 5: Predict fraud
            prediction = model.predict(final_features)
            result = 'Fraud' if prediction[0] == 1 else 'Safe'

            # Step 6: Store the prediction result in DB
            final_features[TARGET_COLUMN] = result
            final_features_dict = final_features.to_dict(orient='records')[0]

            # Let's store the prediction input records in the database, including the original input data.
            if "TERMINAL_ID" not in final_features_dict:
                input_data_dict = input_data.to_dict(orient='records')[0]
                final_features_dict = final_features_dict | input_data_dict 

            store_prediction_records_to_database(
                mongo_client=mongo_client,
                database_name=database_name,
                collection_name="latest_transactions",
                data=final_features_dict
            )

            # Step 7: Render result page
            return render_template(
                'result.html',
                final_features=final_features.to_dict(orient='records')[0],
                result=result
            )

        except Exception as e:
            return f"<h2 style='color:red; text-align:center;'>Error: {str(e)}</h2>"

    # If GET request, show form
    return render_template('predict.html')

# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    app.run(debug=True, port=8501)
