from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import io

app = Flask(__name__)

# Load the model
model = joblib.load("cardiacarrest_predict_model.pkl")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    # Check if the file is not empty
    if file.filename == '':
        return render_template('index.html', error='No selected file')

    # Check if the file is a CSV
    if file and file.filename.endswith('.csv'):
        # Read the CSV file into a DataFrame
        input_data = pd.read_csv(file)

        # Make predictions
        predictions = model.predict(input_data)

        # # Assuming a binary classification (0 or 1), you can convert the results to a human-readable format
        # results = ["At Risk" if pred ==
        #            1 else "Not at risk" for pred in predictions]

        # Create a new column 'Prediction Result' in the DataFrame
        input_data['Prediction Result'] = ["At Risk" if pred ==
                                           1 else "Not at risk" for pred in predictions]

        # Convert the DataFrame to HTML for rendering
        results_table = input_data.to_html(
            classes='table table-striped', index=False, escape=False)

        return render_template('results.html', results_table=results_table)

    else:
        return render_template('index.html', error='Invalid file format. Please upload a CSV file.')


if __name__ == '__main__':
    app.run(debug=True)
