from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('SLR.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        experience = float(request.form.get('experience'))
        
        # Reshape for scikit-learn (expects 2D array)
        input_data = np.array([[experience]])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Format the result (assuming it's a dollar amount)
        output = round(prediction[0], 2)

        return render_template('index.html', 
                               prediction_text=f'Estimated Result: ${output}',
                               original_input=experience)
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)