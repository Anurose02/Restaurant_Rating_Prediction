from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load your trained model and columns (adjust path)
model = joblib.load('restaurant_rating_model.joblib')
X_columns = joblib.load('X_columns.joblib')

# Extract unique values for dropdowns
locations = sorted([col.replace('location_', '') for col in X_columns if col.startswith('location_')])
rest_types = sorted([col.replace('rest_type_', '') for col in X_columns if col.startswith('rest_type_')])
listed_in_types = sorted([col.replace('listed_in(type)_', '') for col in X_columns if col.startswith('listed_in(type)_')])
listed_in_cities = sorted([col.replace('listed_in(city)_', '') for col in X_columns if col.startswith('listed_in(city)_')])
primary_cuisines = sorted([col.replace('primary_cuisine_', '') for col in X_columns if col.startswith('primary_cuisine_')])

@app.route('/')
def home():
    # This route passes all your lists to the HTML template.
    return render_template('index.html', 
                           locations=locations,
                           rest_types=rest_types,
                           listed_in_types=listed_in_types,
                           listed_in_cities=listed_in_cities,
                           primary_cuisines=primary_cuisines)

@app.route('/predict', methods=['POST'])
def predict():
    # --- THIS IS THE DEBUGGING LINE ---
    # Check your terminal to see this output when you submit the form
    print(f"Form data received: {request.form}")
    # --- END DEBUGGING LINE ---
    
    try:
        # Get form data
        # Get the '0' or '1' from the form and convert to integer
        online_order = int(request.form.get('online_order'))
        book_table = int(request.form.get('book_table'))
        
        # --- ROBUSTNESS FIX ---
        # Check for None or empty string before converting to float
        cost_str = request.form.get('approx_cost')
        if not cost_str: # Catches None or ''
             return jsonify({
                'success': False,
                'error': "Cost for Two must be a valid number."
            }), 400
        
        # Now it's safe to convert
        cost_for_two = int(cost_str)
        # --- END FIX ---

        location = request.form.get('location')
        rest_type = request.form.get('rest_type')
        listed_in_type = request.form.get('listed_in_type')
        listed_in_city = request.form.get('listed_in_city')
        primary_cuisine = request.form.get('primary_cuisine')
        
        # Prepare input data
        input_data = pd.DataFrame(np.zeros((1, len(X_columns))), columns=X_columns)
        
        # Set numeric/binary features directly
        input_data['online_order'] = online_order
        input_data['book_table'] = book_table
        
        # --- COLUMN NAME FIX ---
        # Match the column name from X_columns.joblib
        input_data['cost_for_two'] = cost_for_two
        
        # Function to set one-hot encoded columns
        def set_one_hot(col_prefix, value):
            col_name = f"{col_prefix}_{value}"
            if col_name in input_data.columns:
                input_data.at[0, col_name] = 1
        
        set_one_hot('location', location)
        set_one_hot('rest_type', rest_type)
        set_one_hot('listed_in(type)', listed_in_type)
        set_one_hot('listed_in(city)', listed_in_city)
        set_one_hot('primary_cuisine', primary_cuisine)
        
        # Predict
        pred_rating = model.predict(input_data)[0]
        
        return jsonify({
            'success': True,
            'rating': round(pred_rating, 2) # Send the real rating
        })
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True)