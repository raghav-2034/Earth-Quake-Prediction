#!/usr/bin/env python
# coding: utf-8

from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
import pickle
import os
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

app = Flask(__name__)
app.secret_key = 'earthquake_analysis_secret_key'  # Required for session

# Configure file uploads
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx', 'xls'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def load_dataset(filepath=None):
    """Load the earthquake dataset from file"""
    if filepath is None:
        filepath = '/Users/raghav/Desktop/EarthQuakecode /Eartquakes-1990-2023.csv'  # Note the space after EarthQuakecode
    
    # Check file extension and load accordingly
    if filepath.endswith('.csv'):
        try:
            # First try with standard delimiter
            df = pd.read_csv(filepath)
        except:
            # Try with whitespace delimiter if standard delimiter fails
            try:
                df = pd.read_csv(filepath, delim_whitespace=True)
            except:
                # Try with semicolon delimiter
                df = pd.read_csv(filepath, sep=';')
    else:
        df = pd.read_excel(filepath)
    
    # Print the columns for analysis
    print("Original columns:", df.columns.tolist())
    
    # Map common column names to our expected format
    column_mapping = {
        # Common latitude variations
        'lat': 'Latitude(deg)',
        'latitude': 'Latitude(deg)',
        'latitude_degrees': 'Latitude(deg)',
        
        # Common longitude variations
        'lon': 'Longitude(deg)',
        'long': 'Longitude(deg)',
        'longitude': 'Longitude(deg)',
        'longitude_degrees': 'Longitude(deg)',
        
        # Common depth variations
        'depth': 'Depth(km)',
        'depth_km': 'Depth(km)',
        
        # Common magnitude variations
        'mag': 'Magnitude',
        'magnitude': 'Magnitude',
        'magnitude_value': 'Magnitude',
        
        # Common station variations
        'nst': 'No_of_Stations',
        'stations': 'No_of_Stations',
        'station_count': 'No_of_Stations',
    }
    
    # Apply column mapping (case insensitive)
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in column_mapping and column_mapping[col_lower] not in df.columns:
            df.rename(columns={col: column_mapping[col_lower]}, inplace=True)
    
    # Ensure essential columns exist
    essential_columns = ['Latitude(deg)', 'Longitude(deg)', 'Depth(km)', 'Magnitude']
    missing_columns = [col for col in essential_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing essential columns: {missing_columns}")
        # Try to derive missing columns from available data
        for col in missing_columns:
            if col == 'Latitude(deg)':
                lat_candidates = [c for c in df.columns if 'lat' in c.lower()]
                if lat_candidates:
                    df['Latitude(deg)'] = df[lat_candidates[0]]
            elif col == 'Longitude(deg)':
                lon_candidates = [c for c in df.columns if 'lon' in c.lower() or 'long' in c.lower()]
                if lon_candidates:
                    df['Longitude(deg)'] = df[lon_candidates[0]]
            elif col == 'Depth(km)':
                depth_candidates = [c for c in df.columns if 'depth' in c.lower()]
                if depth_candidates:
                    df['Depth(km)'] = df[depth_candidates[0]]
            elif col == 'Magnitude':
                mag_candidates = [c for c in df.columns if 'mag' in c.lower()]
                if mag_candidates:
                    df['Magnitude'] = df[mag_candidates[0]]
    
    # Check for No_of_Stations column
    if 'No_of_Stations' not in df.columns:
        station_candidates = [c for c in df.columns if 'station' in c.lower() or 'nst' in c.lower()]
        if station_candidates:
            df['No_of_Stations'] = df[station_candidates[0]]
        else:
            # Create default value if no station column exists
            df['No_of_Stations'] = 1
    
    # Create Magnitude_Category if it doesn't exist
    if 'Magnitude_Category' not in df.columns and 'Magnitude' in df.columns:
        df['Magnitude_Category'] = pd.cut(
            df['Magnitude'], 
            bins=[0, 5, 6, 7, np.inf], 
            labels=['Minor', 'Moderate', 'Strong', 'Major']
        )
    
    # Convert columns to appropriate data types
    try:
        df['Latitude(deg)'] = pd.to_numeric(df['Latitude(deg)'])
        df['Longitude(deg)'] = pd.to_numeric(df['Longitude(deg)'])
        df['Depth(km)'] = pd.to_numeric(df['Depth(km)'])
        df['Magnitude'] = pd.to_numeric(df['Magnitude'])
        df['No_of_Stations'] = pd.to_numeric(df['No_of_Stations'])
    except Exception as e:
        print(f"Error converting data types: {e}")
    
    # Print the final columns for verification
    print("Final columns:", df.columns.tolist())
    
    return df

# Load the default dataset
df = load_dataset()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Train models
def train_models(dataset=None):
    # Use provided dataset or default
    data = dataset if dataset is not None else df
    
    # Select relevant columns
    X = data[['Latitude(deg)', 'Longitude(deg)', 'Depth(km)', 'No_of_Stations']]
    y = data['Magnitude']  # Using the Magnitude column
    
    # Drop any rows with NaN values
    combined = pd.concat([X, y.to_frame()], axis=1)
    combined = combined.dropna()
    
    # Split back into X and y
    X = combined[['Latitude(deg)', 'Longitude(deg)', 'Depth(km)', 'No_of_Stations']]
    y = combined['Magnitude']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # SVM
    svm_model = SVR(kernel='rbf', C=1e3, gamma=0.1)
    # Use a subset for SVM training
    subset_size = min(500, len(X_train))
    svm_model.fit(X_train[:subset_size], y_train[:subset_size])
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Save models
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    with open(os.path.join(models_dir, 'linear_regression.pkl'), 'wb') as f:
        pickle.dump(lr_model, f)
    
    with open(os.path.join(models_dir, 'svm.pkl'), 'wb') as f:
        pickle.dump(svm_model, f)
        
    with open(os.path.join(models_dir, 'random_forest.pkl'), 'wb') as f:
        pickle.dump(rf_model, f)
    
    # Calculate and return metrics
    metrics = {}
    
    # Linear Regression metrics
    y_pred_lr = lr_model.predict(X_test)
    metrics['lr'] = {
        'r2': r2_score(y_test, y_pred_lr),
        'mse': mean_squared_error(y_test, y_pred_lr)
    }
    
    # SVM metrics
    y_pred_svm = svm_model.predict(X_test)
    metrics['svm'] = {
        'r2': r2_score(y_test, y_pred_svm),
        'mse': mean_squared_error(y_test, y_pred_svm)
    }
    
    # Random Forest metrics
    y_pred_rf = rf_model.predict(X_test)
    metrics['rf'] = {
        'r2': r2_score(y_test, y_pred_rf),
        'mse': mean_squared_error(y_test, y_pred_rf)
    }
    
    return metrics, lr_model, svm_model, rf_model

# Train models and get metrics
metrics, lr_model, svm_model, rf_model = train_models()

@app.route('/')
def index():
    return render_template('index.html', 
                          metrics=metrics, 
                          min_lat=df['Latitude(deg)'].min(),
                          max_lat=df['Latitude(deg)'].max(),
                          min_lon=df['Longitude(deg)'].min(),
                          max_lon=df['Longitude(deg)'].max(),
                          min_depth=df['Depth(km)'].min(),
                          max_depth=df['Depth(km)'].max(),
                          min_stations=df['No_of_Stations'].min(),
                          max_stations=df['No_of_Stations'].max())

@app.route('/data')
def get_data():
    data = []
    for _, row in df.iterrows():
        data.append({
            'latitude': row['Latitude(deg)'],
            'longitude': row['Longitude(deg)'],
            'depth': row['Depth(km)'],
            'magnitude': row['Magnitude'],
            'stations': row['No_of_Stations']
        })
    return jsonify(data)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Get input values
    latitude = float(data['latitude'])
    longitude = float(data['longitude'])
    depth = float(data['depth'])
    stations = int(data['stations'])
    model_name = data['model']
    
    # Check if we should use a custom dataset
    custom_dataset_path = session.get('custom_dataset_path')
    if custom_dataset_path and os.path.exists(custom_dataset_path):
        # Load and train models on custom dataset
        custom_df = load_dataset(custom_dataset_path)
        _, lr_model, svm_model, rf_model = train_models(custom_df)
    
    # Create input array
    input_data = np.array([[latitude, longitude, depth, stations]])
    
    # Make prediction based on selected model
    if model_name == 'linear_regression':
        prediction = lr_model.predict(input_data)[0]
    elif model_name == 'svm':
        prediction = svm_model.predict(input_data)[0]
    elif model_name == 'random_forest':
        prediction = rf_model.predict(input_data)[0]
    else:
        return jsonify({'error': 'Invalid model name'}), 400
    
    # Return prediction
    return jsonify({
        'prediction': float(prediction),
        'model': model_name,
        'magnitude_category': get_magnitude_category(float(prediction))
    })

def get_magnitude_category(magnitude):
    """Categorize earthquake magnitude"""
    if magnitude < 5:
        return 'Minor'
    elif magnitude < 6:
        return 'Moderate'
    elif magnitude < 7:
        return 'Strong'
    else:
        return 'Major'

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the new dataset
        try:
            # Load the custom dataset
            new_dataset = load_dataset(filepath)
            
            # Store path in session for later use
            session['custom_dataset_path'] = filepath
            
            # Train models on the new dataset
            new_metrics, _, _, _ = train_models(new_dataset)
            
            return jsonify({
                'success': True,
                'message': 'Dataset uploaded successfully',
                'columns': new_dataset.columns.tolist(),
                'rows': len(new_dataset),
                'metrics': new_metrics
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    app.run(debug=True)