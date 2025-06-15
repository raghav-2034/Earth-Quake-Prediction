# Earthquake Prediction Using Machine Learning

A comprehensive machine learning-based system for predicting earthquake magnitudes using historical seismic data. This project implements multiple supervised learning algorithms to analyze seismic patterns and provide accurate earthquake magnitude predictions.

## 🌟 Features

- Multiple ML Algorithms**: Linear Regression, Support Vector Machine (SVM), Random Forest, and Naive Bayes
- **Interactive Web Interface**: User-friendly Flask-based web application
- **Real-time Predictions**: Input seismic parameters and get instant magnitude predictions
- **Data Visualization**: Interactive charts and graphs for data analysis
- **Model Comparison**: Performance evaluation and comparison of different algorithms
- **Export Functionality**: Download predictions and visualizations in CSV/PNG formats
- **Responsive Design**: Works across desktop, tablet, and mobile devices

## 📊 Project Overview

Earthquakes are among the most catastrophic natural disasters, causing extensive loss of life and property damage. This project addresses the critical need for accurate earthquake prediction by leveraging machine learning techniques to analyze historical seismic data from the California region.

### Key Objectives

- Collect and preprocess historical earthquake data
- Apply multiple machine learning algorithms for magnitude prediction
- Evaluate and compare model performance using metrics like accuracy, MSE, and R²
- Build a user-friendly prediction system with visualization capabilities
- Support decision-makers with actionable insights for disaster management

## 🛠️ Technology Stack

### Backend
- **Python 3.x**: Core programming language
- **Flask**: Web framework for API and routing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib/Seaborn**: Data visualization

### Frontend
- **HTML5**: Structure and markup
- **CSS3**: Styling and responsive design
- **JavaScript**: Interactive functionality and client-side validation

### Machine Learning Models
- **Linear Regression**: Baseline model for magnitude prediction
- **Support Vector Machine (SVM)**: Non-linear pattern recognition
- **Random Forest**: Ensemble learning for robust predictions
- **Naive Bayes**: Probabilistic classification approach

## 📋 Requirements

### Software Requirements
- Python 3.7 or higher
- Flask
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Openpyxl (for Excel file handling)

### Hardware Requirements
- Minimum 8GB RAM
- Stable internet connection
- Modern web browser (Chrome, Firefox, Safari, Edge)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/raghav-2034/earthquake-prediction-ml.git
   cd earthquake-prediction-ml
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv earthquake_env
   source earthquake_env/bin/activate  # On Windows: earthquake_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the dataset**
   - Ensure `Earthquake_data_processed.xlsx` is in the project directory
   - The dataset should contain columns: Latitude(deg), Longitude(deg), Depth(km), No_of_Stations, Magnitude(ergs)

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   - Open your web browser and navigate to `http://localhost:5000`

## 📊 Dataset

The system uses historical earthquake data with the following features:
- **Latitude (deg)**: Geographical latitude coordinate
- **Longitude (deg)**: Geographical longitude coordinate  
- **Depth (km)**: Earthquake depth below surface
- **No_of_Stations**: Number of seismic monitoring stations
- **Magnitude (ergs)**: Target variable for prediction

## 🔧 Usage

### Web Interface

1. **Access the Dashboard**: Navigate to the main page to view system overview and model performance metrics

2. **View Data Analysis**: Explore interactive visualizations showing:
   - Depth vs Magnitude correlation
   - Magnitude distribution patterns
   - Seismic activity trends

3. **Make Predictions**: 
   - Enter seismic parameters (latitude, longitude, depth, stations)
   - Select preferred ML model
   - Get instant magnitude predictions

4. **Compare Models**: View performance metrics for all implemented algorithms

### API Endpoints

- `GET /`: Main dashboard
- `GET /data`: Retrieve earthquake dataset sample
- `POST /predict`: Make magnitude predictions

## 📈 Model Performance

Based on evaluation metrics, the models perform as follows:

| Model | R² Score | Mean Squared Error | Best Use Case |
|-------|----------|-------------------|---------------|
| **Random Forest** | 0.1420 | 0.1651 | **Recommended** - Best overall performance |
| Linear Regression | - | - | Baseline comparisons |
| SVM | - | - | Non-linear pattern detection |
| Naive Bayes | - | - | Probabilistic predictions |

**Random Forest** emerged as the optimal choice due to:
- Highest correlation with actual earthquake data
- Lowest prediction error
- Superior handling of non-linear seismic patterns
- Robustness to outliers

## 🧪 Testing

The system includes comprehensive testing:

- **Unit Testing**: Individual component validation
- **Integration Testing**: Module interaction verification
- **Functional Testing**: Feature requirement compliance
- **Performance Testing**: Speed and responsiveness evaluation
- **UI Testing**: User interface validation
- **Accuracy Testing**: Model prediction verification

## 🔮 Future Enhancements

- **Real-time Data Integration**: Connect with live seismic sensors
- **Advanced Deep Learning**: Implement CNN and RNN models
- **Geospatial Visualization**: Interactive maps and regional risk assessment
- **Cloud Deployment**: Scalable multi-user access
- **Mobile Application**: Real-time earthquake alerts
- **Damage Estimation**: Population impact analysis
- **International Datasets**: Expand beyond California region

## 👥 Contributors

- **P. Raghavendra** (227Z1A05D4)
- **M. Keerthi** (227Z1A0596)  
- **K. Raviteja Goud** (227Z1A0573)

**Project Guide**: Mr. T. Madhu, Associate Professor, Department of Computer Science and Engineering

**Institution**: Nalla Narasimha Reddy Education Society's Group of Institutions (NNRESGI)

## 📄 License

This project is developed as part of academic requirements for Bachelor of Technology in Computer Science and Engineering. Please refer to your institution's policies for usage and distribution guidelines.

## 🔗 References

### Research Papers
1. Pratik Shrote et al., "Earthquake Prediction Through Machine Learning Approach," IEEE ICTACS 2024
2. Rakesh Kumar et al., "Earthquake Prediction Using Machine Learning," IEEE ICAC3N 2023
3. Gaurav Singh Manral et al., "Prediction of Earthquake Using Machine Learning Algorithms," IEEE ICIEM 2023

### Datasets
- USGS Earthquake Database
- Statistics Online Computational Resource (SOCR)
- Kaggle Earthquake Datasets

