from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
import requests
import io

app = Flask(__name__)
CORS(app)

# Function to load CSV files from HTTP server
def load_csv_files():
    base_url = 'http://192.168.237.24/csv_files'
    response = requests.get(f'{base_url}/file_list.txt')
    file_names = response.text.splitlines()

    dataframes = []
    for file_name in file_names:
        url = f'{base_url}/{file_name}'
        response = requests.get(url)
        df = pd.read_csv(io.StringIO(response.text), names=header)
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

# Load the model and data
header = ['first_subject', 'second_subject', 'third_subject', 
          'first_performance', 'second_performance', 'third_performance', 
          'course_names']

combined_data = load_csv_files()

# Check for and handle NaN values
combined_data.dropna(inplace=True)

model = RandomForestClassifier(n_estimators=100, random_state=42)
X = combined_data.drop('course_names', axis=1)
y = combined_data['course_names']
X_encoded = pd.get_dummies(X)
model.fit(X_encoded, y)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    subjects = data['subjects']
    performances = data['performances']

    user_data = pd.DataFrame({'first_subject': [subjects[0]], 'second_subject': [subjects[1]], 'third_subject': [subjects[2]],
                              'first_performance': [performances[0]], 'second_performance': [performances[1]], 'third_performance': [performances[2]]})
    user_encoded = pd.get_dummies(user_data)
    user_encoded = user_encoded.reindex(columns=X_encoded.columns, fill_value=0)
    course_prediction = model.predict(user_encoded)

    matching_rows = combined_data[
        (combined_data['first_subject'] == subjects[0]) &
        (combined_data['second_subject'] == subjects[1]) &
        (combined_data['third_subject'] == subjects[2]) &
        (combined_data['first_performance'] == performances[0]) &
        (combined_data['second_performance'] == performances[1]) &
        (combined_data['third_performance'] == performances[2])
    ]

    if not matching_rows.empty:
        available_courses = []
        for _, row in matching_rows.iterrows():
            courses = row['course_names'].split(',')
            available_courses.extend([course.strip() for course in courses])
        return jsonify({'available_courses': available_courses, 'predicted_course': course_prediction[0]})
    else:
        return jsonify({'available_courses': [], 'predicted_course': course_prediction[0]})

@app.route('/random-course', methods=['POST'])
def random_course():
    data = request.json
    predicted_courses = data['predicted_courses']
    return jsonify({'liked_course': random.choice(predicted_courses)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
