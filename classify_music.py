import librosa
import numpy as np
import json
from keras.models import load_model
import joblib
from xgboost import XGBClassifier
import os
import time
import timeit

def extract_features(file_path):
    try:
        audio, _ = librosa.load(file_path, sr=22050, mono=True)
        mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio, sr=22050)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=22050)
        tonnetz = librosa.feature.tonnetz(y=audio, sr=22050)
        features = np.vstack([mfccs, chroma, spectral_contrast, tonnetz])
        return np.mean(features.T, axis=0)
    except Exception as e:
        print(f"Error encountered while parsing file '{file_path}': {e}")
        return None
    
def classify_music(file_path):
    # saved_folder = 'saved_folder'
    saved_folder = 'files_w_noise'
    

    # Load data and models from the saved folder
    y_test = np.load(os.path.join(saved_folder, 'y_test.npy'))
    cnn_model = load_model(os.path.join(saved_folder, 'cnn_model.h5'))
    baseline_cnn_model = load_model(os.path.join(saved_folder, 'baseline_cnn_model.h5'))
    
    rf_model = joblib.load(os.path.join(saved_folder, 'rf_model.joblib'))
    svm_model = joblib.load(os.path.join(saved_folder, 'svm_model.joblib'))
    xgb_model = joblib.load(os.path.join(saved_folder, 'xgb_model.joblib'))
    ensemble_input = np.load(os.path.join(saved_folder, 'ensemble_input.npy'))

    # Load label dictionary
    with open(os.path.join(saved_folder, 'label_dict.json'), 'r') as json_file:
        label_dict = json.load(json_file)

    sample_features = extract_features(file_path)

    # Reshape the feature vector for compatibility with the CNN model
    sample_features_cnn = sample_features.reshape(1, sample_features.shape[0], 1)
    
    start_time_cnn = timeit.default_timer()
    # Use the CNN model to predict the genre probabilities
    cnn_prediction_prob = baseline_cnn_model.predict(sample_features_cnn)[0]
    end_time_cnn = timeit.default_timer()
    cnn_prediction_time = end_time_cnn - start_time_cnn

    print(f"CNN Prediction Time: {cnn_prediction_time} seconds")
    
    # Map the numerical labels to genre names
    genre_names = {idx: genre for genre, idx in label_dict.items()}
    # Create a dictionary to store predicted percentages for each class
    predicted_percentages = {}
    # Loop through each class and store the predicted percentage
    for idx, genre_prob in enumerate(cnn_prediction_prob):
        genre_name = genre_names[idx]
        predicted_percentages[genre_name] = float(genre_prob) * 100
    # Get the predicted genre with the highest probability
    predicted_genre = max(predicted_percentages, key=predicted_percentages.get)
    # Create a dictionary for the JSON result
    json_result_cnn = {
        "Predicted Genre (CNN)": predicted_genre,
        "Predicted Percentages (CNN)": predicted_percentages
    }
    cnn_features_sample = cnn_model.predict(sample_features_cnn)
    svm_prediction_sample = svm_model.predict(cnn_features_sample)
    rf_prediction_sample = rf_model.predict(cnn_features_sample)
    start_time_xgb = timeit.default_timer()
    ensemble_input_sample = np.column_stack((svm_prediction_sample, rf_prediction_sample, np.argmax(cnn_prediction_prob)))
    end_time_xgb = timeit.default_timer()
    xgb_prediction_time = end_time_xgb - start_time_xgb
    print(f"XGBoost Prediction Time: {xgb_prediction_time} seconds")

    xgb_model_ensemble = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
    xgb_model_ensemble.fit(ensemble_input, y_test)
    ensemble_prediction_sample = xgb_model_ensemble.predict(ensemble_input_sample)
    predicted_genre_ensemble = genre_names[ensemble_prediction_sample[0]]
    predicted_probabilities_ensemble = xgb_model_ensemble.predict_proba(ensemble_input_sample)[0]
    predicted_percentages_ensemble = {}
    for genre, percentage in zip(genre_names.values(), predicted_probabilities_ensemble):
        predicted_percentages_ensemble[genre] = float(percentage) * 100
    sorted_cnn_predictions = sorted(predicted_percentages.items(), key=lambda x: x[1], reverse=True)
    sorted_cnn_predictions = {genre: percentage for genre, percentage in sorted_cnn_predictions}
    sorted_ensemble_predictions = sorted(predicted_percentages_ensemble.items(), key=lambda x: x[1], reverse=True)
    sorted_ensemble_predictions = {genre: percentage for genre, percentage in sorted_ensemble_predictions}
    # Create sorted JSON result for CNN model
    if predicted_genre != "noise":
        sorted_json_result_cnn = {
            "Predicted Genre (CNN)": predicted_genre,
            "Predicted Percentages (CNN)": sorted_cnn_predictions
        }
    else:
        sorted_json_result_cnn = {
            "Predicted Genre (CNN)": predicted_genre
        }
    # Create sorted JSON result for ensemble model
    if predicted_genre_ensemble != "noise":
        sorted_json_result_ensemble = {
            "Predicted Genre (OCNN)": predicted_genre_ensemble,
            "Predicted Percentages (OCNN)": sorted_ensemble_predictions
        }
    else:
        sorted_json_result_ensemble = {
            "Predicted Genre (OCNN)": predicted_genre_ensemble
        }
    # Merge sorted JSON results
    merged_json_result = {
        "CNN": {
            **sorted_json_result_cnn,
            "Prediction Time": cnn_prediction_time
        },
        "OCNN": {
            **sorted_json_result_ensemble,
            "Prediction Time": xgb_prediction_time
        }
    }
    return merged_json_result