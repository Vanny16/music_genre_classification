# import librosa
# import numpy as np
# import json
# from keras.models import load_model
# import joblib
# from xgboost import XGBClassifier
# import os

# def extract_features(file_path):
#     try:
#         # Load MP3 file
#         audio, _ = librosa.load(file_path, sr=22050, mono=True)  # Adjust duration as needed
#         mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=13)
#         chroma = librosa.feature.chroma_stft(y=audio, sr=22050)
#         spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=22050)
#         tonnetz = librosa.feature.tonnetz(y=audio, sr=22050)
#         features = np.vstack([mfccs, chroma, spectral_contrast, tonnetz])
#         mean_features = np.mean(features.T, axis=0)
#         return mean_features
#     except Exception as e:
#         print(f"Error encountered while parsing file '{file_path}': {e}")
#         return None
    
# def classify_music(file_path):
#     cnn_model = load_model('test_cnn_model.h5')

#     def extract_features(file_path):
#         try:
#             # Load MP3 file
#             audio, _ = librosa.load(file_path, sr=22050, mono=True)  # Adjust duration as needed
#             mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=13)
#             chroma = librosa.feature.chroma_stft(y=audio, sr=22050)
#             spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=22050)
#             tonnetz = librosa.feature.tonnetz(y=audio, sr=22050)
#             features = np.vstack([mfccs, chroma, spectral_contrast, tonnetz])
#             mean_features = np.mean(features.T, axis=0)
#             return mean_features
#         except Exception as e:
#             print(f"Error encountered while parsing file '{file_path}': {e}")
#             return None
        
#     saved_folder = 'saved_folder'

#     # Load data and models from the saved folder
#     y_test = np.load(os.path.join(saved_folder, 'y_test.npy'))
#     cnn_model = load_model(os.path.join(saved_folder, 'test_cnn_model.h5'))
#     rf_model = joblib.load(os.path.join(saved_folder, 'test_rf_model.joblib'))
#     ensemble_input = np.load(os.path.join(saved_folder, 'ensemble_input.npy'))
#     svm_model = joblib.load(os.path.join(saved_folder, 'test_svm_model.joblib'))
#     xgb_model = joblib.load(os.path.join(saved_folder, 'xgb_model.joblib'))
#     # Load label dictionary
#     with open(os.path.join(saved_folder, 'label_dict.json'), 'r') as json_file:
#         label_dict = json.load(json_file)
        
#         # sample_features = extract_features('..\\Data\\genres_original\\metal\\metal.00000.wav')
#         # sample_features = extract_features('countryroad.wav')
#         sample_features = extract_features('bark.wav')
#         # sample_features = extract_features('rock.wav')

#     sample_features_cnn = sample_features.reshape(1, sample_features.shape[0], 1)
#     # Use the CNN model to predict the genre probabilities
#     cnn_prediction_prob = cnn_model.predict(sample_features_cnn)[0]
#     # Map the numerical labels to genre names
#     genre_names = {idx: genre for genre, idx in label_dict.items()}
#     # Create a dictionary to store predicted percentages for each class
#     predicted_percentages = {}
#     # Loop through each class and store the predicted percentage
#     for idx, genre_prob in enumerate(cnn_prediction_prob):
#         genre_name = genre_names[idx]
#         predicted_percentages[genre_name] = float(genre_prob) * 100
#     # Get the predicted genre with the highest probability
#     predicted_genre = max(predicted_percentages, key=predicted_percentages.get)
#     # Create a dictionary for the JSON result
#     json_result_cnn = {
#         "Predicted Genre (CNN)": predicted_genre,
#         "Predicted Percentages (CNN)": predicted_percentages
#     }
#     cnn_features_sample = cnn_model.predict(sample_features_cnn)
#     # Make predictions using SVM and Random Forest models
#     svm_prediction_sample = svm_model.predict(cnn_features_sample)
#     rf_prediction_sample = rf_model.predict(cnn_features_sample)
#     # Ensemble predictions
#     ensemble_input_sample = np.column_stack((svm_prediction_sample, rf_prediction_sample, np.argmax(cnn_prediction_prob)))
#     # ensemble_input_sample = np.column_stack((svm_prediction_sample,svm_prediction_sample,rf_prediction_sample, np.argmax(cnn_prediction_prob)))

#     # Train an XGBoost model on the ensemble predictions
#     xgb_model_ensemble = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
#     xgb_model_ensemble.fit(ensemble_input, y_test)
#     # Make predictions using the ensemble XGBoost model
#     ensemble_prediction_sample = xgb_model_ensemble.predict(ensemble_input_sample)
#     # Get the predicted genre with the highest probability
#     predicted_genre_ensemble = genre_names[ensemble_prediction_sample[0]]
#     # Get the predicted probabilities for each genre
#     predicted_probabilities_ensemble = xgb_model_ensemble.predict_proba(ensemble_input_sample)[0]

#     # Find the maximum predicted percentage
#     max_predicted_percentage = max(predicted_probabilities_ensemble)

#     # Create a dictionary to store predicted percentages for each class
#     # Initialize an empty dictionary to store predicted percentages
#     predicted_percentages_ensemble = {}

#     # Loop through each genre and its corresponding probability
#     for genre, percentage in zip(genre_names.values(), predicted_probabilities_ensemble):
#         predicted_percentages_ensemble[genre] = float(percentage) * 100

#     # Add a condition to check if the maximum predicted percentage is less than 50%
#     if max_predicted_percentage < 50:
#         predicted_genre_ensemble = "Audio Unidentified"
#         predicted_percentages_ensemble = {}

#     # Create a dictionary for the JSON result
#     json_result_ensemble = {
#         "Predicted Genre (XGB)": predicted_genre_ensemble,
#         "Predicted Percentages (XGB)": predicted_percentages_ensemble
#     }

#     merged_json_result = {"CNN": json_result_cnn, "OCNN": json_result_ensemble}
#     return merged_json_result
