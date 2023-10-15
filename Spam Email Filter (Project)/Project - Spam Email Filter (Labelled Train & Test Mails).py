import os
import joblib
import tkinter as tk
from tkinter import filedialog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define the function to load and preprocess the emails
def load_and_preprocess_emails(directory):
    emails = []
    labels = []

    for label in ['ham', 'spam']:
        label_dir = os.path.join(directory, label)
        for filename in os.listdir(label_dir):
            with open(os.path.join(label_dir, filename), 'r', encoding='latin1') as file:
                email_text = file.read()
                emails.append(email_text)
                labels.append(0 if label == 'ham' else 1)

    return emails, labels

# Train and save the model function
def train_and_save_model():
    global model
    global vectorizer
    global model_dir
    global model_filename

    # Load and preprocess email data
    train_email, train_email_label = load_and_preprocess_emails(train_dir)

    # Create a TF-IDF vectorizer to convert text to numerical features
    vectorizer = TfidfVectorizer(max_features=90000, stop_words='english')
    train_email_tfidf = vectorizer.fit_transform(train_email)

    # Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=135, random_state=20)
    model.fit(train_email_tfidf, train_email_label)

    # Save the trained model and vectorizer to the directory
    model_filename = os.path.join(model_dir, 'email_classifier_model.joblib')
    vectorizer_filename = os.path.join(model_dir, 'email_vectorizer.joblib')
    joblib.dump(model, model_filename)
    joblib.dump(vectorizer, vectorizer_filename)
    print("Model trained and saved.")

# Load and test the model function
def load_and_test_model():
    global model
    global vectorizer
    global model_filename
    global test_dir

    # Load the trained model and vectorizer
    loaded_model = joblib.load(model_filename)
    loaded_vectorizer = joblib.load(os.path.join(model_dir, 'email_vectorizer.joblib'))

    # Load and preprocess test email data
    test_email, test_email_label = load_and_preprocess_emails(test_dir)

    # Use the loaded vectorizer to transform test data
    test_email_tfidf = loaded_vectorizer.transform(test_email)

    # Make predictions on the test data
    pred_test_email_label = loaded_model.predict(test_email_tfidf)

    # Evaluate the model
    accuracy = accuracy_score(test_email_label, pred_test_email_label)
    confusion = confusion_matrix(test_email_label, pred_test_email_label)
    report = classification_report(test_email_label, pred_test_email_label, target_names=['ham', 'spam'])

    print('Accuracy:', accuracy)
    print('Confusion Matrix:')
    print(confusion)
    print('Classification Report:')
    print(report)

# Create a UI with buttons
window = tk.Tk()
window.title("Email Classifier")
model_dir = 'trained_model'
model_filename = os.path.join(model_dir, 'email_classifier_model.joblib')
train_dir = 'train-mails'
test_dir = 'test-mails'

train_button = tk.Button(window, text="Train Model", command=train_and_save_model)
train_button.pack()

test_button = tk.Button(window, text="Load & Test Model", command=load_and_test_model)
test_button.pack()

window.mainloop()
