import os
import joblib
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re

# Define a function to preprocess a single email
def preprocess_email(email_text):
    # Split the email into lines and remove headers
    lines = email_text.split('\n')
    content_start = 0
    for i, line in enumerate(lines):
        if line.strip() == '':
            content_start = i + 1
            break
    email_text = '\n'.join(lines[content_start:])

    # Lowercase the text
    email_text = email_text.lower()

    # Remove special characters and numbers
    email_text = re.sub(r'[^a-z\s]', '', email_text)

    return email_text

# Define a function to load and preprocess the emails
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

    # Save the trained model and vectorizer to the same directory
    model_filename = os.path.join(model_dir, 'email_classifier_model.joblib')

    # Create the "trained_model" directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model, model_filename)
    joblib.dump(vectorizer, os.path.join(model_dir, 'email_vectorizer.joblib'))
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

# Function to upload a text file to a specified folder and label (ham or spam)
def upload_text_file(target_dir, label):
    file_paths = filedialog.askopenfilenames(filetypes=[("Text Files", "*.txt")])
    
    for file_path in file_paths:
        if not file_path:
            continue  # User canceled the file dialog
        
        with open(file_path, 'r', encoding='latin1') as file:
            email_text = file.read()
            preprocessed_email = preprocess_email(email_text)
        
        if label:
            label_dir = os.path.join(target_dir, label)
            # Copy the preprocessed content to the selected folder and label
            file_name = os.path.basename(file_path)
            target_path = os.path.join(label_dir, file_name)
            with open(target_path, 'w', encoding='utf-8') as target_file:
                target_file.write(preprocessed_email)
            
            print(f"File '{file_name}' copied and preprocessed to '{label}' in '{target_dir}'")

# Function to classify email text as spam or ham
def classify_email():
    global model
    global vectorizer
    email_text = email_text_box.get("1.0", "end-1c")  # Get text from the text box
    preprocessed_email = preprocess_email(email_text)
    email_tfidf = vectorizer.transform([preprocessed_email])
    prediction = model.predict(email_tfidf)
    result_label.config(text="Result: " + ("Spam" if prediction[0] == 1 else "Ham"))

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

upload_frame = ttk.LabelFrame(window, text="Upload Text File")
upload_frame.pack()

train_upload_label = ttk.Label(upload_frame, text="Upload to:")
train_upload_label.grid(row=0, column=0)

train_upload_ham_button = tk.Button(upload_frame, text="Train Ham", command=lambda: upload_text_file(train_dir, "ham"))
train_upload_ham_button.grid(row=0, column=1)

train_upload_spam_button = tk.Button(upload_frame, text="Train Spam", command=lambda: upload_text_file(train_dir, "spam"))
train_upload_spam_button.grid(row=0, column=2)

test_upload_label = ttk.Label(upload_frame, text="Upload to:")
test_upload_label.grid(row=1, column=0)

test_upload_ham_button = tk.Button(upload_frame, text="Test Ham", command=lambda: upload_text_file(test_dir, "ham"))
test_upload_ham_button.grid(row=1, column=1)

test_upload_spam_button = tk.Button(upload_frame, text="Test Spam", command=lambda: upload_text_file(test_dir, "spam"))
test_upload_spam_button.grid(row=1, column=2)

classify_frame = ttk.LabelFrame(window, text="Classify Email Text")
classify_frame.pack()

email_text_label = ttk.Label(classify_frame, text="Email Text:")
email_text_label.grid(row=0, column=0)

email_text_box = tk.Text(classify_frame, height=10, width=40)
email_text_box.grid(row=0, column=1)

classify_button = tk.Button(classify_frame, text="Classify Email", command=classify_email)
classify_button.grid(row=0, column=2)

result_label = ttk.Label(classify_frame, text="Result:")
result_label.grid(row=1, column=0, columnspan=3)

window.mainloop()
