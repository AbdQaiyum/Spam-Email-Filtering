import os
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

# Define the input and output directories
input_dir = 'input_emails'
output_dir = 'preprocessed_emails'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each email in the input directory and save in the output directory
for filename in os.listdir(input_dir):
    with open(os.path.join(input_dir, filename), 'r', encoding='utf-8', errors='ignore') as file:
        email_text = file.read()
        preprocessed_email = preprocess_email(email_text)
        with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as output_file:
            output_file.write(preprocessed_email)
