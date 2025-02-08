# Action-Detector-System

## Overview
The **Action Detector System** is a machine learning-based application designed to analyze game commentary transcripts and extract valid action phrases. The system identifies whether a detected action accurately represents an event on the court.

## Features
- Processes game transcripts to detect action phrases.
- Determines the validity of detected actions based on context.
- Provides a user-friendly interface using **Streamlit**.

## Installation
To set up and run the system, follow these steps:

### 1. Clone the Repository
```sh
git clone https://github.com/IdanArbiv/Action-Detector-System.git
cd Action-Detector-System
```
### 2. Download the Classifiers Directory
After cloning the repository, download the `classifiers` directory from the following link:  
[Download Classifiers Directory](https://drive.google.com/drive/folders/1hXqL2rggNzm3vS047ZNr9z2NOAkIbYOo?hl=he)

Once downloaded, extract and paste the `classifiers` directory into the root of the project.

### 3. Install Dependencies
Ensure you have **Python 3.9+** installed, then run:
```sh
pip install -r requirements.txt
```

### 4. Run the Web Application
Launch the Streamlit interface with:
```sh
streamlit run ./web_app/run_ui.py
```

## Usage
1. Open the web application in your browser.
2. Enter a transcript describing a game event.
3. The system will analyze the text and return a valid action phrase if one exists.

## System I/O
- **Input:** A game commentary transcript.
- **Output:** A valid action phrase (if detected and valid) or a message indicating no valid action was found.

## Contributors
- **Idan Arbiv**


