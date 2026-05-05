# MedicalChatBot

# How to run?
### Step 1:

Clone the repository

```bash
Project repo: https://github.com/kaiser-ahmed-siyam/MedicalChatBot
```
### Step 2- Create a virtual environment after opening the repository

```bash
python3.11 -m venv venv
```
### Step 3-  Activate environment
```bash
venv/Scripts/Activate.ps1

```
### Step 4- to run the app in Windows
```bash

python app.py  

```
#### Python version 3.11.9

### Run the app in Linux 
gunicorn app:app --workers 1 --timeout 120 --bind 0.0.0.0:$PORT