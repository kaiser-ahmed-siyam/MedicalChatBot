# MedicalChatBot

# How to run?
### STEPS:

Clone the repository

```bash
Project repo: https://github.com/
```
### STEP 01- Create a virtual environment after opening the repository

```bash
python -m venv venv
```

```bash
venv/Scripts/Activate.ps1

```
```bash
python app.py # to run the app localhost 

```
#### Python version 3.11.9

### Run the app
gunicorn app:app --workers 1 --timeout 120 --bind 0.0.0.0:$PORT