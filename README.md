# Agent Project

A simple agent project with Streamlit frontend and FastAPI backend.

## Project Structure

```
├── backend/
│   └── main.py          # FastAPI backend
├── frontend/
│   └── app.py           # Streamlit frontend
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables
├── .env.example         # Environment variables template
└── .gitignore          # Git ignore file
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Copy environment variables:
```bash
cp .env.example .env
```

## Running the Application

1. Start the backend (in one terminal):
```bash
cd backend
python main.py
```

2. Start the frontend (in another terminal):
```bash
cd frontend
streamlit run app.py
```

The FastAPI backend will run on http://localhost:8000
The Streamlit frontend will run on http://localhost:8501