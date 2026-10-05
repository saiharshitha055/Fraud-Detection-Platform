# Real-Time Enterprise Fraud Detection and Risk Analytics Platform

A production-grade, full-stack fintech platform designed for real-time transaction risk scoring, model explainability, portfolio telemetry, and automated batch analysis.

## Live Demo and Architecture
- Backend API (FastAPI): Hosted live on Railway (https://fraud-detection-platform-production-d699.up.railway.app)
- Frontend Dashboard (Streamlit): Hosted on Streamlit Community Cloud (https://fintech-fraud-detection-platform-055.streamlit.app/)
- Database: PostgreSQL / SQLite via SQLAlchemy ORM

## Key Features

- Interactive Risk Scoring Engine: Real-time simulation of financial transactions (Amount, Balances, Steps, and Channel Types) with instant ML inference.
- Model Explainability and Feature Contribution Weights: Dynamic SHAP/feature attribution visualizations utilizing Plotly horizontal bar charts to transparently display top risk factors.
- Executive Analytics Dashboard: Comprehensive overview tracking portfolio volume, calculated fraud rates, high-risk vectors, and channel distribution.
- Batch Dataset Upload: Automated CSV ingestion and column mapping for bulk portfolio risk categorization.
- Production-Grade Backend: Built with FastAPI, featuring CORS middleware, structured Pydantic schemas, and automated database persistence.

## Tech Stack

- Backend: Python, FastAPI, Uvicorn, SQLAlchemy, Pydantic, Scikit-learn, TensorFlow
- Frontend: Streamlit, Plotly, Requests, Pandas, NumPy
- Database: SQLite / PostgreSQL
- Deployment and DevOps: Railway (Backend), Streamlit Community Cloud (Frontend), Git, Docker, Swagger UI

## Project Structure

Fraud-Detection-Platform/
│
├── app/                  # FastAPI Backend
│   ├── api/              # API routers (predictions.py)
│   ├── core/             # Database configuration and session management
│   ├── models/           # SQLAlchemy database models
│   ├── schemas/          # Pydantic request/response validation schemas
│   └── services/         # ML inference and business logic services
│
├── streamlit_app/        # Streamlit Frontend
│   ├── pages/
│   │   ├── 1_Risk_Scorer.py  # Interactive risk scoring and explainability charts
│   │   └── 2_Fraud_Trends.py # Portfolio telemetry and trend analytics
│   └── app.py            # Executive dashboard entry point
│
├── tests/                # Unit and integration test suite
├── requirements.txt      # Project dependencies
└── main.py               # FastAPI application entry point

## Local Installation and Setup

1. Clone the Repository:
   git clone https://github.com/saiharshitha055/Fraud-Detection-Platform.git
   cd Fraud-Detection-Platform

2. Create and Activate Virtual Environment:
   python -m venv venv
   # On Windows:
   venv\Scripts\activate

3. Install Dependencies:
   pip install -r requirements.txt

4. Run the FastAPI Backend Locally:
   uvicorn main:app --reload --port 8000
   *(Access interactive Swagger documentation at http://localhost:8000/docs)*

5. Run the Streamlit Frontend Locally:
   streamlit run streamlit_app/app.py

## API Endpoints

- GET /health - System health check and database connectivity status.
- POST /predictions/score - Evaluates transaction payloads, performs ML inference, logs results to the database, and returns risk scores with explainability weights.
