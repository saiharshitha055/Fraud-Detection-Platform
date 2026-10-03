from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text
from datetime import datetime
from app.core.database import Base

class UserDB(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="VIEWER")  # ADMIN, FRAUD_ANALYST, VIEWER
    is_active = Column(Boolean, default=True)

class TransactionDB(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String, unique=True, index=True)
    step = Column(Integer)
    type = Column(String)
    amount = Column(Float)
    oldbalanceOrg = Column(Float)
    newbalanceOrig = Column(Float)
    oldbalanceDest = Column(Float)
    newbalanceDest = Column(Float)
    
    # Prediction Results
    fraud_probability = Column(Float)
    risk_category = Column(String)  # LOW, MEDIUM, HIGH
    model_version = Column(String)
    
    # Investigation Workflow
    investigation_status = Column(String, default="PENDING")  # PENDING, CONFIRMED_FRAUD, LEGITIMATE, UNDER_REVIEW
    analyst_notes = Column(Text, nullable=True)
    
    timestamp = Column(DateTime, default=datetime.utcnow)