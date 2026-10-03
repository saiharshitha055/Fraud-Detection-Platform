from pydantic import BaseModel, Field
from typing import Optional

class TransactionScoreRequest(BaseModel):
    step: int = Field(..., description="Hour step of the transaction simulation")
    type: str = Field(..., description="Transaction type e.g., TRANSFER, CASH_OUT")
    amount: float = Field(..., gt=0, description="Transaction amount")
    oldbalanceOrg: float = Field(..., ge=0, description="Sender original balance")
    newbalanceOrig: float = Field(..., ge=0, description="Sender new balance")
    oldbalanceDest: float = Field(..., ge=0, description="Receiver original balance")
    newbalanceDest: float = Field(..., ge=0, description="Receiver new balance")

class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    risk_category: str
    model_version: str
    top_risk_factors: list[str]

class InvestigationUpdate(BaseModel):
    status: str = Field(..., description="CONFIRMED_FRAUD, LEGITIMATE, or UNDER_REVIEW")
    notes: Optional[str] = None