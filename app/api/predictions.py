from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import uuid

from app.core.database import get_db
from app.schemas.pydantic_schemas import TransactionScoreRequest, PredictionResponse
from app.services.ml_services import ml_service
from app.models.database_models import TransactionDB

router = APIRouter(prefix="/predictions", tags=["Real-Time Scoring"])

@router.post("/score", response_model=PredictionResponse)
def score_transaction(payload: TransactionScoreRequest, db: Session = Depends(get_db)):
    try:
        result = ml_service.predict_transaction(payload.dict())
        tx_id = f"TXN-{uuid.uuid4().hex[:8].upper()}"
        
        db_transaction = TransactionDB(
            transaction_id=tx_id,
            step=payload.step,
            type=payload.type,
            amount=payload.amount,
            oldbalanceOrg=payload.oldbalanceOrg,
            newbalanceOrig=payload.newbalanceOrig,
            oldbalanceDest=payload.oldbalanceDest,
            newbalanceDest=payload.newbalanceDest,
            fraud_probability=result["fraud_probability"],
            risk_category=result["risk_category"],
            model_version=result["model_version"],
            investigation_status="PENDING"
        )
        
        db.add(db_transaction)
        db.commit()
        db.refresh(db_transaction)
        
        return {
            "transaction_id": tx_id,
            "fraud_probability": result["fraud_probability"],
            "risk_category": result["risk_category"],
            "model_version": result["model_version"],
            "top_risk_factors": result["top_risk_factors"]
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Prediction processing error: {str(e)}")