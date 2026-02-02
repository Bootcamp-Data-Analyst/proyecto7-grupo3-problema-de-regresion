from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

Base = declarative_base()

class PredictionFeedback(Base):
    __tablename__ = 'feedback'
    id = Column(Integer, primary_key=True)
    brand = Column(String)
    model_year = Column(Integer)
    milage = Column(Float)
    predicted_price = Column(Float)
    actual_price = Column(Float)
    model_used = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

class NewData(Base):
    __tablename__ = 'training_data'
    id = Column(Integer, primary_key=True)
    brand = Column(String)
    model = Column(String)
    model_year = Column(Integer)
    milage = Column(Float)
    fuel_type = Column(String)
    engine = Column(String)
    transmission = Column(String)
    ext_col = Column(String)
    int_col = Column(String)
    accident = Column(String)
    clean_title = Column(String)
    price = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

def init_db():
    engine = create_engine('sqlite:///data/ml_ops.db')
    Base.metadata.create_all(engine)
    return engine

if __name__ == "__main__":
    import os
    os.makedirs('data', exist_ok=True)
    init_db()
    print("Database initialized.")
