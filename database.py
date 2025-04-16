from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session
import bcrypt
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Employee(Base):
    __tablename__ = 'employees'
    
    id = Column(Integer, primary_key=True)
    employee_name = Column(String)
    age = Column(Integer)
    gender = Column(String)
    relationship_status = Column(String)
    distance_from_home = Column(Float)
    monthly_income = Column(Float)
    job_role = Column(String)
    department = Column(String)
    years_at_company = Column(Integer)
    work_life_balance = Column(Integer)
    job_level = Column(String)
    training_times_last_year = Column(Integer)
    job_satisfaction = Column(Integer)
    performance_rating = Column(Integer)
    environment_satisfaction = Column(Integer)
    attrition = Column(Boolean)
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer)

class PredictionHistory(Base):
    __tablename__ = 'prediction_history'
    
    id = Column(Integer, primary_key=True)
    employee_name = Column(String)
    age = Column(Integer)
    gender = Column(String)
    relationship_status = Column(String)
    distance_from_home = Column(Float)
    monthly_income = Column(Float)
    job_role = Column(String)
    department = Column(String)
    years_at_company = Column(Integer)
    work_life_balance = Column(Integer)
    job_level = Column(String)
    training_times_last_year = Column(Integer)
    job_satisfaction = Column(Integer)
    performance_rating = Column(Integer)
    environment_satisfaction = Column(Integer)
    attrition_risk = Column(Float)  # Store the actual risk score (0-1)
    prediction_date = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer)

# Create database engine
engine = create_engine('sqlite:///employee_attrition.db')
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Session:
    db = SessionLocal()
    try:
        return db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def create_user(db: Session, username: str, password: str, email: str):
    # Hash the password
    salt = bcrypt.gensalt()
    password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
    
    # Create new user
    user = User(
        username=username,
        password_hash=password_hash.decode('utf-8'),
        email=email
    )
    db.add(user)
    db.commit()
    return user

def verify_user(db: Session, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if user and bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
        return user
    return None

def save_employee_data(db: Session, data: dict, user_id: int):
    employee = Employee(
        employee_name=data.get('EmployeeName', ''),
        age=data.get('Age', 0),
        gender=data.get('Gender', ''),
        relationship_status=data.get('RelationshipStatus', ''),
        distance_from_home=data.get('DistanceFromHome', 0),
        monthly_income=data.get('MonthlyIncome', 0),
        job_role=data.get('JobRole', ''),
        department=data.get('Department', ''),
        years_at_company=data.get('YearsAtCompany', 0),
        work_life_balance=data.get('WorkLifeBalance', 0),
        job_level=data.get('JobLevel', ''),
        training_times_last_year=data.get('TrainingTimesLastYear', 0),
        job_satisfaction=data.get('JobSatisfaction', 0),
        performance_rating=data.get('PerformanceRating', 0),
        environment_satisfaction=data.get('EnvironmentSatisfaction', 0),
        attrition=data.get('Attrition', False),
        user_id=user_id
    )
    db.add(employee)
    db.commit()
    return employee

def get_employee_data(db: Session, user_id: int):
    return db.query(Employee).filter(Employee.user_id == user_id).all()

def save_prediction_history(db: Session, data: dict, user_id: int):
    prediction = PredictionHistory(
        employee_name=data.get('EmployeeName', ''),
        age=data.get('Age', 0),
        gender=data.get('Gender', ''),
        relationship_status=data.get('RelationshipStatus', ''),
        distance_from_home=data.get('DistanceFromHome', 0),
        monthly_income=data.get('MonthlyIncome', 0),
        job_role=data.get('JobRole', ''),
        department=data.get('Department', ''),
        years_at_company=data.get('YearsAtCompany', 0),
        work_life_balance=data.get('WorkLifeBalance', 0),
        job_level=data.get('JobLevel', ''),
        training_times_last_year=data.get('TrainingTimesLastYear', 0),
        job_satisfaction=data.get('JobSatisfaction', 0),
        performance_rating=data.get('PerformanceRating', 0),
        environment_satisfaction=data.get('EnvironmentSatisfaction', 0),
        attrition_risk=data.get('AttritionRisk', 0.0),
        user_id=user_id
    )
    db.add(prediction)
    db.commit()
    return prediction

def get_prediction_history(db: Session, user_id: int, limit: int = 50):
    return db.query(PredictionHistory).filter(PredictionHistory.user_id == user_id).order_by(PredictionHistory.prediction_date.desc()).limit(limit).all() 