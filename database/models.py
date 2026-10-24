from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class ParkingFacility(db.Model):
    __tablename__ = 'parking_facility'
    id = db.Column(db.Integer, primary_key=True)
    facility_id = db.Column(db.String(64), unique=True, nullable=False)
    name = db.Column(db.String(128), nullable=False)
    capacity = db.Column(db.Integer)
    address = db.Column(db.String(256))
    parking_fee = db.Column(db.Float)
    walking_time_min = db.Column(db.Float)
    travel_time_min = db.Column(db.Float)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    observations = db.relationship('ParkingObservation', backref='facility', lazy=True)

class ParkingObservation(db.Model):
    __tablename__ = 'parking_observation'
    id = db.Column(db.Integer, primary_key=True)
    facility_id = db.Column(db.String(64), db.ForeignKey('parking_facility.facility_id'), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, index=True)
    capacity = db.Column(db.Integer)
    occupancy = db.Column(db.Integer)
    available_spaces = db.Column(db.Integer)
    occupancy_rate = db.Column(db.Float)
    entry_count = db.Column(db.Integer)
    exit_count = db.Column(db.Integer)
    data_source = db.Column(db.String(64))
    
    __table_args__ = (db.Index('idx_facility_ts', 'facility_id', 'timestamp'),)

class PredictionRecord(db.Model):
    __tablename__ = 'prediction_record'
    id = db.Column(db.Integer, primary_key=True)
    facility_id = db.Column(db.String(64), nullable=False)
    prediction_timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    target_timestamp = db.Column(db.DateTime, nullable=False)
    predicted_occupancy = db.Column(db.Float)
    predicted_available_spaces = db.Column(db.Float)
    predicted_occupancy_rate = db.Column(db.Float)
    model_name = db.Column(db.String(64))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class RecommendationRecord(db.Model):
    __tablename__ = 'recommendation_record'
    id = db.Column(db.Integer, primary_key=True)
    requested_event_time = db.Column(db.DateTime)
    estimated_arrival_time = db.Column(db.DateTime)
    selected_facility_id = db.Column(db.String(64))
    recommendation_reason = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
