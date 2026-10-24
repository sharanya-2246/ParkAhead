import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from datetime import datetime, timedelta
from config import Config
from database.models import db, ParkingFacility, ParkingObservation, PredictionRecord, RecommendationRecord
from ml.preprocessing import ParkingDataPreprocessor
from ml.feature_engineering import FeatureEngineer
from ml.train import ParkingModelTrainer
from ml.predict import ParkingPredictor
from services.recommendation_service import RecommendationService
import pandas as pd

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

preprocessor = ParkingDataPreprocessor()
fe = FeatureEngineer()
trainer = ParkingModelTrainer()
predictor = ParkingPredictor()
recommender = RecommendationService()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/planner', methods=['GET', 'POST'])
def planner():
    facilities = ParkingFacility.query.filter_by(is_active=True).all()
    if request.method == 'POST':
        try:
            destination = request.form.get('destination')
            main_facility = request.form.get('main_facility')
            event_date = request.form.get('event_date')
            event_time_str = request.form.get('event_time')
            eta_date = request.form.get('eta_date')
            eta_time_str = request.form.get('eta_time')
            budget = request.form.get('budget')
            
            event_datetime = datetime.strptime(f'{event_date} {event_time_str}', '%Y-%m-%d %H:%M')
            eta_datetime = datetime.strptime(f'{eta_date} {eta_time_str}', '%Y-%m-%d %H:%M')
            
            if event_datetime < datetime.now():
                flash('Event time is in the past. Please enter a future event time.', 'warning')
                return render_template('planner.html', facilities=facilities)
            
            if eta_datetime > event_datetime:
                flash('Estimated arrival time is after event start time. Consider leaving earlier.', 'warning')
            
            # Load historical data
            try:
                df_hist = trainer.prepare_data('birmingham')
            except Exception as e:
                flash(f'Error loading data: {e}. Please ensure datasets are in data/raw/', 'danger')
                return render_template('planner.html', facilities=facilities)
            
            # Load model
            if not predictor.load_model('all'):
                flash('Model not trained yet. Please train the model first via /train or dashboard.', 'warning')
                return render_template('planner.html', facilities=facilities)
            
            # Get all facilities with data
            fac_list = df_hist['facility_id'].unique()
            predictions = []
            for fid in fac_list:
                try:
                    fac_obs = df_hist[df_hist['facility_id'] == fid]
                    cap = int(fac_obs['capacity'].iloc[0]) if 'capacity' in fac_obs.columns else None
                    pred = predictor.predict_at_eta(df_hist, fid, eta_datetime, cap)
                    # Add facility metadata
                    pf = ParkingFacility.query.filter_by(facility_id=fid).first()
                    if pf:
                        pred['facility_name'] = pf.name
                        pred['travel_time_min'] = pf.travel_time_min or 15
                        pred['walking_time_min'] = pf.walking_time_min or 3
                        pred['parking_fee'] = pf.parking_fee or 0
                    else:
                        pred['facility_name'] = fid
                        pred['travel_time_min'] = 15
                        pred['walking_time_min'] = 3
                        pred['parking_fee'] = 0
                    pred['search_time_min'] = 5
                    predictions.append(pred)
                except Exception as e:
                    continue
            
            if not predictions:
                flash('No predictions available for configured facilities.', 'danger')
                return render_template('planner.html', facilities=facilities)
            
            user_input = {
                'event_time': event_datetime,
                'eta': eta_datetime,
                'budget': float(budget) if budget else None
            }
            
            recommendations = recommender.recommend(predictions, user_input)
            
            # Save recommendation
            try:
                best = recommendations[0] if recommendations else None
                rec = RecommendationRecord(
                    requested_event_time=event_datetime,
                    estimated_arrival_time=eta_datetime,
                    selected_facility_id=best['facility_id'] if best else None,
                    recommendation_reason=best.get('recommendation_reason', '') if best else ''
                )
                db.session.add(rec)
                db.session.commit()
            except:
                db.session.rollback()
            
            return render_template('results.html', recommendations=recommendations, 
                                 event_time=event_datetime, eta=eta_datetime,
                                 destination=destination)
            
        except Exception as e:
            flash(f'Error processing request: {str(e)}', 'danger')
            return render_template('planner.html', facilities=facilities)
    
    return render_template('planner.html', facilities=facilities)

@app.route('/results')
def results():
    return render_template('results.html', recommendations=[])

@app.route('/dashboard')
def dashboard():
    try:
        trainer_inst = ParkingModelTrainer()
        df = trainer_inst.prepare_data('birmingham')
        cleaning_report = trainer_inst.cleaning_report
        metrics = trainer_inst.get_metrics_df() if hasattr(trainer_inst, 'get_metrics_df') else pd.DataFrame()
        facilities = df['facility_id'].unique()
        return render_template('dashboard.html', 
                             cleaning_report=cleaning_report,
                             metrics=metrics,
                             facilities=facilities,
                             total_records=len(df))
    except Exception as e:
        flash(f'Error loading dashboard: {str(e)}. Train model first.', 'warning')
        return render_template('dashboard.html', cleaning_report={}, metrics=pd.DataFrame(), facilities=[], total_records=0)

@app.route('/train')
def train_model():
    try:
        trainer_inst = ParkingModelTrainer()
        df = trainer_inst.prepare_data('birmingham')
        results = trainer_inst.train_models(df)
        model_path = trainer_inst.save_best_model('all')
        flash(f'Model trained successfully! Saved to {model_path}', 'success')
        return redirect(url_for('dashboard'))
    except Exception as e:
        flash(f'Training error: {str(e)}', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/init_db')
def init_db():
    try:
        db.create_all()
        # Add sample facilities
        if ParkingFacility.query.count() == 0:
            facilities = [
                ParkingFacility(facility_id='BHMBCCMKT01', name='Bull Ring Main Car Park', capacity=577, 
                              parking_fee=50, walking_time_min=2, travel_time_min=10),
                ParkingFacility(facility_id='BHMBRCMP01', name='Bull Ring Centre Parking', capacity=387,
                              parking_fee=40, walking_time_min=3, travel_time_min=12),
                ParkingFacility(facility_id='BHMBRTARC01', name='Ratcliff Terrace Car Park', capacity=293,
                              parking_fee=30, walking_time_min=7, travel_time_min=15),
            ]
            for f in facilities:
                db.session.add(f)
            db.session.commit()
        flash('Database initialized with sample facilities!', 'success')
    except Exception as e:
        flash(f'DB init error: {str(e)}', 'danger')
    return redirect(url_for('index'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='127.0.0.1', port=5000)
