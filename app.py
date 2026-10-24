
import os
import json
import logging
import pandas as pd

from datetime import datetime, timedelta
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify
)

from config import Config
from database.models import (
    db,
    ParkingFacility,
    ParkingObservation,
    PredictionRecord,
    RecommendationRecord
)

from ml.preprocessing import ParkingDataPreprocessor
from ml.feature_engineering import FeatureEngineer
from ml.train import ParkingModelTrainer
from ml.predict import ParkingPredictor
from services.recommendation_service import RecommendationService


# --------------------------------------------------
# Flask application configuration
# --------------------------------------------------

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
METRICS_PATH = os.path.join(MODELS_DIR, 'parkahead_metrics.json')

os.makedirs(MODELS_DIR, exist_ok=True)

# Initialize project services
preprocessor = ParkingDataPreprocessor()
fe = FeatureEngineer()
trainer = ParkingModelTrainer()
predictor = ParkingPredictor()
recommender = RecommendationService()


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


# --------------------------------------------------
# Parking planner and prediction
# --------------------------------------------------

@app.route('/planner', methods=['GET', 'POST'])
def planner():
    facilities = ParkingFacility.query.filter_by(
        is_active=True
    ).all()

    if request.method == 'POST':
        try:
            destination = request.form.get('destination', '').strip()
            main_facility = request.form.get('main_facility', '')
            event_date = request.form.get('event_date', '')
            event_time_str = request.form.get('event_time', '')
            eta_date = request.form.get('eta_date', '')
            eta_time_str = request.form.get('eta_time', '')
            budget = request.form.get('budget', '')

            if not event_date or not event_time_str:
                flash('Please enter the event date and time.', 'warning')
                return render_template(
                    'planner.html', facilities=facilities
                )

            if not eta_date or not eta_time_str:
                flash('Please enter your estimated arrival date and time.', 'warning')
                return render_template(
                    'planner.html', facilities=facilities
                )

            event_datetime = datetime.strptime(
                f'{event_date} {event_time_str}',
                '%Y-%m-%d %H:%M'
            )

            eta_datetime = datetime.strptime(
                f'{eta_date} {eta_time_str}',
                '%Y-%m-%d %H:%M'
            )

            if event_datetime < datetime.now():
                flash(
                    'Event time is in the past. Please enter a future event time.',
                    'warning'
                )
                return render_template(
                    'planner.html', facilities=facilities
                )

            if eta_datetime > event_datetime:
                flash(
                    'Your estimated arrival is after the event starts. '
                    'Consider choosing an earlier arrival time.',
                    'warning'
                )

            if budget:
                budget_value = float(budget)
                if budget_value < 0:
                    raise ValueError('Budget cannot be negative.')
            else:
                budget_value = None

            # Load and preprocess historical parking data
            try:
                df_hist = trainer.prepare_data('birmingham')
            except Exception as e:
                app.logger.exception('Failed to load parking data')
                flash(
                    f'Error loading historical data: {str(e)}',
                    'danger'
                )
                return render_template(
                    'planner.html', facilities=facilities
                )

            # Load the trained model
            if not predictor.load_model('all'):
                flash(
                    'The model has not been trained yet. '
                    'Please train it from the dashboard first.',
                    'warning'
                )
                return render_template(
                    'planner.html', facilities=facilities
                )

            predictions = []

            # Generate predictions for facilities in the dataset
            facility_ids = df_hist['facility_id'].dropna().unique()

            for fid in facility_ids:
                try:
                    fac_obs = df_hist[
                        df_hist['facility_id'] == fid
                    ]

                    if fac_obs.empty:
                        continue

                    capacity = (
                        int(fac_obs['capacity'].iloc[0])
                        if 'capacity' in fac_obs.columns
                        else None
                    )

                    pred = predictor.predict_at_eta(
                        df_hist,
                        fid,
                        eta_datetime,
                        capacity
                    )

                    # Add configured facility information if available
                    parking_facility = ParkingFacility.query.filter_by(
                        facility_id=fid
                    ).first()

                    if parking_facility:
                        pred['facility_name'] = parking_facility.name
                        pred['travel_time_min'] = (
                            parking_facility.travel_time_min or 15
                        )
                        pred['walking_time_min'] = (
                            parking_facility.walking_time_min or 3
                        )
                        pred['parking_fee'] = (
                            parking_facility.parking_fee or 0
                        )
                    else:
                        pred['facility_name'] = fid
                        pred['travel_time_min'] = 15
                        pred['walking_time_min'] = 3
                        pred['parking_fee'] = 0

                    pred['search_time_min'] = 5

                    predictions.append(pred)

                except Exception:
                    app.logger.exception(
                        'Prediction failed for facility %s', fid
                    )
                    continue

            if not predictions:
                flash(
                    'No predictions could be generated. '
                    'Check the historical dataset and trained model.',
                    'danger'
                )
                return render_template(
                    'planner.html', facilities=facilities
                )

            user_input = {
                'event_time': event_datetime,
                'eta': eta_datetime,
                'budget': budget_value,
                'destination': destination,
                'main_facility': main_facility
            }

            # Rank facilities using the recommendation service
            recommendations = recommender.recommend(
                predictions, user_input
            )

            # Save the top recommendation
            try:
                best = (
                    recommendations[0]
                    if recommendations else None
                )

                recommendation_record = RecommendationRecord(
                    requested_event_time=event_datetime,
                    estimated_arrival_time=eta_datetime,
                    selected_facility_id=(
                        best['facility_id'] if best else None
                    ),
                    recommendation_reason=(
                        best.get('recommendation_reason', '')
                        if best else ''
                    )
                )

                db.session.add(recommendation_record)
                db.session.commit()

            except Exception:
                db.session.rollback()
                app.logger.exception(
                    'Could not save recommendation record'
                )

            return render_template(
                'results.html',
                recommendations=recommendations,
                event_time=event_datetime,
                eta=eta_datetime,
                destination=destination
            )

        except (ValueError, TypeError) as e:
            flash(f'Please check your input: {str(e)}', 'warning')
            return render_template(
                'planner.html', facilities=facilities
            )

        except Exception as e:
            app.logger.exception('Planner request failed')
            flash(
                f'Error processing your request: {str(e)}',
                'danger'
            )
            return render_template(
                'planner.html', facilities=facilities
            )

    return render_template('planner.html', facilities=facilities)


# --------------------------------------------------
# Results page
# --------------------------------------------------

@app.route('/results')
def results():
    return render_template(
        'results.html',
        recommendations=[],
        event_time=None,
        eta=None,
        destination=None
    )


# --------------------------------------------------
# Dashboard
# --------------------------------------------------

@app.route('/dashboard')
def dashboard():
    try:
        # Load and preprocess historical data
        trainer_inst = ParkingModelTrainer()
        df = trainer_inst.prepare_data('birmingham')

        cleaning_report = trainer_inst.cleaning_report or {}
        facilities = sorted(
            df['facility_id'].dropna().unique().tolist()
        )
        total_records = len(df)

        # Load saved evaluation metrics, if available
        if os.path.exists(METRICS_PATH):
            with open(
                METRICS_PATH, 'r', encoding='utf-8'
            ) as file:
                saved_metrics = json.load(file)

            metrics = pd.DataFrame.from_dict(
                saved_metrics, orient='index'
            )
        else:
            metrics = pd.DataFrame()

        return render_template(
            'dashboard.html',
            cleaning_report=cleaning_report,
            metrics=metrics,
            facilities=facilities,
            total_records=total_records
        )

    except Exception as e:
        app.logger.exception('Failed to load dashboard')

        flash(
            f'Could not load dashboard data: {str(e)}',
            'danger'
        )

        return render_template(
            'dashboard.html',
            cleaning_report={},
            metrics=pd.DataFrame(),
            facilities=[],
            total_records=0
        )


# --------------------------------------------------
# Train or retrain the model
# --------------------------------------------------

@app.route('/train', methods=['GET', 'POST'])
def train_model():
    try:
        trainer_inst = ParkingModelTrainer()

        # Prepare historical data
        df = trainer_inst.prepare_data('birmingham')

        # Train and evaluate available models
        results = trainer_inst.train_models(df)

        # Save the Random Forest model and its feature list
        model_path = trainer_inst.save_best_model('all')

        if model_path is None:
            raise RuntimeError(
                'The Random Forest model could not be saved.'
            )

        # Persist evaluation metrics for the dashboard
        with open(
            METRICS_PATH, 'w', encoding='utf-8'
        ) as file:
            json.dump(results, file, indent=4)

        flash(
            f'Model trained successfully and saved to {model_path}',
            'success'
        )

    except Exception as e:
        app.logger.exception('Model training failed')
        flash(f'Training failed: {str(e)}', 'danger')

    return redirect(url_for('dashboard'))


# --------------------------------------------------
# About page
# --------------------------------------------------

@app.route('/about')
def about():
    return render_template('about.html')


# --------------------------------------------------
# Initialize database and sample facilities
# --------------------------------------------------

@app.route('/init_db')
def init_db():
    try:
        db.create_all()

        if ParkingFacility.query.count() == 0:
            facilities = [
                ParkingFacility(
                    facility_id='BHMBCCMKT01',
                    name='Bull Ring Main Car Park',
                    capacity=577,
                    parking_fee=50,
                    walking_time_min=2,
                    travel_time_min=10
                ),
                ParkingFacility(
                    facility_id='BHMBRCMP01',
                    name='Bull Ring Centre Parking',
                    capacity=387,
                    parking_fee=40,
                    walking_time_min=3,
                    travel_time_min=12
                ),
                ParkingFacility(
                    facility_id='BHMBRTARC01',
                    name='Ratcliff Terrace Car Park',
                    capacity=293,
                    parking_fee=30,
                    walking_time_min=7,
                    travel_time_min=15
                )
            ]

            db.session.add_all(facilities)
            db.session.commit()

        flash(
            'Database initialized with sample facilities!',
            'success'
        )

    except Exception as e:
        db.session.rollback()
        app.logger.exception('Database initialization failed')
        flash(f'Database initialization failed: {str(e)}', 'danger')

    return redirect(url_for('index'))


# --------------------------------------------------
# Application entry point
# --------------------------------------------------

if __name__ == '__main__':
    with app.app_context():
        db.create_all()

    app.run(
        debug=True,
        host='127.0.0.1',
        port=5000
    )
