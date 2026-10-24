import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'parkahead-dev-secret-key'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///' + os.path.join(BASE_DIR, 'parkahead.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = DATA_RAW
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
