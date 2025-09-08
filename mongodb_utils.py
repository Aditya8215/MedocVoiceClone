import os
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = os.getenv('MONGO_DB_NAME', 'medoc_voice')
COLLECTION_NAME = os.getenv('MONGO_COLLECTION_NAME', 'prescriptions')

# MongoDB client setup
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

def upload_prescription_to_mongodb(prescription_data=None, audio_url=None,original_script=None,transcription=None,feedback_rating=None,feedback_text=None):
    """
    Uploads prescription data and audio URL to MongoDB.
    Returns the inserted document ID or error.
    """
    try:
        doc = {
            "prescription": prescription_data,
            "audio_url": audio_url,
            "created_at": datetime.now(),
            "Original_Script":  original_script,
            "Transcription": transcription,
            "feedback_rating": feedback_rating,
            "feedback_text": feedback_text
        }
        result = collection.insert_one(doc)
        return {"inserted_id": str(result.inserted_id)}
    except Exception as e:
        return {"error": str(e)}
