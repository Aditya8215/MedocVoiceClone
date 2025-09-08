import cloudinary
import cloudinary.uploader
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
CLOUD_NAME = os.getenv('CLOUDINARY_CLOUD_NAME')
API_KEY = os.getenv('CLOUDINARY_API_KEY')
API_SECRET = os.getenv('CLOUDINARY_API_SECRET')

# Configure Cloudinary
cloudinary.config(
    cloud_name=CLOUD_NAME,
    api_key=API_KEY,
    api_secret=API_SECRET
)

def upload_audio_to_cloudinary(audio_path, public_id=None):
    """
    Uploads an audio file to Cloudinary and returns the URL and public_id.
    """
    try:
        response = cloudinary.uploader.upload(
            audio_path,
            resource_type="video",  # Cloudinary treats audio as video
            public_id=public_id,
            overwrite=True
        )
        return {
            "url": response.get("secure_url"),
            "public_id": response.get("public_id")
        }
    except Exception as e:
        return {"error": str(e)}

def delete_audio_from_cloudinary(public_id):
    """
    Deletes an audio file from Cloudinary using its public_id.
    """
    try:
        response = cloudinary.uploader.destroy(
            public_id,
            resource_type="video"
        )
        return response
    except Exception as e:
        return {"error": str(e)}
