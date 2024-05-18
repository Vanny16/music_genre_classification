from fastapi import FastAPI, File, UploadFile
from classify_music import classify_music  # Import your function for classifying music
import os
from fastapi.middleware.cors import CORSMiddleware
import shutil

app = FastAPI()

# Allow CORS from specific origins for enhanced security
origins = [
    "http://localhost:3000",
    "https://optimizedcnn.vercel.app",
    "null"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,  # Allow cookies for session management if needed
    allow_methods=["*"],  # Allow all HTTP methods (adjust as required)
    allow_headers=["*"],  # Allow all headers (adjust as required)
)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
@app.post("/classify_music/")
async def classify_music_post(file: UploadFile = File(...)):
    """Classifies uploaded music using the classify_music function.

    Handles potential errors and returns appropriate responses.
    """

    try:
        # Validate file type (optional, adapt to your needs)
        if not file.content_type.lower().startswith("audio/"):
            return {"error": "Invalid file type. Please upload an audio file."}

        # Create a temporary file with a secure, unique name
        temp_filename = f"temp_{file.filename}"
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Call the classify_music function with the temporary file path
        result = classify_music(temp_filename)

        # Remove the temporary file (consider using a temporary directory for more robustness)
        os.remove(temp_filename)

        return result

    except Exception as e:
        return {"error": str(e)}
