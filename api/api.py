# Importing Required Libraries
from fastapi import FastAPI, File, UploadFile
from inference_sdk import InferenceHTTPClient
import os
import uvicorn
import openai

from openai import OpenAI
client = OpenAI()

# Set OpenAI API key
openai_api_key = os.environ["OPENAI_API_KEY"]

# System Prompt for the chatbot
system_prompt = os.environ ["SYSTEM_PROMPT"]

# Care tips prompt
care_tips_prompt_system = os.environ ["CARE_TIPS_PROMPT_SYSTEM"]
 
 # Care tips prompt
care_tips_prompt_user = os.environ ["CARE_TIPS_PROMPT_USER"]

# Variable if not recognized
generic_plant = os.environ ["GENERIC_PLANT_NAME"]

# Initialize the OpenAI API client
openai.api_key = openai_api_key

os.makedirs("raw_data", exist_ok=True)

# Initializing the FastAPI App
app = FastAPI()

# Initializing the Roboflow API Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=os.environ["ROBOFLOW_API_KEY"]
)

# Function to perform inference on a local image
def inference_local_image(image_path: str):
    result = CLIENT.infer(image_path, model_id="houseplants-image-detection/1")
    # Extract plant name from the result (assumption)
    plant_name = result['predictions'][0]['class'] if result['predictions'] else generic_plant
    return plant_name

# Function to get care tips using OpenAI GPT-3
def get_care_tips(plant_class):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
                messages=[
            {"role": "system", "content": care_tips_prompt_system.format(plant_class = plant_class)},
            {"role": "user", "content": care_tips_prompt_user.format(plant_class = plant_class)},
        ],
        temperature=0.7,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    return completion.choices[0].message.content

# Function to handle user questions using OpenAI GPT-3
def handle_user_question(plant_class, user_question):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": system_prompt.format(plant_class = plant_class)},
            {"role": "user", "content": f"{user_question}"}
        ],
        temperature=0.7,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    return completion.choices[0].message.content

# API Endpoints 
@app.post("/prediction")  # Method: POST
async def prediction(file: UploadFile = File(...)):
    file_location = f"raw_data/_{file.filename}"
    with open(file_location, "wb") as f:
        f.write(file.file.read())

    plant = inference_local_image(file_location)
    if plant=="generic plant":
        return {"text": "No care tips available, follow your heart. 💚", "plant": plant}
    
    care_tips = get_care_tips(plant)
    return {"text": care_tips, "plant": plant}  # Returns the care tips as a JSON response.

# chat Endpoint
@app.get("/chat")  # Method: GET
async def chat(plant_class: str, user_prompt: str):
    response = handle_user_question(plant_class, user_prompt)
    return {"response": response} # Returns the response as a JSON response.

@app.get("/")
def home():
    return {"message": ""}
# Running the Application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
