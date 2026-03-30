from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv
import base64
from io import BytesIO
from PIL import Image
import uuid

load_dotenv()

app = FastAPI(title="Pear Media Studio API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys (set these in .env)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY")

# Models
class TextEnhanceRequest(BaseModel):
    prompt: str

class TextEnhanceResponse(BaseModel):
    original: str
    enhanced: str
    api_used: str

class ImageGenerationRequest(BaseModel):
    prompt: str

# ==================== TEXT ENHANCEMENT APIs ====================

@app.post("/api/enhance-text", response_model=TextEnhanceResponse)
async def enhance_text(request: TextEnhanceRequest):
    """
    Enhance text prompt using NLP API (OpenAI or Cohere)
    """
    try:
        # Try OpenAI first if key exists
        if OPENAI_API_KEY:
            enhanced = await enhance_with_openai(request.prompt)
            return TextEnhanceResponse(
                original=request.prompt,
                enhanced=enhanced,
                api_used="OpenAI GPT"
            )
        # Fallback to Cohere
        elif COHERE_API_KEY:
            enhanced = await enhance_with_cohere(request.prompt)
            return TextEnhanceResponse(
                original=request.prompt,
                enhanced=enhanced,
                api_used="Cohere"
            )
        else:
            raise HTTPException(status_code=400, detail="No NLP API keys configured")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {str(e)}")

async def enhance_with_openai(prompt: str) -> str:
    """Enhance prompt using OpenAI GPT"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a prompt engineer. Enhance the user's image generation prompt with vivid details about style, lighting, composition, and mood. Only output the enhanced prompt, no explanations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 150
            }
        )
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

async def enhance_with_cohere(prompt: str) -> str:
    """Enhance prompt using Cohere"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.cohere.ai/v1/generate",
            headers={
                "Authorization": f"Bearer {COHERE_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "command",
                "prompt": f"Enhance this image generation prompt with rich details about style, lighting, composition: {prompt}\n\nEnhanced prompt:",
                "max_tokens": 100,
                "temperature": 0.8
            }
        )
        data = response.json()
        return data["generations"][0]["text"].strip()

# ==================== IMAGE GENERATION APIs ====================

@app.post("/api/generate-image")
async def generate_image(request: ImageGenerationRequest):
    """
    Generate image from prompt using Stability AI or Replicate
    """
    try:
        if STABILITY_API_KEY:
            image_url = await generate_with_stability(request.prompt)
            return {"success": True, "image_url": image_url, "api_used": "Stability AI"}
        elif REPLICATE_API_KEY:
            image_url = await generate_with_replicate(request.prompt)
            return {"success": True, "image_url": image_url, "api_used": "Replicate"}
        else:
            raise HTTPException(status_code=400, detail="No image generation API keys configured")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

async def generate_with_stability(prompt: str) -> str:
    """Generate image using Stability AI API"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
            headers={
                "Authorization": f"Bearer {STABILITY_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "text_prompts": [{"text": prompt, "weight": 1}],
                "cfg_scale": 7,
                "height": 512,
                "width": 512,
                "samples": 1,
                "steps": 30
            },
            timeout=30.0
        )
        data = response.json()
        # Return base64 image
        return f"data:image/png;base64,{data['artifacts'][0]['base64']}"

async def generate_with_replicate(prompt: str) -> str:
    """Generate image using Replicate API"""
    async with httpx.AsyncClient() as client:
        # Start prediction
        response = await client.post(
            "https://api.replicate.com/v1/predictions",
            headers={
                "Authorization": f"Token {REPLICATE_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "version": "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                "input": {
                    "prompt": prompt,
                    "width": 512,
                    "height": 512,
                    "num_outputs": 1
                }
            }
        )
        data = response.json()
        prediction_url = data["urls"]["get"]
        
        # Poll for result
        import time
        for _ in range(30):  # 30 seconds timeout
            time.sleep(1)
            status_response = await client.get(
                prediction_url,
                headers={"Authorization": f"Token {REPLICATE_API_KEY}"}
            )
            status_data = status_response.json()
            if status_data["status"] == "succeeded":
                return status_data["output"][0]
            elif status_data["status"] == "failed":
                raise Exception("Replicate generation failed")
        
        raise Exception("Timeout waiting for image generation")

# ==================== IMAGE ANALYSIS & VARIATION APIs ====================

@app.post("/api/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze uploaded image using Replicate or Hugging Face
    """
    try:
        # Read image
        contents = await file.read()
        image_b64 = base64.b64encode(contents).decode()
        
        # Use Replicate's BLIP model for image captioning
        if REPLICATE_API_KEY:
            caption = await analyze_with_replicate(image_b64)
            return {
                "success": True,
                "analysis": caption,
                "style": extract_style(caption),
                "api_used": "Replicate BLIP"
            }
        else:
            # Fallback to simple analysis
            return {
                "success": True,
                "analysis": "Image analyzed successfully",
                "style": "digital art",
                "api_used": "Basic analysis"
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

async def analyze_with_replicate(image_b64: str) -> str:
    """Analyze image using Replicate's BLIP model"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.replicate.com/v1/predictions",
            headers={
                "Authorization": f"Token {REPLICATE_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "version": "salesforce/blip:2e1dddc8621f72155f24cf2e0adbde5484103cd9ad6f86a2457b43f32b2c6a4f",
                "input": {
                    "image": f"data:image/jpeg;base64,{image_b64}",
                    "task": "image_captioning"
                }
            }
        )
        data = response.json()
        prediction_url = data["urls"]["get"]
        
        import time
        for _ in range(20):
            time.sleep(1)
            status_response = await client.get(
                prediction_url,
                headers={"Authorization": f"Token {REPLICATE_API_KEY}"}
            )
            status_data = status_response.json()
            if status_data["status"] == "succeeded":
                return status_data["output"][0]
        
        return "Unable to generate caption"

def extract_style(caption: str) -> str:
    """Extract style keywords from caption"""
    styles = ["photorealistic", "painting", "digital art", "sketch", "abstract"]
    for style in styles:
        if style in caption.lower():
            return style
    return "digital art"

@app.post("/api/generate-variations")
async def generate_variations(
    file: UploadFile = File(...),
    style: str = Form("digital art")
):
    """
    Generate image variations using Stability AI or Replicate img2img
    """
    try:
        contents = await file.read()
        image_b64 = base64.b64encode(contents).decode()
        
        variations = []
        
        if STABILITY_API_KEY:
            # Generate 2 variations with different prompts
            prompts = [
                f"{style} style variation, creative interpretation",
                f"{style} style variation, different color palette, artistic"
            ]
            
            for i, prompt in enumerate(prompts[:2]):
                variation = await generate_variation_stability(image_b64, prompt)
                variations.append(variation)
        
        elif REPLICATE_API_KEY:
            # Use Replicate's img2img
            for i in range(2):
                variation = await generate_variation_replicate(image_b64)
                variations.append(variation)
        
        return {
            "success": True,
            "variations": variations,
            "count": len(variations)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Variation generation failed: {str(e)}")

async def generate_variation_stability(image_b64: str, prompt: str) -> str:
    """Generate variation using Stability AI img2img"""
    async with httpx.AsyncClient() as client:
        # Prepare multipart form data
        files = {
            "init_image": (None, base64.b64decode(image_b64), "image/png"),
            "init_image_mode": (None, "IMAGE_STRENGTH"),
            "image_strength": (None, "0.5"),
            "text_prompts[0][text]": (None, prompt),
            "cfg_scale": (None, "7"),
            "samples": (None, "1"),
            "steps": (None, "30")
        }
        
        response = await client.post(
            "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/image-to-image",
            headers={"Authorization": f"Bearer {STABILITY_API_KEY}"},
            files=files,
            timeout=30.0
        )
        data = response.json()
        return f"data:image/png;base64,{data['artifacts'][0]['base64']}"

async def generate_variation_replicate(image_b64: str) -> str:
    """Generate variation using Replicate"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.replicate.com/v1/predictions",
            headers={
                "Authorization": f"Token {REPLICATE_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "version": "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                "input": {
                    "image": f"data:image/jpeg;base64,{image_b64}",
                    "prompt": "variation, creative interpretation, artistic style",
                    "width": 512,
                    "height": 512
                }
            }
        )
        data = response.json()
        prediction_url = data["urls"]["get"]
        
        import time
        for _ in range(30):
            time.sleep(1)
            status_response = await client.get(
                prediction_url,
                headers={"Authorization": f"Token {REPLICATE_API_KEY}"}
            )
            status_data = status_response.json()
            if status_data["status"] == "succeeded":
                return status_data["output"][0]
        
        raise Exception("Timeout")

# ==================== SERVE FRONTEND ====================

@app.get("/")
async def serve_frontend():
    return FileResponse("../frontend/index.html")

# Mount static files if needed
# app.mount("/static", StaticFiles(directory="../frontend"), name="static")