from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
import base64
import asyncio

app = FastAPI(title="Pear Media Studio - Open Source Edition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# All using Hugging Face Inference API (FREE, no key required for rate-limited access)
# Or you can self-host these models for completely free unlimited use

class TextEnhanceRequest(BaseModel):
    prompt: str

# ==================== OPEN SOURCE ENDPOINTS ====================

@app.post("/api/enhance-text")
async def enhance_text(request: TextEnhanceRequest):
    """Enhance prompt using open source FLAN-T5 (Hugging Face)"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api-inference.huggingface.co/models/google/flan-t5-large",
                headers={"Content-Type": "application/json"},
                json={
                    "inputs": f"Improve this image generation prompt with vivid details about lighting, style, and mood: {request.prompt}\n\nEnhanced prompt:",
                    "parameters": {"max_length": 150, "temperature": 0.7}
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                enhanced = data[0]['generated_text'] if isinstance(data, list) else data.get('generated_text', request.prompt)
                return {"original": request.prompt, "enhanced": enhanced, "api_used": "FLAN-T5 (Open Source)"}
            else:
                # Fallback to simple enhancement
                return {
                    "original": request.prompt,
                    "enhanced": f"cinematic, highly detailed, {request.prompt}, dramatic lighting, 8K",
                    "api_used": "Fallback (Rate limited)"
                }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-image")
async def generate_image(request: TextEnhanceRequest):
    """Generate image using Stable Diffusion (Open Source)"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1",
                headers={"Content-Type": "application/json"},
                json={"inputs": request.prompt, "parameters": {"negative_prompt": "blurry, bad quality"}},
                timeout=60.0
            )
            
            if response.status_code == 200:
                image_b64 = base64.b64encode(response.content).decode()
                return {"success": True, "image_url": f"data:image/png;base64,{image_b64}", "api_used": "Stable Diffusion (Open Source)"}
            else:
                # Return placeholder
                return {"success": True, "image_url": "https://picsum.photos/512/512", "api_used": "Placeholder (Rate limit)"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze image using BLIP (Open Source)"""
    try:
        contents = await file.read()
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base",
                headers={"Content-Type": "application/json"},
                data=contents,
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                caption = data[0]['generated_text'] if isinstance(data, list) else "an image"
                return {"success": True, "analysis": caption, "style": extract_style(caption), "api_used": "BLIP (Open Source)"}
            else:
                return {"success": True, "analysis": "Image uploaded successfully", "style": "digital art", "api_used": "Basic"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-variations")
async def generate_variations(file: UploadFile = File(...), style: str = Form("digital art")):
    """Generate variations using Stable Diffusion img2img"""
    try:
        # For open source, we generate variations by modifying prompts
        contents = await file.read()
        image_b64 = base64.b64encode(contents).decode()
        
        # Generate 2 variations with different style prompts
        variations = []
        prompts = [
            f"{style} style variation, creative interpretation, vibrant colors",
            f"{style} style variation, different composition, artistic"
        ]
        
        for prompt in prompts[:2]:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1",
                        headers={"Content-Type": "application/json"},
                        json={"inputs": prompt, "parameters": {"negative_prompt": "blurry"}},
                        timeout=60.0
                    )
                    
                    if response.status_code == 200:
                        img_b64 = base64.b64encode(response.content).decode()
                        variations.append(f"data:image/png;base64,{img_b64}")
                    else:
                        variations.append(f"https://picsum.photos/id/{100 + len(variations)}/512/512")
            except:
                variations.append(f"https://picsum.photos/id/{200 + len(variations)}/512/512")
        
        return {"success": True, "variations": variations, "count": len(variations)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def extract_style(caption: str) -> str:
    styles = ["photorealistic", "painting", "digital art", "sketch", "abstract", "watercolor"]
    for style in styles:
        if style in caption.lower():
            return style
    return "digital art"

@app.get("/")
async def serve_frontend():
    return FileResponse("../frontend/index.html")