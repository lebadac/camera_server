from google import genai
import os
from typing import Optional
from config import GEMINI_API_KEY
from PIL import Image

# Initialize Gemini client
client = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        print("🤖 Gemini AI initialized successfully.")
    except Exception as e:
        print(f"⚠️ Gemini AI initialization failed: {e}")

def analyze_fire_context(image_path: str) -> Optional[str]:
    """
    Analyzes fire alert image using Gemini AI to provide brief context.
    
    Args:
        image_path (str): Path to the fire alert image
        
    Returns:
        Optional[str]: Brief context string (e.g., "High severity. Flammables nearby.")
                      Returns None if Gemini is not configured or analysis fails.
    """
    if not client or not os.path.exists(image_path):
        return None
    
    try:
        # Open image with PIL
        image = Image.open(image_path)
        
        # Minimal prompt for low token usage
        prompt = """Analyze this fire image briefly (max 10 words):
1. Severity: low/medium/high
2. Flammables nearby: yes/no

Format: "Severity: X. Flammables: Y."
Example: "Severity: High. Flammables: Yes (curtains)." """
        
        # Generate response with new SDK (pass image directly)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, image],
        )
        
        # Extract and clean response
        context = response.text.strip()
        
        # Limit to 50 characters max
        if len(context) > 50:
            context = context[:47] + "..."
        
        return context
    
    except Exception as e:
        print(f"❌ Gemini analysis failed: {e}")
        return None
