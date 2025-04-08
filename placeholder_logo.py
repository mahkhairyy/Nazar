import streamlit as st
from PIL import Image, ImageDraw
import io

# Create a simple red eye logo
def create_eye_logo():
    # Create a new image with white background
    img = Image.new('RGBA', (200, 200), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw the eye outline (red circle)
    draw.ellipse((50, 50, 150, 150), outline=(230, 57, 70), width=5)
    
    # Draw the pupil (filled red circle)
    draw.ellipse((85, 85, 115, 115), fill=(230, 57, 70))
    
    # Save to a bytes buffer
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()

# Function to get the logo as bytes
def get_logo_bytes():
    return create_eye_logo()
