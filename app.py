import streamlit as st
import torch
import pandas as pd
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
import base64
from PIL import Image
import io
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Config
st.set_page_config(
    page_title="Toxic Comment Classifier",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Logo is now displayed in the header

# Load model
MODEL_PATH = "./saved_model"
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Load and encode logo
def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Function to remove background from image
def remove_background(image_path, threshold_value=240):
    try:
        # Open the image
        img = Image.open(image_path)

        # Convert to RGBA if it's not already
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        # Create a new image with transparent background
        datas = img.getdata()
        new_data = []

        # More aggressive background removal
        for item in datas:
            # Check if the pixel is close to white or light gray
            r, g, b = item[0], item[1], item[2]

            # Calculate how "white" the pixel is
            whiteness = (r + g + b) / 3

            # If the pixel is whitish or light gray, make it transparent
            if whiteness > threshold_value:
                new_data.append((255, 255, 255, 0))  # Transparent
            # Also make light gray pixels transparent
            elif abs(r - g) < 20 and abs(r - b) < 20 and abs(g - b) < 20 and whiteness > threshold_value - 40:
                new_data.append((255, 255, 255, 0))  # Transparent
            else:
                if len(item) == 4:  # If the image already has an alpha channel
                    new_data.append(item)
                else:  # If the image doesn't have an alpha channel
                    new_data.append(item + (255,))  # Add full opacity

        img.putdata(new_data)

        # Save to a BytesIO object
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")

        # Return base64 encoded string
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        st.error(f"Error processing image with threshold {threshold_value}: {e}")
        # Fall back to regular encoding
        return get_base64_encoded_image(image_path)

# Simplify logo loading to fix missing logo issue
try:
    # Just use the basic encoding without background removal
    logo_base64 = get_base64_encoded_image("logo.png")
    print(f"Logo loaded successfully, length: {len(logo_base64)}")
except Exception as e:
    st.error(f"Error loading logo: {e}")
    logo_base64 = ""

# Session state
if "history" not in st.session_state:
    st.session_state.history = []

# Load the logo image
try:
    logo_image = Image.open('logo.png')
    # Store the logo image in session state for later use
    st.session_state.logo_image = logo_image
except Exception as e:
    st.error(f"Error loading logo: {e}")
    st.session_state.logo_image = None

if "uploaded_results" not in st.session_state:
    st.session_state.uploaded_results = {}

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "single_comment"

# Add meta viewport tag, SVG filter, theme toggle, and toast notification
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">

<!-- SVG filter to remove white background -->
<svg width="0" height="0" style="position: absolute;">
  <filter id="remove-white" color-interpolation-filters="sRGB">
    <feColorMatrix type="matrix" values="1 0 0 0 0
                                         0 1 0 0 0
                                         0 0 1 0 0
                                         0 0 0 30 -15" />
  </filter>
</svg>

<!-- Custom Header inspired by index.html -->
<header class="custom-header" id="custom-header">
  <div class="header-logo">
    <div class="eye-logo">
      <div class="eye-outer">
        <div class="eye-inner"></div>
      </div>
    </div>
  </div>
  <nav class="nav-links">
    <a href="#" onclick="window.parent.postMessage({type: 'streamlit:setStateValue', key: 'active_tab', value: 'single_comment'}, '*');">Classify</a>
    <a href="#" onclick="window.parent.postMessage({type: 'streamlit:setStateValue', key: 'active_tab', value: 'upload_csv'}, '*');">Upload</a>
    <a href="#" onclick="window.parent.postMessage({type: 'streamlit:setStateValue', key: 'active_tab', value: 'history'}, '*');">History</a>
    <a href="#" onclick="window.parent.postMessage({type: 'streamlit:setStateValue', key: 'active_tab', value: 'about'}, '*');">About</a>
  </nav>
</header>

<!-- Toast Notification Container -->
<div id="toast" class="toast"></div>

<!-- Logo loading script removed -->

<script>
  // Theme toggle functionality removed as requested
  document.addEventListener('DOMContentLoaded', function() {
    // Set default light theme
    const root = document.documentElement;
    root.style.setProperty('--background-color', '#f9f9f9');
    root.style.setProperty('--card-background', '#ffffff');
    root.style.setProperty('--text-color', '#333333');
    root.style.setProperty('--border-color', '#e5e7eb');
    root.style.setProperty('--accent-color', '#6b7280');
    root.style.setProperty('--shadow-color', 'rgba(0, 0, 0, 0.1)');
    document.body.classList.add('light-theme');

  // Toast notification function - defined at global scope
  window.showToast = function(message, duration = 3000) {
    const toast = document.getElementById('toast');
    if (!toast) {
      console.error('Toast element not found');
      return;
    }
    toast.textContent = message;
    toast.classList.add('show');

    setTimeout(function() {
      toast.classList.remove('show');
    }, duration);
  };
});
</script>
""", unsafe_allow_html=True)

# Custom CSS for modern design with layout improvements
st.markdown("""
<style>
    /* Base styles with modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Tenor+Sans&family=Cormorant:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&family=Poppins:wght@300;400;500;600;700&family=Noto+Naskh+Arabic:wght@400;700&display=swap');

    :root {
        --background-color: #f9f9f9;
        --card-background: #ffffff;
        --text-color: #333333;
        --accent-color: #6b7280;
        --border-color: #e5e7eb;
        --highlight-color: #4b5563;
        --shadow-color: rgba(0, 0, 0, 0.1);
        --primary-color: #e11d48;
    }

    /* Modern typography */
    body, .stMarkdown, p, div {
        font-family: 'Inter', sans-serif !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
        color: var(--text-color);
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
        color: var(--text-color);
        margin-bottom: 1rem !important;
    }

    /* Soft background */
    body, .stApp, [data-testid="stAppViewContainer"] {
        background-color: var(--background-color) !important;
    }

    /* Center header section */
    header {
        text-align: center !important;
        margin-top: 40px !important;
        margin-bottom: 30px !important;
    }

    [data-testid="stAppViewContainer"] {
        background-color: var(--background-color);
    }

    /* Card styling */
    .card {
        background: var(--card-background);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 8px var(--shadow-color);
        margin-bottom: 20px;
        border: 1px solid var(--border-color);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px var(--shadow-color);
    }

    /* Confidence bar animation */
    .confidence-bar {
        transition: width 0.5s ease, background-color 0.3s ease;
        height: 8px;
        border-radius: 4px;
    }

    /* Toast notification */
    .toast {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: #333;
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 9999;
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .toast.show {
        opacity: 1;
    }

    /* Footer styling */
    .footer {
        text-align: center;
        padding: 20px 0;
        margin-top: 40px;
        border-top: 1px solid var(--border-color);
        color: var(--accent-color);
        font-size: 14px !important;
    }

    /* Theme toggle styles removed as requested */

    /* Responsive improvements */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem !important;
        }

        .card {
            padding: 15px;
        }
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }

    /* Center all content */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }

    /* Remove padding from app container */
    .appview-container {
        padding-left: 0 !important;
        padding-right: 0 !important;
    }

    /* Center all elements */
    .stApp > header, .stApp > .main, .stApp > footer {
        max-width: 1200px;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    /* Full width background */
    html, body {
        width: 100%;
        max-width: 100%;
        overflow-x: hidden;
        background-color: var(--background-color) !important;
        transition: background-color 0.3s ease, color 0.3s ease;
    }

    /* Background colors based on theme */
    .light-theme div, .light-theme section, .light-theme main, .light-theme header, .light-theme footer, .light-theme aside, .light-theme nav {
        background-color: var(--background-color) !important;
    }

    .dark-theme div, .dark-theme section, .dark-theme main, .dark-theme header, .dark-theme footer, .dark-theme aside, .dark-theme nav {
        background-color: var(--background-color) !important;
    }

    /* Remove any potential margins causing black space */
    body {
        margin: 0 !important;
        padding: 0 !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Cormorant', serif;
        font-weight: 700;
        color: var(--text-color);
    }

    p, li, div {
        font-family: 'Tenor Sans', sans-serif;
        color: var(--text-color);
        font-weight: 400;
        letter-spacing: 0.02em;
    }

    /* Hide Streamlit elements */
    #MainMenu, footer {
        visibility: hidden;
    }

    /* Custom header styling from index.html */
    .custom-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.8rem 2rem;
        background-color: #ffffff;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        margin-bottom: 2rem;
        border-radius: 0 0 10px 10px;
        position: relative;
        z-index: 1000;
    }

    .header-logo {
        display: flex;
        align-items: center;
    }

    /* Eye logo styling */
    .eye-logo {
        width: 80px;
        height: 80px;
        margin-right: 20px;
        animation: pulse 2s infinite ease-in-out;
        filter: drop-shadow(0 3px 6px rgba(0,0,0,0.15));
    }

    .eye-outer {
        width: 100%;
        height: 100%;
        background-color: #e63946;
        border-radius: 50%;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .eye-inner {
        width: 50%;
        height: 50%;
        background-color: white;
        border-radius: 50%;
        display: flex;
        justify-content: center;
        align-items: center;
        position: relative;
    }

    .eye-inner::after {
        content: "";
        width: 50%;
        height: 50%;
        background-color: #e63946;
        border-radius: 50%;
        position: absolute;
    }

    .logo {
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }

    .logo-icon-container {
        width: 45px;
        height: 45px;
        margin-right: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .logo-icon {
        width: 100%;
        height: 100%;
        object-fit: contain;
        animation: pulse 2s infinite ease-in-out;
        filter: drop-shadow(0 2px 5px rgba(230, 57, 70, 0.3));
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }

    .logo-text {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
    }

    .logo h2 {
        margin: 0;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        color: #e63946;
        font-size: 1.8rem;
        line-height: 1.2;
        letter-spacing: 0.5px;
    }

    .arabic-text {
        font-family: 'Noto Naskh Arabic', serif;
        font-size: 1.6rem;
        color: #333;
        margin-top: 0px;
        opacity: 0.9;
        font-weight: 700;
    }

    .nav-links {
        display: flex;
        gap: 1.5rem;
    }

    .nav-links a {
        text-decoration: none;
        color: #222;
        font-weight: 600;
        transition: color 0.2s;
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
    }

    .nav-links a:hover {
        color: #e63946;
    }

    @media (max-width: 768px) {
        .custom-header {
            flex-direction: column;
            padding: 1rem;
            gap: 1rem;
        }

        .nav-links {
            width: 100%;
            justify-content: space-between;
        }
    }

    div[data-testid="stToolbar"] {
        visibility: hidden;
    }

    div[data-testid="stDecoration"] {
        visibility: hidden;
    }

    div[data-testid="stStatusWidget"] {
        visibility: hidden;
    }

    section[data-testid="stSidebar"] {
        visibility: hidden;
        width: 0px;
    }

    /* Navigation */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 0;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 2rem;
    }

    .navbar-brand {
        font-family: 'Tenor Sans', sans-serif;
        font-weight: 400;
        font-size: 1.2rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-color);
        text-decoration: none;
        display: flex;
        align-items: center;
    }

    .logo {
        height: 40px;
        width: auto;
        margin-right: 1rem;
        background-color: transparent !important;
        mix-blend-mode: multiply; /* This helps with white backgrounds */
        filter: url(#remove-white) drop-shadow(0 0 0 transparent); /* Apply SVG filter */
        -webkit-filter: drop-shadow(0 0 0 transparent);
    }

    .navbar-links {
        display: flex;
        gap: 2rem;
    }

    .navbar-link {
        font-family: 'Tenor Sans', sans-serif;
        font-weight: 400;
        font-size: 0.9rem;
        color: var(--text-color);
        text-decoration: none;
        position: relative;
        letter-spacing: 0.05em;
    }

    .navbar-link.active {
        font-weight: 500;
    }

    .navbar-link.active:after {
        content: "";
        position: absolute;
        bottom: -5px;
        left: 0;
        width: 100%;
        height: 1px;
        background-color: var(--text-color);
    }

    /* Hero section - inspired by index.html */
    .hero {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        padding: 4rem 2rem 2rem;
        background: linear-gradient(180deg, #f8fafc, #fff);
        border-radius: 12px;
        margin-bottom: 3rem;
    }

    .hero-title {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 3rem;
        line-height: 1.1;
        margin-bottom: 1rem;
        margin-top: 0;
        color: #333;
        animation: slideInFromRight 1.5s ease-out;
    }

    @keyframes slideInFromRight {
        0% {
            opacity: 0;
            transform: translateX(100px);
        }
        100% {
            opacity: 1;
            transform: translateX(0);
        }
    }

    .hero-description {
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        font-size: 1.2rem;
        color: #555;
        max-width: 600px;
        margin: 0 auto 2rem auto;
        line-height: 1.6;
    }

    .hero-logo {
        width: auto;
        height: 100px;
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease;
        background-color: transparent !important;
        mix-blend-mode: multiply; /* This helps with white backgrounds */
        filter: url(#remove-white) drop-shadow(0 0 0 transparent); /* Apply SVG filter */
        -webkit-filter: drop-shadow(0 0 0 transparent);
    }

    .hero-logo:hover {
        transform: scale(1.05);
    }

    /* Divider */
    .divider {
        display: flex;
        align-items: center;
        margin: 3rem 0;
    }

    .divider-line {
        flex-grow: 1;
        height: 1px;
        background-color: var(--border-color);
    }

    .divider-text {
        padding: 0 1rem;
        font-family: 'Tenor Sans', sans-serif;
        font-weight: 400;
        font-size: 0.8rem;
        color: var(--accent-color);
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    /* Simple divider */
    .simple-divider {
        height: 1px;
        background-color: var(--border-color);
        margin: 3rem 0;
        width: 100%;
    }

    /* Content sections */
    .section {
        margin-bottom: 4rem;
    }

    .section-title {
        font-family: 'Cormorant', serif;
        font-weight: 600;
        font-size: 2rem;
        margin-bottom: 2rem;
        letter-spacing: -0.01em;
    }

    .section-content {
        font-family: 'Tenor Sans', sans-serif;
        font-weight: 400;
        font-size: 0.95rem;
        line-height: 1.7;
        letter-spacing: 0.02em;
    }

    .section-subtitle {
        font-family: 'Tenor Sans', sans-serif;
        font-weight: 400;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--accent-color);
        margin-bottom: 1rem;
    }

    /* Gallery/Grid */
    .gallery {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 2rem;
        margin-bottom: 3rem;
    }

    .gallery-item {
        position: relative;
    }

    .gallery-image {
        width: 100%;
        height: auto;
        object-fit: cover;
    }

    .gallery-caption {
        margin-top: 0.8rem;
        font-family: 'Tenor Sans', sans-serif;
        font-weight: 400;
        font-size: 0.9rem;
        line-height: 1.4;
    }

    .gallery-caption strong {
        display: block;
        font-family: 'Cormorant', serif;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.3rem;
    }

    /* Quote */
    .quote {
        font-family: 'Cormorant', serif;
        font-weight: 500;
        font-style: italic;
        font-size: 2rem;
        line-height: 1.4;
        text-align: center;
        max-width: 800px;
        margin: 5rem auto;
    }

    .quote-attribution {
        font-family: 'Tenor Sans', sans-serif;
        font-weight: 400;
        font-size: 0.9rem;
        text-align: center;
        margin-top: 1.5rem;
        color: var(--accent-color);
        letter-spacing: 0.05em;
    }

    /* CTA */
    .cta {
        text-align: center;
        padding: 4rem 0;
    }

    /* Buttons styling from index.html */
    .buttons {
        margin-top: 2rem;
        display: flex;
        gap: 1rem;
        justify-content: center;
    }

    .buttons a {
        padding: 0.8rem 1.6rem;
        font-size: 1rem;
        font-weight: 600;
        color: white;
        background-color: #e63946;
        border-radius: 8px;
        text-decoration: none;
        transition: background 0.2s;
        display: inline-block;
    }

    .buttons a:hover {
        background-color: #c92d3c;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    .cta-title {
        font-family: 'Cormorant', serif;
        font-weight: 600;
        font-size: 3.5rem;
        margin-bottom: 2rem;
        letter-spacing: -0.02em;
    }

    .cta-button {
        display: inline-block;
        padding: 1rem 2.5rem;
        background-color: #ef4444;
        color: white;
        font-family: 'Tenor Sans', sans-serif;
        font-weight: 400;
        font-size: 0.9rem;
        text-decoration: none;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    .cta-button:hover {
        background-color: #dc2626;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        border-top: 1px solid var(--border-color);
        font-family: 'Tenor Sans', sans-serif;
        font-weight: 400;
        font-size: 0.8rem;
        color: var(--accent-color);
        letter-spacing: 0.05em;
    }

    /* Email signup */
    .email-signup {
        max-width: 500px;
        margin: 0 auto 3rem auto;
        text-align: center;
    }

    .email-signup-title {
        font-family: 'Tenor Sans', sans-serif;
        font-weight: 400;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
        color: var(--accent-color);
        letter-spacing: 0.1em;
    }

    .email-signup-title:before {
        content: '●';
        margin-right: 0.5rem;
        color: var(--accent-color);
    }

    /* Form styling */
    div[data-baseweb="input"], div[data-baseweb="textarea"] {
        background-color: white !important;
    }

    /* Override Streamlit's default form styling */
    .stTextInput input, .stTextArea textarea, .stSelectbox, .stMultiselect {
        background-color: white !important;
    }

    /* File uploader styling - target all possible class combinations */
    .st-emotion-cache-9ycgxx,
    [data-testid="stFileUploader"] span,
    .e17y52ym3,
    .st-emotion-cache-9ycgxx.e17y52ym3,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] div,
    [data-testid="stFileUploader"] label,
    .st-emotion-cache-9ycgxx e17y52ym3,
    div[data-testid="stFileUploader"] > div > div > span,
    div[data-testid="stFileUploader"] > div > div > div > span {
        color: white !important;
        fill: white !important; /* For SVG icons */
    }

    /* File uploader container */
    [data-testid="stFileUploader"] {
        background-color: rgba(239, 68, 68, 0.8) !important;
        border: 1px dashed white !important;
        border-radius: 4px;
        padding: 1rem;
        color: white !important;
    }

    /* Target all elements inside the file uploader */
    [data-testid="stFileUploader"] * {
        color: white !important;
    }

    /* Target the specific drag and drop text */
    [data-testid="stFileUploader"] .st-emotion-cache-9ycgxx {
        color: white !important;
        font-weight: 500;
    }

    div[data-baseweb="input"] input {
        background-color: white;
        border: 1px solid var(--border-color);
        border-radius: 0;
        font-family: 'Tenor Sans', sans-serif;
        font-weight: 400;
        padding: 1rem;
        font-size: 0.9rem;
        letter-spacing: 0.02em;
    }

    div[data-baseweb="textarea"] textarea {
        background-color: #fef2f2 !important;
        border: 1px solid #fecaca !important;
        border-radius: 0;
        font-family: 'Tenor Sans', sans-serif;
        font-weight: 400;
        padding: 1rem;
        font-size: 0.9rem;
        letter-spacing: 0.02em;
    }

    /* Additional styling for textarea */
    .stTextArea textarea, textarea, [data-testid="stTextArea"] textarea {
        background-color: #fef2f2 !important;
        color: var(--text-color) !important;
        border-color: #fecaca !important;
    }

    /* Target the specific textarea classes */
    .st-ba, .st-bv, .st-bw, .st-bx, .st-by, .st-bz, .st-c0, .st-c1, .st-c2, .st-c3, .st-c4, .st-b8, .st-c5, .st-c6, .st-c7, .st-c8, .st-c9, .st-ca, .st-cb, .st-cc, .st-ae, .st-af, .st-ag, .st-cd, .st-ai, .st-aj, .st-bu, .st-ce, .st-cf, .st-cg, .st-ch, .st-ci, .st-cj {
        background-color: #fef2f2 !important;
    }

    /* Direct textarea styling */
    textarea {
        background-color: #fef2f2 !important;
        background: #fef2f2 !important;
        border-color: #fecaca !important;
    }

    /* Force all form elements to have appropriate background */
    input, select, .stTextArea, [role="textbox"] {
        background-color: white !important;
        background: white !important;
    }

    /* Specific styling for textareas */
    textarea, .stTextArea textarea {
        background-color: #fef2f2 !important;
        background: #fef2f2 !important;
        border-color: #fecaca !important;
    }

    div[data-baseweb="input"] input:focus, div[data-baseweb="textarea"] textarea:focus {
        border-color: var(--text-color);
        box-shadow: none;
    }

    /* Form submit buttons */
    button[kind="secondaryFormSubmit"], .stFormSubmitButton button {
        background-color: #ef4444 !important;
        color: white !important;
        border-radius: 0;
        font-family: 'Tenor Sans', sans-serif;
        font-weight: 400;
        padding: 0.7rem 2rem;
        border: none;
        font-size: 0.9rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    button[kind="secondaryFormSubmit"]:hover, .stFormSubmitButton button:hover {
        background-color: #dc2626 !important;
    }

    /* Target the specific form submit button */
    .stFormSubmitButton button p, [data-testid="stFormSubmitButton"] button p, [data-testid="stBaseButton-secondaryFormSubmit"] p {
        color: white !important;
    }

    /* Services list */
    .services-list {
        list-style: none;
        padding: 0;
        margin: 2rem 0;
    }

    .service-item {
        padding: 1.5rem 0;
        border-top: 1px solid var(--border-color);
        font-family: 'Cormorant', serif;
        font-weight: 600;
        font-size: 1.5rem;
    }

    .service-item:last-child {
        border-bottom: 1px solid var(--border-color);
    }

    /* Tabs */
    .tabs {
        display: flex;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 2rem;
    }

    .tab {
        padding: 1rem 2rem;
        cursor: pointer;
        font-family: 'Tenor Sans', sans-serif;
        font-weight: 400;
        color: var(--accent-color);
        position: relative;
        transition: all 0.3s ease;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        font-size: 0.9rem;
    }

    .tab:hover {
        color: var(--text-color);
    }

    .tab.active {
        color: var(--text-color);
        font-weight: 500;
    }

    .tab.active:after {
        content: "";
        position: absolute;
        bottom: -1px;
        left: 0;
        width: 100%;
        height: 2px;
        background-color: var(--text-color);
    }

    /* Results styling */
    .result-card {
        padding: 1.5rem;
        border: 1px solid var(--border-color);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Result cards with theme support */
    .light-theme .result-toxic {
        border-left: 8px solid #ef4444;
        background-color: #fef2f2;
    }

    .light-theme .result-clean {
        border-left: 8px solid #10b981;
        background-color: #ecfdf5;
    }

    .dark-theme .result-toxic {
        border-left: 8px solid #ef4444;
        background-color: #3b1a1a;
        color: #f8f8f8;
    }

    .dark-theme .result-clean {
        border-left: 8px solid #10b981;
        background-color: #1a3b2a;
        color: #f8f8f8;
    }

    .dark-theme .result-content,
    .dark-theme .result-label,
    .dark-theme .result-confidence {
        color: #f8f8f8;
    }

    .result-icon {
        font-size: 3.5rem;
        margin-right: 1.5rem;
        font-weight: bold;
        width: 70px;
        height: 70px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        flex-shrink: 0;
    }

    .result-clean .result-icon {
        color: white;
        background-color: #10b981;
    }

    .result-toxic .result-icon {
        color: white;
        background-color: #ef4444;
    }

    .result-content {
        flex: 1;
    }

    .result-label {
        font-family: 'Tenor Sans', sans-serif;
        font-weight: 500;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }

    .result-confidence {
        font-family: 'Tenor Sans', sans-serif;
        font-weight: 300;
        font-size: 0.9rem;
        color: var(--accent-color);
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 3rem;
        }

        .hero-subtitle {
            font-size: 2rem;
        }

        .navbar-links {
            gap: 1rem;
        }

        .gallery {
            grid-template-columns: 1fr;
        }

        /* Mobile-friendly result cards */
        .result-card {
            flex-direction: column;
            text-align: center;
            padding: 1.2rem;
        }

        .result-icon {
            margin-right: 0;
            margin-bottom: 1rem;
        }

        /* Adjust form elements */
        textarea, .stTextArea textarea {
            min-height: 100px;
        }

        /* Adjust spacing */
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        /* Make buttons more tappable */
        button, .cta-button {
            padding: 0.8rem 1.5rem;
            min-height: 44px; /* Minimum touch target size */
        }

        /* Adjust service items */
        .service-item {
            font-size: 1.2rem;
            padding: 1.2rem 0;
        }
    }

    /* Small mobile devices */
    @media (max-width: 480px) {
        .hero-title {
            font-size: 2.5rem;
        }

        .hero-subtitle {
            font-size: 1.8rem;
            margin-bottom: 1.5rem;
        }

        .navbar {
            flex-direction: column;
            gap: 1rem;
            padding-bottom: 1rem;
        }

        .navbar-links {
            width: 100%;
            justify-content: space-between;
        }

        .logo {
            height: 30px;
            margin-right: 0.5rem;
        }

        .hero-logo {
            height: 60px;
            margin-bottom: 1rem;
        }

        .cta-title {
            font-size: 2.5rem;
        }

        /* Adjust tab buttons */
        [data-testid="baseButton-secondary"], [data-testid="baseButton-primary"] {
            font-size: 0.8rem !important;
            padding: 0.8rem 0.5rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Add JavaScript to detect mobile devices
st.markdown("""
<script>
    // Detect if the device is mobile
    function isMobile() {
        return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) || window.innerWidth < 768;
    }

    // Set the mobile_view state based on device detection
    if (isMobile()) {
        window.parent.postMessage({type: 'streamlit:setStateValue', key: 'mobile_view', value: true}, '*');
    } else {
        window.parent.postMessage({type: 'streamlit:setStateValue', key: 'mobile_view', value: false}, '*');
    }

    // Listen for window resize events to update the state
    window.addEventListener('resize', function() {
        if (isMobile()) {
            window.parent.postMessage({type: 'streamlit:setStateValue', key: 'mobile_view', value: true}, '*');
        } else {
            window.parent.postMessage({type: 'streamlit:setStateValue', key: 'mobile_view', value: false}, '*');
        }
    });
</script>
""", unsafe_allow_html=True)

# Add custom styled tab buttons for navigation
st.markdown("""
<style>
    /* Custom menu tabs CSS removed as requested */

    /* Menu item styles removed as requested */
</style>

<!-- Menu container removed as requested -->

<script>
    // Set active tab based on current state
    document.addEventListener('DOMContentLoaded', function() {
        // Get current tab from URL or default to single_comment
        const urlParams = new URLSearchParams(window.location.search);
        const currentTab = urlParams.get('active_tab') || 'single_comment';

        // Remove active class from all tabs
        document.querySelectorAll('.menu-item').forEach(tab => {
            tab.classList.remove('active');
        });

        // Add active class to current tab
        if (currentTab === 'single_comment') {
            document.getElementById('tab-classify').classList.add('active');
        } else if (currentTab === 'about') {
            document.getElementById('tab-about').classList.add('active');
        } else if (currentTab === 'services') {
            document.getElementById('tab-services').classList.add('active');
        } else if (currentTab === 'results') {
            document.getElementById('tab-results').classList.add('active');
        }
    });
</script>
""", unsafe_allow_html=True)

# Logo display - extremely big in the middle with animation
# Add simple animation styles for the logo
st.markdown("""
<style>
    @keyframes pulse {
        0% { transform: scale(1); filter: drop-shadow(0 0 5px rgba(225, 29, 72, 0.1)); }
        50% { transform: scale(1.05); filter: drop-shadow(0 0 15px rgba(225, 29, 72, 0.3)); }
        100% { transform: scale(1); filter: drop-shadow(0 0 5px rgba(225, 29, 72, 0.1)); }
    }

    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-15px); }
        100% { transform: translateY(0px); }
    }

    .animated-logo {
        animation: pulse 3s infinite ease-in-out, float 6s infinite ease-in-out;
        transform-origin: center center;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .animated-logo:hover {
        filter: brightness(1.1) drop-shadow(0 0 20px rgba(225, 29, 72, 0.5)) !important;
        animation-play-state: paused;
        transform: scale(1.1);
    }
</style>
""", unsafe_allow_html=True)

# Then add the logo with the base64 image and simple animations
st.markdown(f"""
<div style="text-align: center; margin: 0 auto -70px auto; max-width: 100%;">
    <img src="data:image/png;base64,{logo_base64}" alt="Toxic Classifier Logo" class="animated-logo" style="height: 500px; background-color: transparent !important; mix-blend-mode: multiply; filter: url(#remove-white);">
</div>
""", unsafe_allow_html=True)

# We're now using custom HTML/CSS for the menu

# Show different content based on active tab
if st.session_state.active_tab == "about":
    # About page content
    st.markdown(f"""
    <div class="about-container">
        <div class="about-header">
            <h1>ABOUT NAZAR</h1>
        </div>

        <div class="about-hero">
            <img src="data:image/png;base64,{logo_base64}" alt="Nazar Logo" class="about-logo" style="height: 600px; background-color: transparent !important; mix-blend-mode: multiply; filter: url(#remove-white);">
            <div style="height: 20px;"></div>
        </div>

        <div class="about-section">
            <div class="section-label">THE WATCHFUL EYE OF ONLINE SAFETY</div>

            <p class="about-lead">Nazar (نظر) is an AI-powered platform designed to safeguard digital spaces by detecting and classifying toxic comments in Arabic, with a special focus on the Egyptian dialect.</p>

            <div class="simple-divider"></div>

            <p>Inspired by the Arabic word "نظر" meaning "watchful eye," Nazar reflects vigilance, protection, and cultural awareness in maintaining respectful conversations online.</p>

            <p>At its core, Nazar leverages the power of transformer-based models like DistilBERT, fine-tuned on real-world toxic comment datasets. The system evaluates content based on linguistic and contextual cues to determine if a comment is toxic, clean, or in future updates—threatening, insulting, or obscene.</p>
        </div>

        <div class="about-image-full">
            <img src="https://images.unsplash.com/photo-1563906267088-b029e7101114?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2070&q=80" alt="Digital Safety Concept">
        </div>

        <div class="about-section">
            <div class="section-label">WHY NAZAR?</div>

            <p class="about-lead">With the rise of digital communication, content moderation has become a vital necessity. Arabic-speaking communities face unique challenges due to dialect diversity and lack of robust NLP support.</p>

            <div class="simple-divider"></div>

            <ul class="services-list">
                <li class="service-item">Accurate, context-aware analysis of Arabic comments</li>
                <div class="simple-divider"></div>
                <li class="service-item">Support for both single comments and batch file uploads</li>
                <div class="simple-divider"></div>
                <li class="service-item">Confidence-based results and visualization</li>
                <div class="simple-divider"></div>
                <li class="service-item">Ongoing learning from real-world interactions</li>
            </ul>
        </div>

        <div class="quote">
            "To create a safer, more inclusive internet for Arabic users by providing intelligent, culturally aware moderation tools that promote respectful dialogue and trust."
            <div class="quote-attribution">Our Mission</div>
        </div>

        <div class="about-section">
            <div class="section-label">POWERED BY AI, BUILT FOR COMMUNITIES</div>

            <p class="about-lead">Nazar is more than just a toxic comment detector—it's a step toward responsible tech.</p>

            <div class="simple-divider"></div>

            <p>Built with transparency and adaptability in mind, our platform is continuously improving with user feedback, dataset expansion, and advanced model tuning.</p>
        </div>

        <div class="cta">
            <div class="cta-title">Let's create safer spaces together</div>
            <button class="cta-button" style="background-color: #ef4444; color: white;" onclick="document.getElementById('classify_tab').click();">Start Classifying</button>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Add CSS for About page
    st.markdown("""
    <style>
        .about-container {
            max-width: 900px;
            margin: 0 auto;
            padding: 1rem 0;
            text-align: center;
        }

        .about-header {
            text-align: center;
            margin-bottom: 3rem;
        }

        .about-header h1 {
            font-family: 'Cormorant', serif;
            font-weight: 700;
            font-size: 3.5rem;
            letter-spacing: -0.02em;
        }

        .about-section {
            margin-bottom: 4rem;
            text-align: center;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
        }

        .section-label {
            font-family: 'Tenor Sans', sans-serif;
            font-weight: 400;
            font-size: 0.9rem;
            letter-spacing: 0.1em;
            color: var(--accent-color);
            margin-bottom: 1.5rem;
        }

        .about-lead {
            font-family: 'Cormorant', serif;
            font-weight: 600;
            font-size: 1.8rem;
            line-height: 1.3;
            margin-bottom: 1.5rem;
        }

        .about-image-full {
            margin: 4rem 0;
        }

        .about-image-full img {
            width: 100%;
            height: auto;
        }

        @media (max-width: 768px) {
            .about-header h1 {
                font-size: 2.5rem;
            }

            .about-lead {
                font-size: 1.5rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)
else:
    # Hero Section - different for mobile and desktop
    if st.session_state.get('mobile_view', False):
        st.markdown(f"""
        <div class="hero" style="margin-top: -70px;">
            <div class="hero-title">TOXIC CLASSIFIER</div>
            <p class="hero-description">
                AI-POWERED CONTENT MODERATION
            </p>
            <!-- Image will be added via Streamlit -->
        </div>
        """, unsafe_allow_html=True)

        # Display the image using Streamlit's native image functionality
        from PIL import Image
        try:
            image = Image.open('images/image1.png')
            st.image(image, width=400, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")
    else:
        st.markdown(f"""
        <div class="hero" style="margin-top: -70px;">
            <div class="hero-title">TOXIC CLASSIFIER</div>
            <p class="hero-description">
                AI-POWERED CONTENT MODERATION
            </p>
            <!-- Image will be added via Streamlit -->
        </div>
        """, unsafe_allow_html=True)

        # Display the image using Streamlit's native image functionality
        try:
            image = Image.open('images/image1.png')
            st.image(image, width=400, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")

# Only show the main content when not on the About page
if st.session_state.active_tab != "about":
    # Divider with section title and tooltip
    st.markdown("""
    <div class="divider">
        <div class="divider-line"></div>
        <div class="divider-text">
            CLASSIFY COMMENTS
            <span class="tooltip-container">
                <i class="info-icon">ℹ️</i>
                <span class="tooltip-text">
                    <strong>What is a toxic comment?</strong><br>
                    Toxic comments contain rude, disrespectful, or unreasonable language that is likely to make someone leave a discussion. This includes hate speech, threats, insults, and identity-based attacks.
                </span>
            </span>
        </div>
        <div class="divider-line"></div>
    </div>

    <style>
        /* Tooltip styling */
        .tooltip-container {
            position: relative;
            display: inline-block;
            margin-left: 8px;
            cursor: help;
        }

        .info-icon {
            font-size: 16px;
            opacity: 0.8;
            transition: opacity 0.3s;
        }

        .tooltip-container:hover .info-icon {
            opacity: 1;
        }

        .tooltip-text {
            visibility: hidden;
            width: 300px;
            background-color: #333;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 12px;
            position: absolute;
            z-index: 1000;
            bottom: 125%;
            left: 50%;
            margin-left: -150px;
            opacity: 0;
            transition: opacity 0.3s;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            font-size: 14px;
            line-height: 1.5;
            font-weight: normal;
            text-transform: none;
        }

        .tooltip-text::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #333 transparent transparent transparent;
        }

        .tooltip-container:hover .tooltip-text {
            visibility: visible;
            opacity: 1;
        }
    </style>
    """, unsafe_allow_html=True)

# Tabs for different classification methods - only show when not on About page
if st.session_state.active_tab != "about":
    # Custom styled buttons like the top menu
    active_tab_js = "'single_comment'" if st.session_state.active_tab == "single_comment" else "'upload_csv'"

    # First add the styles
    st.markdown("""
    <style>
        /* Custom styled sub-menu - clearer and more distinct */
        .submenu-container {
            display: flex;
            width: 100%;
            margin: 20px 0 30px 0;
            gap: 20px;
        }

        .submenu-item {
            flex: 1;
            padding: 18px 10px;
            text-align: center;
            font-family: 'Tenor Sans', sans-serif;
            font-weight: 700;
            font-size: 1.1rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            cursor: pointer;
            transition: all 0.3s ease;
            color: white;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            border: 2px solid rgba(255,255,255,0.2);
        }

        .submenu-item:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        }

        .submenu-item.active {
            border: 2px solid white;
            box-shadow: 0 0 15px rgba(255,255,255,0.3);
            font-weight: 700;
        }
    </style>
    """, unsafe_allow_html=True)

    # Create two clear, prominent buttons for tab switching
    col1, col2 = st.columns(2)

    with col1:
        if st.button("✏️ ANALYZE COMMENT", key="single_tab_btn", use_container_width=True,
                    type="primary" if st.session_state.active_tab == "single_comment" else "secondary"):
            st.session_state.active_tab = "single_comment"
            st.rerun()

    with col2:
        if st.button("📄 ANALYZE CSV", key="csv_tab_btn", use_container_width=True,
                    type="primary" if st.session_state.active_tab == "upload_csv" else "secondary"):
            st.session_state.active_tab = "upload_csv"
            st.rerun()

    # Add styling for the buttons - simple, modern design with sticky menu
    st.markdown("""
    <style>
        /* Sticky menu styles - positioned much higher */
        .stApp {
            margin-top: 40px !important;
        }

        section[data-testid="stSidebar"] {
            z-index: 1000 !important;
        }

        header[data-testid="stHeader"] {
            display: none !important;
        }

        div[data-testid="block-container"] > div:first-child {
            position: fixed !important;
            top: -25px !important; /* Move up by 25px */
            left: 0 !important;
            right: 0 !important;
            z-index: 999 !important;
            background-color: white !important;
            padding: 2px 1rem !important; /* Further reduced padding */
            box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
        }

        /* Add padding to the top of the content to account for the fixed menu */
        div[data-testid="block-container"] > div:nth-child(2) {
            padding-top: 40px !important; /* Further reduced padding */
        }

        /* Target the main container to move it up */
        .st-emotion-cache-t1wise.eht7o1d4 {
            margin-top: -30px !important;
        }
        /* Style the primary button (active tab) */
        [data-testid="baseButton-primary"] {
            background-color: #e11d48 !important;
            background-image: linear-gradient(135deg, #e11d48, #f43f5e) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            font-family: 'Tenor Sans', sans-serif !important;
            font-weight: 700 !important;
            font-size: 0.9rem !important;
            padding: 0.5rem 0.75rem !important;
            box-shadow: 0 6px 12px rgba(225, 29, 72, 0.3) !important;
            transition: all 0.3s ease !important;
            letter-spacing: 0.05em !important;
            text-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
        }

        /* Style the secondary button (inactive tab) */
        [data-testid="baseButton-secondary"] {
            background-color: #f8f8f8 !important;
            color: #333 !important;
            border: 2px solid #e0e0e0 !important;
            border-radius: 12px !important;
            font-family: 'Tenor Sans', sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            padding: 0.5rem 0.75rem !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
            transition: all 0.3s ease !important;
            letter-spacing: 0.05em !important;
        }

        /* Hover effects */
        [data-testid="baseButton-primary"]:hover {
            box-shadow: 0 8px 15px rgba(225, 29, 72, 0.4) !important;
            transform: translateY(-3px) !important;
        }

        [data-testid="baseButton-secondary"]:hover {
            background-color: white !important;
            border-color: #ccc !important;
            box-shadow: 0 6px 12px rgba(0,0,0,0.15) !important;
            transform: translateY(-3px) !important;
        }

        /* Make sure text is visible and prominent */
        [data-testid="baseButton-primary"] p, [data-testid="baseButton-secondary"] p {
            color: inherit !important;
            margin: 0 !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }

        /* Add animation to the buttons */
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); }
        }

        [data-testid="baseButton-primary"] {
            animation: pulse 2s infinite ease-in-out;
        }

        /* Add spacing between buttons */
        [data-testid="column"] {
            padding: 0 5px !important;
        }

        /* Adjust button margins for the sticky menu */
        [data-testid="baseButton-primary"], [data-testid="baseButton-secondary"] {
            margin-bottom: 0px !important;
            margin-top: -5px !important;
        }
    </style>
    """, unsafe_allow_html=True)



    # Add a small spacer
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# Add styling for the tab buttons
st.markdown("""
<style>
    /* Style the tab buttons to look like tabs */
    [data-testid="baseButton-secondary"], [data-testid="baseButton-primary"] {
        border-radius: 0 !important;
        border: none !important;
        padding: 1rem !important;
        font-family: 'Tenor Sans', sans-serif !important;
        font-weight: 400 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        font-size: 0.9rem !important;
    }

    /* Primary button (active tab) */
    [data-testid="baseButton-primary"] {
        background-color: white !important;
        color: var(--text-color) !important;
        border-bottom: 2px solid var(--text-color) !important;
    }

    /* Secondary button (inactive tab) */
    [data-testid="baseButton-secondary"] {
        background-color: #f8f8f8 !important;
        color: var(--accent-color) !important;
    }

    /* Remove padding between columns */
    [data-testid="column"] {
        padding: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# No need for the hidden button anymore

# Content based on active tab
if st.session_state.active_tab == "about":
    # About page content
    st.markdown(f"""
    <div class="about-container">
        <div class="about-header">
            <h1>ABOUT NAZAR</h1>
        </div>

        <div class="about-hero">
            <img src="data:image/png;base64,{logo_base64}" alt="Nazar Logo" class="about-logo" style="background-color: transparent !important; mix-blend-mode: multiply; filter: url(#remove-white);">
        </div>

        <div class="about-section">
            <div class="section-label">THE WATCHFUL EYE OF ONLINE SAFETY</div>

            <p class="about-lead">Nazar (نظر) is an AI-powered platform designed to safeguard digital spaces by detecting and classifying toxic comments in Arabic, with a special focus on the Egyptian dialect.</p>

            <div class="simple-divider"></div>

            <p>Inspired by the Arabic word "نظر" meaning "watchful eye," Nazar reflects vigilance, protection, and cultural awareness in maintaining respectful conversations online.</p>

            <p>At its core, Nazar leverages the power of transformer-based models like DistilBERT, fine-tuned on real-world toxic comment datasets. The system evaluates content based on linguistic and contextual cues to determine if a comment is toxic, clean, or in future updates—threatening, insulting, or obscene.</p>
        </div>

        <div class="about-image-full">
            <img src="https://images.unsplash.com/photo-1563906267088-b029e7101114?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2070&q=80" alt="Digital Safety Concept">
        </div>

        <div class="about-section">
            <div class="section-label">WHY NAZAR?</div>

            <p class="about-lead">With the rise of digital communication, content moderation has become a vital necessity. Arabic-speaking communities face unique challenges due to dialect diversity and lack of robust NLP support.</p>

            <div class="simple-divider"></div>

            <ul class="services-list">
                <li class="service-item">Accurate, context-aware analysis of Arabic comments</li>
                <div class="simple-divider"></div>
                <li class="service-item">Support for both single comments and batch file uploads</li>
                <div class="simple-divider"></div>
                <li class="service-item">Confidence-based results and visualization</li>
                <div class="simple-divider"></div>
                <li class="service-item">Ongoing learning from real-world interactions</li>
            </ul>
        </div>

        <div class="quote">
            "To create a safer, more inclusive internet for Arabic users by providing intelligent, culturally aware moderation tools that promote respectful dialogue and trust."
            <div class="quote-attribution">Our Mission</div>
        </div>

        <div class="about-section">
            <div class="section-label">POWERED BY AI, BUILT FOR COMMUNITIES</div>

            <p class="about-lead">Nazar is more than just a toxic comment detector—it's a step toward responsible tech.</p>

            <div class="simple-divider"></div>

            <p>Built with transparency and adaptability in mind, our platform is continuously improving with user feedback, dataset expansion, and advanced model tuning.</p>
        </div>

        <div class="cta">
            <div class="cta-title">Let's create safer spaces together</div>
            <button class="cta-button" style="background-color: #ef4444; color: white;" onclick="window.parent.postMessage({{'type': 'streamlit:setStateValue', 'key': 'active_tab', 'value': 'single_comment'}}, '*');">Start Classifying</button>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Add CSS for About page
    st.markdown("""
    <style>
        .about-container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 2rem 0;
        }

        .about-header {
            text-align: center;
            margin-bottom: 3rem;
        }

        .about-header h1 {
            font-family: 'Cormorant', serif;
            font-weight: 700;
            font-size: 3.5rem;
            letter-spacing: -0.02em;
        }

        .about-hero {
            text-align: center;
            margin-bottom: 4rem;
            position: relative;
        }

        .about-logo {
            width: auto;
            height: 600px;
            margin: 0 auto;
            display: block;
        }

        .about-section {
            margin-bottom: 4rem;
        }

        .section-label {
            font-family: 'Tenor Sans', sans-serif;
            font-weight: 400;
            font-size: 0.9rem;
            letter-spacing: 0.1em;
            color: var(--accent-color);
            margin-bottom: 1.5rem;
        }

        .about-lead {
            font-family: 'Cormorant', serif;
            font-weight: 600;
            font-size: 1.8rem;
            line-height: 1.3;
            margin-bottom: 1.5rem;
        }

        .about-image-full {
            margin: 4rem 0;
        }

        .about-image-full img {
            width: 100%;
            height: auto;
        }

        @media (max-width: 768px) {
            .about-header h1 {
                font-size: 2.5rem;
            }

            .about-logo {
                height: 150px;
            }

            .about-lead {
                font-size: 1.5rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)

elif st.session_state.active_tab == "single_comment":
    # Single Comment Classifier
    # Use different column ratios on mobile vs desktop
    if st.session_state.get('mobile_view', False):
        col1, col2 = st.columns([1, 1])
    else:
        col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-title">Classify a Single Comment</div>', unsafe_allow_html=True)
        st.markdown('<p class="section-content">Enter a comment below to check if it contains toxic content. Our AI model will analyze the text and provide a classification with confidence score.</p>', unsafe_allow_html=True)

        with st.form("comment_form", clear_on_submit=True):
            # Add custom styling for an enhanced text area and button
            st.markdown("""
            <style>
                /* Enhanced text area styling with theme support */
                textarea, .stTextArea textarea, div[data-baseweb="textarea"] textarea {
                    border-radius: 12px !important;
                    padding: 15px !important;
                    font-size: 16px !important;
                    transition: all 0.3s ease !important;
                    font-family: 'Inter', sans-serif !important;
                }

                /* Light theme text area */
                .light-theme textarea, .light-theme .stTextArea textarea, .light-theme div[data-baseweb="textarea"] textarea {
                    background-color: white !important;
                    background: white !important;
                    border: 2px solid #e2e8f0 !important;
                    color: #334155 !important;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;
                }

                /* Dark theme text area */
                .dark-theme textarea, .dark-theme .stTextArea textarea, .dark-theme div[data-baseweb="textarea"] textarea {
                    background-color: #1e293b !important;
                    background: #1e293b !important;
                    border: 2px solid #475569 !important;
                    color: #e2e8f0 !important;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.2) !important;
                }

                /* Light theme focus */
                .light-theme textarea:focus, .light-theme .stTextArea textarea:focus, .light-theme div[data-baseweb="textarea"] textarea:focus {
                    border-color: #e11d48 !important;
                    box-shadow: 0 0 0 3px rgba(225, 29, 72, 0.2) !important;
                }

                /* Dark theme focus */
                .dark-theme textarea:focus, .dark-theme .stTextArea textarea:focus, .dark-theme div[data-baseweb="textarea"] textarea:focus {
                    border-color: #f43f5e !important;
                    box-shadow: 0 0 0 3px rgba(244, 63, 94, 0.3) !important;
                }

                /* Add a subtle animation when typing - light theme */
                @keyframes typing-pulse-light {
                    0% { box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
                    50% { box-shadow: 0 2px 15px rgba(225, 29, 72, 0.15); }
                    100% { box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
                }

                /* Add a subtle animation when typing - dark theme */
                @keyframes typing-pulse-dark {
                    0% { box-shadow: 0 2px 10px rgba(0,0,0,0.2); }
                    50% { box-shadow: 0 2px 15px rgba(244, 63, 94, 0.25); }
                    100% { box-shadow: 0 2px 10px rgba(0,0,0,0.2); }
                }

                .light-theme textarea:focus, .light-theme .stTextArea textarea:focus {
                    animation: typing-pulse-light 2s infinite;
                }

                .dark-theme textarea:focus, .dark-theme .stTextArea textarea:focus {
                    animation: typing-pulse-dark 2s infinite;
                }

                /* Label styling with theme support */
                .stTextArea label, div[data-baseweb="textarea"] label {
                    font-family: 'Tenor Sans', sans-serif !important;
                    font-weight: 600 !important;
                    font-size: 18px !important;
                    margin-bottom: 8px !important;
                }

                /* Light theme label */
                .light-theme .stTextArea label, .light-theme div[data-baseweb="textarea"] label {
                    color: #334155 !important;
                }

                /* Dark theme label */
                .dark-theme .stTextArea label, .dark-theme div[data-baseweb="textarea"] label {
                    color: #e2e8f0 !important;
                }

                /* Form submit button styling with theme support */
                .stFormSubmitButton button, [data-testid="stFormSubmitButton"] button, button[kind="secondaryFormSubmit"], [data-testid="stBaseButton-secondaryFormSubmit"] {
                    color: white !important;
                    border-radius: 12px !important;
                    border: none !important;
                    transition: all 0.3s ease !important;
                }

                /* Light theme button */
                .light-theme .stFormSubmitButton button, .light-theme [data-testid="stFormSubmitButton"] button,
                .light-theme button[kind="secondaryFormSubmit"], .light-theme [data-testid="stBaseButton-secondaryFormSubmit"] {
                    background-color: #ef4444 !important;
                    background-image: linear-gradient(135deg, #e11d48, #f43f5e) !important;
                    box-shadow: 0 6px 12px rgba(225, 29, 72, 0.3) !important;
                }

                /* Dark theme button */
                .dark-theme .stFormSubmitButton button, .dark-theme [data-testid="stFormSubmitButton"] button,
                .dark-theme button[kind="secondaryFormSubmit"], .dark-theme [data-testid="stBaseButton-secondaryFormSubmit"] {
                    background-color: #f43f5e !important;
                    background-image: linear-gradient(135deg, #f43f5e, #fb7185) !important;
                    box-shadow: 0 6px 12px rgba(244, 63, 94, 0.4) !important;
                }

                /* Light theme button hover */
                .light-theme .stFormSubmitButton button:hover, .light-theme [data-testid="stFormSubmitButton"] button:hover {
                    background-color: #dc2626 !important;
                    transform: translateY(-2px) !important;
                    box-shadow: 0 8px 15px rgba(225, 29, 72, 0.4) !important;
                }

                /* Dark theme button hover */
                .dark-theme .stFormSubmitButton button:hover, .dark-theme [data-testid="stFormSubmitButton"] button:hover {
                    background-color: #fb7185 !important;
                    transform: translateY(-2px) !important;
                    box-shadow: 0 8px 15px rgba(244, 63, 94, 0.5) !important;
                }

                /* Button text color */
                .stFormSubmitButton button p, [data-testid="stFormSubmitButton"] button p, [data-testid="stBaseButton-secondaryFormSubmit"] p, .st-emotion-cache-b0y9n5 p, .st-emotion-cache-ovf5rk p {
                    color: white !important;
                }

                /* Target by class */
                .st-emotion-cache-b0y9n5 {
                    background-color: #ef4444 !important;
                }
            </style>
            """, unsafe_allow_html=True)

            comment = st.text_area("Enter your comment:", height=150, key="comment_textarea", placeholder="Type or paste your comment here to analyze its content...")

            # Add custom styling for the submit button - targeting the exact button class
            st.markdown("""
            <style>
                /* Target the exact button class */
                button[kind="secondaryFormSubmit"], [data-testid="stBaseButton-secondaryFormSubmit"] {
                    background-color: #e11d48 !important;
                    background-image: linear-gradient(135deg, #e11d48, #f43f5e) !important;
                    color: white !important;
                    border-radius: 6px !important;
                    padding: 6px 10px !important;
                    font-family: 'Tenor Sans', sans-serif !important;
                    font-weight: 500 !important;
                    font-size: 0.75rem !important;
                    letter-spacing: 0.02em !important;
                    text-transform: uppercase !important;
                    border: 1px solid white !important;
                    box-shadow: 0 2px 6px rgba(225, 29, 72, 0.25) !important;
                    transition: all 0.3s ease !important;
                    width: 100% !important;
                    margin-top: 8px !important;
                    position: relative !important;
                    overflow: hidden !important;
                    animation: pulse-glow 2s infinite ease-in-out !important;
                }

                /* Add pulsing glow effect */
                @keyframes pulse-glow {
                    0% { box-shadow: 0 2px 4px rgba(225, 29, 72, 0.2); }
                    50% { box-shadow: 0 2px 6px rgba(225, 29, 72, 0.3); }
                    100% { box-shadow: 0 2px 4px rgba(225, 29, 72, 0.2); }
                }

                /* Hover effect */
                button[kind="secondaryFormSubmit"]:hover, [data-testid="stBaseButton-secondaryFormSubmit"]:hover {
                    transform: translateY(-1px) !important;
                    box-shadow: 0 4px 8px rgba(225, 29, 72, 0.3) !important;
                    border: 1px solid white !important;
                }

                /* Style the text inside the button */
                [data-testid="stBaseButton-secondaryFormSubmit"] p, button[kind="secondaryFormSubmit"] p {
                    color: white !important;
                    text-shadow: 0 1px 1px rgba(0,0,0,0.15) !important;
                    margin: 0 !important;
                    font-size: 0.75rem !important;
                    font-weight: 500 !important;
                }

                /* Add icons using ::before and ::after */
                .stFormSubmitButton {
                    position: relative;
                    width: 100%;
                }

                /* Emoji icons removed as requested */
            </style>
            """, unsafe_allow_html=True)

            # Create the submit button with a more descriptive label
            classify = st.form_submit_button("ANALYZE NOW")

            if classify:
                if comment.strip() == "":
                    st.warning("Please enter a comment before classifying.")
                else:
                    inputs = tokenizer(comment, return_tensors="pt", truncation=True, padding=True)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        probs = F.softmax(outputs.logits, dim=1)
                        pred = torch.argmax(probs, dim=1).item()
                        confidence = probs[0][1].item()

                    label = "Toxic" if pred == 1 else "Clean"
                    result_class = "result-toxic" if pred == 1 else "result-clean"

                    # Add icons based on the result
                    icon = "✓" if label == "Clean" else "✗"

                    # Add a loading animation before showing results
                    with st.spinner("Analyzing comment..."):
                        # Simulate a brief delay for better UX
                        import time
                        time.sleep(0.5)

                    # Determine emoji based on result
                    emoji = "✅" if label == "Clean" else "⚠️"

                    # Calculate confidence percentage for the bar
                    confidence_pct = int(confidence * 100)

                    # Determine confidence level text
                    if confidence > 0.8:
                        confidence_level = "High"
                    elif confidence > 0.6:
                        confidence_level = "Medium"
                    else:
                        confidence_level = "Low"

                    # Create a color gradient based on confidence
                    if label == "Toxic":
                        bar_color = f"linear-gradient(90deg, #ef4444 {confidence_pct}%, #fecaca {confidence_pct}%)"
                    else:
                        bar_color = f"linear-gradient(90deg, #10b981 {confidence_pct}%, #d1fae5 {confidence_pct}%)"

                    # Break the HTML into parts to avoid f-string issues
                    copy_text = f"Classification: {label} (Confidence: {confidence:.2f})"
                    confidence_display = f"{confidence:.2f}"

                    # Generate a unique ID for this result
                    import uuid
                    result_id = str(uuid.uuid4())[:8]

                    # Create a single HTML string for the result card
                    result_html = f'''
                    <div class="result-card {result_class} animate-result">
                        <div class="result-icon">{emoji}</div>
                        <div class="result-content">
                            <div class="result-header">
                                <div class="result-label">{label} Comment</div>
                                <div class="result-actions">
                                    <button class="action-button copy-btn" onclick="navigator.clipboard.writeText('{copy_text}').then(() => showToast('Copied to clipboard!'))">
                                        <span>📋</span>
                                    </button>
                                    <button class="action-button info-btn" onclick="toggleInfo('confidence-info-{result_id}')">
                                        <span>ℹ️</span>
                                    </button>
                                </div>
                            </div>
                            <div id="confidence-info-{result_id}" class="info-box" style="display: none;">
                                <p>Confidence score indicates how certain the model is about this classification. Higher values mean greater certainty.</p>
                            </div>
                            <div class="result-confidence">Confidence: <span class="confidence-value">{confidence_display}</span> <span class="confidence-level">({confidence_level})</span></div>
                            <div class="confidence-bar-container">
                                <div class="confidence-bar" style="width: {confidence_pct}%; background: {bar_color};"></div>
                            </div>
                        </div>
                    </div>
                    '''

                    # JavaScript and CSS
                    js_css = '''
                    <script>
                        // Function to show toast notification
                        function showToast(message) {
                            // Create toast element if it doesn't exist
                            let toast = document.getElementById('toast-notification');
                            if (!toast) {
                                toast = document.createElement('div');
                                toast.id = 'toast-notification';
                                toast.className = 'toast';
                                document.body.appendChild(toast);
                            }

                            // Set message and show toast
                            toast.textContent = message;
                            toast.style.opacity = '1';

                            // Hide toast after 3 seconds
                            setTimeout(() => {
                                toast.style.opacity = '0';
                            }, 3000);
                        }

                        // Function to toggle info box
                        function toggleInfo(id) {
                            const infoBox = document.getElementById(id);
                            if (infoBox.style.display === 'none') {
                                infoBox.style.display = 'block';
                            } else {
                                infoBox.style.display = 'none';
                            }
                        }
                    </script>

                    <style>
                        /* Result card animation */
                        @keyframes fadeIn {
                            from { opacity: 0; transform: translateY(10px); }
                            to { opacity: 1; transform: translateY(0); }
                        }

                        .animate-result {
                            animation: fadeIn 0.5s ease-out forwards;
                        }

                        /* Result header with actions */
                        .result-header {
                            display: flex;
                            justify-content: space-between;
                            align-items: center;
                            margin-bottom: 0.5rem;
                        }

                        .result-actions {
                            display: flex;
                            gap: 8px;
                        }

                        .action-button {
                            background: none;
                            border: none;
                            cursor: pointer;
                            font-size: 16px;
                            padding: 4px;
                            border-radius: 4px;
                            transition: all 0.2s;
                        }

                        .action-button:hover {
                            background-color: rgba(0,0,0,0.05);
                        }

                        /* Confidence bar styling */
                        .confidence-bar-container {
                            width: 100%;
                            height: 8px;
                            background-color: #f1f5f9;
                            border-radius: 4px;
                            margin-top: 8px;
                            overflow: hidden;
                        }

                        .confidence-bar {
                            height: 100%;
                            border-radius: 4px;
                            transition: width 1s ease-out;
                        }

                        .confidence-value {
                            font-weight: 600;
                        }

                        .confidence-level {
                            font-size: 0.85em;
                            opacity: 0.8;
                        }

                        /* Info box styling with theme support */
                        .info-box {
                            padding: 10px;
                            margin: 8px 0;
                            font-size: 0.9em;
                            border-radius: 0 4px 4px 0;
                        }

                        .light-theme .info-box {
                            background-color: #f8fafc;
                            border-left: 3px solid #64748b;
                            color: #334155;
                        }

                        .dark-theme .info-box {
                            background-color: #1e293b;
                            border-left: 3px solid #94a3b8;
                            color: #e2e8f0;
                        }

                        /* Toast notification */
                        .toast {
                            position: fixed;
                            bottom: 20px;
                            right: 20px;
                            background: #333;
                            color: white;
                            padding: 12px 20px;
                            border-radius: 8px;
                            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                            z-index: 9999;
                            opacity: 0;
                            transition: opacity 0.3s ease;
                        }
                    </style>
                    '''

                    # Combine HTML and JavaScript/CSS and display
                    st.markdown(result_html + js_css, unsafe_allow_html=True)

                    # Save to history
                    st.session_state.history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "comment": comment,
                        "label": label,
                        "confidence": round(confidence, 2)
                    })

    with col2:
        # Image for the right column - different styling for mobile
        if st.session_state.get('mobile_view', False):
            st.markdown("""
            <div style="padding: 1rem 0; text-align: center;">
                <img src="https://images.unsplash.com/photo-1516321318423-f06f85e504b3?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1470&q=80"
                     alt="AI Analysis" style="max-width: 100%; border: 1px solid var(--border-color);">
                <p style="margin-top: 0.5rem; font-style: italic; color: var(--accent-color); font-size: 0.8rem;">AI analysis of text patterns</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="padding: 2rem; text-align: center;">
                <img src="https://images.unsplash.com/photo-1516321318423-f06f85e504b3?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1470&q=80"
                     alt="AI Analysis" style="max-width: 100%; border: 1px solid var(--border-color);">
                <p style="margin-top: 1rem; font-style: italic; color: var(--accent-color);">Our AI model analyzes text patterns to detect toxic content</p>
            </div>
            """, unsafe_allow_html=True)
elif st.session_state.active_tab == "upload_csv":
    # Upload CSV section
    st.markdown('<div class="section-title">Upload a CSV File to Classify Comments</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-content">Upload a CSV file containing comments to classify them in bulk. The file must have a column named "comment_text".</p>', unsafe_allow_html=True)

    # Add custom styling for the file uploader
    st.markdown("""
    <style>
        /* Additional file uploader styling */
        [data-testid="stFileUploader"] {
            background-color: rgba(239, 68, 68, 0.8) !important;
        }

        [data-testid="stFileUploader"] * {
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)

    file = st.file_uploader("Upload a CSV file with a column named 'comment_text'", type=["csv"])
    if file:
        df = pd.read_csv(file)
        if "comment_text" not in df.columns:
            st.error("CSV must contain a column named 'comment_text'.")
        else:
            comments = df["comment_text"].astype(str).tolist()
            results = []
            for text in comments:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = F.softmax(outputs.logits, dim=1)
                    pred = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][1].item()
                    label = "Toxic" if pred == 1 else "Clean"
                    results.append({"comment": text, "label": label, "confidence": round(confidence, 2)})

            uploaded_df = pd.DataFrame(results)
            st.session_state.uploaded_results[file.name] = uploaded_df

            # Display results in a more styled way
            st.markdown('<div class="section-title">Classification Results</div>', unsafe_allow_html=True)

            # Initialize session state for filter term if it doesn't exist
            if 'filter_term' not in st.session_state:
                st.session_state.filter_term = ""

            # Add search functionality with a simpler approach
            search_col1, search_col2 = st.columns([3, 1])

            # Create a form for the search to handle submission properly
            with st.form(key="search_form", clear_on_submit=True):
                search_col1, search_col2 = st.columns([3, 1])

                with search_col1:
                    search_term = st.text_input("Search comments:", placeholder="Type to filter results...")

                with search_col2:
                    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing to align with text input
                    search_submitted = st.form_submit_button("🔍 SEARCH", type="primary", use_container_width=True)

                if search_submitted and search_term:
                    st.session_state.filter_term = search_term

            # Filter the dataframe based on the stored filter term
            filter_term = st.session_state.filter_term
            if filter_term:
                filtered_df = uploaded_df[uploaded_df['comment'].str.contains(filter_term, case=False, na=False)]
            else:
                filtered_df = uploaded_df

            # Add custom styling for the dataframe
            st.markdown("""
            <style>
                /* Style the dataframe with white and red colors */
                [data-testid="stDataFrame"] {
                    border: 2px solid #ef4444 !important;
                    border-radius: 10px !important;
                    overflow: hidden !important;
                }

                /* Header styling */
                .stDataFrame th {
                    background-color: #ef4444 !important;
                    color: white !important;
                    font-weight: 600 !important;
                    text-transform: uppercase !important;
                    letter-spacing: 0.05em !important;
                    padding: 12px 15px !important;
                }

                /* Row styling */
                .stDataFrame tr:nth-child(even) {
                    background-color: #fff5f5 !important;
                }

                .stDataFrame tr:nth-child(odd) {
                    background-color: white !important;
                }

                /* Cell styling */
                .stDataFrame td {
                    padding: 10px 15px !important;
                    border-bottom: 1px solid #fee2e2 !important;
                }

                /* Toxic label styling */
                .stDataFrame td:contains("Toxic") {
                    color: #ef4444 !important;
                    font-weight: 600 !important;
                }

                /* Clean label styling */
                .stDataFrame td:contains("Clean") {
                    color: #10b981 !important;
                    font-weight: 600 !important;
                }

                /* Search button styling */
                [data-testid="baseButton-primary"], [data-testid="stFormSubmitButton"] button {
                    background-color: #ef4444 !important;
                    background-image: linear-gradient(135deg, #e11d48, #f43f5e) !important;
                    color: white !important;
                    border-radius: 12px !important;
                    border: none !important;
                    font-weight: 600 !important;
                    box-shadow: 0 4px 8px rgba(225, 29, 72, 0.3) !important;
                }

                /* Make sure search button text is white */
                [data-testid="baseButton-primary"] p, [data-testid="stFormSubmitButton"] button p {
                    color: white !important;
                    font-weight: 700 !important;
                }

                /* Search results info */
                .search-results-info {
                    margin-top: 10px;
                    margin-bottom: 10px;
                    font-style: italic;
                    color: #6b7280;
                }
            </style>
            """, unsafe_allow_html=True)

            # Display search results info
            if filter_term:
                st.markdown(f"<div class='search-results-info'>Found {len(filtered_df)} results for '{filter_term}'</div>", unsafe_allow_html=True)

            # Reset index to start from 1 instead of 0
            display_df = filtered_df.copy()
            display_df.index = range(1, len(display_df) + 1)

            # Display the dataframe with row numbers starting from 1 and ensure all rows are visible
            st.dataframe(display_df, use_container_width=True, height=min(500, 100 + len(display_df) * 35))

            # Add a large export button
            # Create a copy with 1-based indexing for export
            export_df = uploaded_df.copy()
            export_df.index = range(1, len(export_df) + 1)
            csv_data = export_df.to_csv(index=True).encode("utf-8")

            # Custom CSS for the download button
            st.markdown("""
            <style>
                /* Large export button styling */
                .big-download-button .stDownloadButton button {
                    background-color: #ef4444 !important;
                    background-image: linear-gradient(135deg, #e11d48, #f43f5e) !important;
                    color: white !important;
                    border-radius: 12px !important;
                    border: none !important;
                    padding: 15px 25px !important;
                    font-size: 18px !important;
                    font-weight: 600 !important;
                    box-shadow: 0 6px 12px rgba(225, 29, 72, 0.3) !important;
                    transition: all 0.3s ease !important;
                    display: flex !important;
                    align-items: center !important;
                    justify-content: center !important;
                    width: 100% !important;
                    margin-top: 20px !important;
                }

                /* Make sure the text is red */
                .big-download-button .stDownloadButton button p {
                    color: white !important;
                    font-weight: 700 !important;
                }

                .big-download-button .stDownloadButton button:hover {
                    transform: translateY(-2px) !important;
                    box-shadow: 0 8px 15px rgba(225, 29, 72, 0.4) !important;
                }

                /* Add download icon */
                .big-download-button .stDownloadButton button::before {
                    content: '⬇️';
                    margin-right: 10px;
                    font-size: 20px;
                }
            </style>
            """, unsafe_allow_html=True)

            # Use Streamlit's built-in download button with custom styling
            st.markdown('''
            <style>
                /* Style the download button */
                [data-testid="stDownloadButton"] {
                    width: 100% !important;
                    margin-top: 20px !important;
                }

                [data-testid="stDownloadButton"] button {
                    background-color: #ef4444 !important;
                    background-image: linear-gradient(135deg, #e11d48, #f43f5e) !important;
                    color: white !important;
                    border-radius: 12px !important;
                    border: none !important;
                    padding: 15px 25px !important;
                    font-size: 18px !important;
                    font-weight: 600 !important;
                    box-shadow: 0 6px 12px rgba(225, 29, 72, 0.3) !important;
                    transition: all 0.3s ease !important;
                    width: 100% !important;
                }

                [data-testid="stDownloadButton"] button:hover {
                    transform: translateY(-2px) !important;
                    box-shadow: 0 8px 15px rgba(225, 29, 72, 0.4) !important;
                }

                /* Style the button text */
                [data-testid="stDownloadButton"] button p {
                    color: #ffffff !important;
                    font-weight: 700 !important;
                    font-size: 18px !important;
                }

                /* Add download icon */
                [data-testid="stDownloadButton"] button::before {
                    content: '⬇️' !important;
                    margin-right: 10px !important;
                    font-size: 20px !important;
                }
            </style>
            ''', unsafe_allow_html=True)

            # Use Streamlit's built-in download button
            st.download_button(
                label="EXPORT RESULTS CSV",
                data=csv_data,
                file_name=f"toxic_classification_{file.name}",
                mime="text/csv",
                use_container_width=True
            )

# Divider for the next section
st.markdown("""
<div class="divider">
    <div class="divider-line"></div>
    <div class="divider-text">FEATURES</div>
    <div class="divider-line"></div>
</div>
""", unsafe_allow_html=True)

# Features in a gallery layout
st.markdown("""
<div class="gallery">
    <div class="gallery-item">
        <img src="https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1470&q=80"
             class="gallery-image" alt="Single Comment Classification">
        <div class="gallery-caption">
            <strong>Single Comment Classification</strong><br>
            Instantly determine whether a comment is toxic or safe with a clear confidence score.
        </div>
    </div>

    <div class="gallery-item">
        <img src="https://images.unsplash.com/photo-1460925895917-afdab827c52f?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1415&q=80"
             class="gallery-image" alt="Batch Processing">
        <div class="gallery-caption">
            <strong>Batch Processing via CSV Upload</strong><br>
            Upload files containing thousands of comments and receive a clean, classified report in seconds.
        </div>
    </div>

    <div class="gallery-item">
        <img src="https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1470&q=80"
             class="gallery-image" alt="Session History">
        <div class="gallery-caption">
            <strong>Session History & Visualization</strong><br>
            View your session log with confidence metrics, timestamps, and a visual chart of results.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Services section with horizontal dividers
st.markdown("""
<div class="divider">
    <div class="divider-line"></div>
    <div class="divider-text">SERVICES</div>
    <div class="divider-line"></div>
</div>

<p style="text-align: center; max-width: 600px; margin: 0 auto 2rem auto;" class="section-content">
    Our AI-powered system offers multiple ways to protect your online spaces from toxic content. Choose the service that best fits your needs.
</p>

<div class="simple-divider"></div>

<ul class="services-list">
    <li class="service-item">Single comment analysis</li>
    <div class="simple-divider"></div>
    <li class="service-item">Batch processing via CSV upload</li>
    <div class="simple-divider"></div>
    <li class="service-item">Real-time content moderation</li>
    <div class="simple-divider"></div>
    <li class="service-item">API integration for your platform</li>
</ul>

<div style="text-align: center; margin: 3rem 0;">
    <a href="#" class="cta-button" style="background-color: #ef4444; color: white;">Get Started</a>
</div>
""", unsafe_allow_html=True)

# Quote section
st.markdown("""
<div class="quote">
    "Our AI-powered system helps protect online spaces by automatically identifying and filtering toxic content with high accuracy."
    <div class="quote-attribution">Toxic Comment Classifier • AI-Powered Content Moderation</div>
</div>
""", unsafe_allow_html=True)

# This section is now handled above in the tab content

# CTA Section
st.markdown("""
<div class="cta">
    <div class="cta-title">Let's protect online spaces together</div>
    <button class="cta-button" style="background-color: #ef4444; color: white;" onclick="document.getElementById('tab-single').click()">Start Classifying</button>
</div>
""", unsafe_allow_html=True)

# Session History (hidden by default)
if st.session_state.active_tab == "history" and st.session_state.history:
    st.markdown('<div class="divider"><div class="divider-line"></div><div class="divider-text">SESSION HISTORY</div><div class="divider-line"></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Your Classification History</div>', unsafe_allow_html=True)

    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df, use_container_width=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        hist_df["label"].value_counts().plot.pie(autopct="%1.1f%%", startangle=90, colors=["#ef4444", "#10b981"], ax=ax1)
        ax1.set_ylabel("")
        ax1.set_title("Session Toxicity Distribution")
        st.pyplot(fig1)

    with col2:
        if st.session_state.history:
            hist_df = pd.DataFrame(st.session_state.history)
            csv = hist_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Session CSV", csv, "session_history.csv", "text/csv")

        if st.button("Reset Session"):
            st.session_state.history = []
            st.session_state.uploaded_results = {}
            st.rerun()

# Uploaded Results (hidden by default)
if st.session_state.active_tab == "results" and st.session_state.uploaded_results:
    st.markdown('<div class="divider"><div class="divider-line"></div><div class="divider-text">UPLOADED RESULTS</div><div class="divider-line"></div></div>', unsafe_allow_html=True)

    for fname, df in st.session_state.uploaded_results.items():
        st.markdown(f'<div class="section-title">{fname}</div>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

        fig2, ax2 = plt.subplots(figsize=(6, 6))
        df["label"].value_counts().plot.pie(autopct="%1.1f%%", startangle=90, colors=["#ef4444", "#10b981"], ax=ax2)
        ax2.set_ylabel("")
        ax2.set_title(f"Toxicity Distribution for {fname}")
        st.pyplot(fig2)

# Email signup
st.markdown("""
<div class="simple-divider"></div>

<div class="email-signup">
    <div class="email-signup-title">Stay in the loop</div>
    <p style="margin-bottom: 1.5rem; font-size: 0.9rem;">Sign up to receive updates, new features, and more. We respect your privacy and will never share your information.</p>
    <div style="display: flex; gap: 1rem;">
        <input type="email" placeholder="Email Address" style="flex-grow: 1; padding: 0.8rem; border: 1px solid var(--border-color); font-family: 'Tenor Sans', sans-serif; font-size: 0.9rem;">
        <button style="background-color: var(--text-color); color: white; border: none; padding: 0 1.5rem; font-family: 'Tenor Sans', sans-serif; font-size: 0.9rem; letter-spacing: 0.05em;">Get in touch</button>
    </div>
</div>

<div class="simple-divider"></div>
""", unsafe_allow_html=True)

# Enhanced Footer
st.markdown("""
<div class="footer card">
    <div style="margin-bottom: 10px;">© 2023 Toxic Comment Classifier | Built with ❤️ using Streamlit and DistilBERT</div>
    <div style="display: flex; justify-content: center; gap: 20px; margin-top: 10px;">
        <a href="#" onclick="window.showToast('GitHub link clicked!'); return false;" style="color: var(--accent-color); text-decoration: none;">GitHub</a>
        <a href="#" onclick="window.showToast('Documentation link clicked!'); return false;" style="color: var(--accent-color); text-decoration: none;">Documentation</a>
        <a href="#" onclick="window.showToast('Contact link clicked!'); return false;" style="color: var(--accent-color); text-decoration: none;">Contact</a>
    </div>
</div>
""", unsafe_allow_html=True)
