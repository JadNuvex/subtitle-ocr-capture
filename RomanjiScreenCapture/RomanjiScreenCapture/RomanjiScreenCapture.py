import cv2
import numpy as np
import pytesseract
import pyautogui
import time
import re
from collections import Counter
from nltk.corpus import words

# Load English dictionary
import nltk
nltk.download('words')
ENGLISH_WORDS = set(words.words())

# Set the path to Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Get screen resolution
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# **Precise subtitle region**
REGION_WIDTH = int(SCREEN_WIDTH * 0.6)  # 50% of screen width
REGION_HEIGHT = int(SCREEN_HEIGHT * 0.50)  # 9% of screen height
REGION_X = (SCREEN_WIDTH - REGION_WIDTH) // 2  # Center horizontally
REGION_Y = int(SCREEN_HEIGHT * 0.80)  # Subtitle position near bottom

SUBTITLE_REGION = (REGION_X, REGION_Y, REGION_X + REGION_WIDTH, REGION_Y + REGION_HEIGHT)

def preprocess_image(image):
    """ Improve OCR accuracy by enhancing subtitle contrast & removing noise """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce background noise
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)  # Sharpen text edges
    _, binary = cv2.threshold(sharpened, 180, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Binarization
    return binary

def clean_text(text):
    """ Remove OCR artifacts, junk characters, and ensure full sentences """
    text = text.strip()

    # Remove non-alphabetic characters (except spaces)
    text = re.sub(r"[^A-Za-z ]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Split words & filter out unwanted ones
    words_only = text.split()

    # **Filter 1: Remove lines with fewer than 3 words**
    if len(words_only) < 3:
        return ""

    # **Filter 2: Remove lines where more than 50% of words aren't in the dictionary**
    valid_words = [word for word in words_only if word.lower() in ENGLISH_WORDS]
    if len(valid_words) / len(words_only) < 0.75:  # Require at least 75% real words
        return ""

    return " ".join(valid_words)

def capture_subtitle():
    """ Capture the designated subtitle area & extract text """
    screenshot = np.array(pyautogui.screenshot())

    # Crop to **only** the subtitle area
    x1, y1, x2, y2 = SUBTITLE_REGION
    subtitle_area = screenshot[y1:y2, x1:x2]

    # Preprocess for better OCR accuracy
    processed_image = preprocess_image(subtitle_area)

    # Extract text using Tesseract (force proper spacing)
    raw_text = pytesseract.image_to_string(processed_image, lang="eng", config="--psm 6").strip()

    return clean_text(raw_text)

def merge_text_variants(text_samples):
    """ Merge multiple subtitle detections to form the best possible line """
    words = []

    for text in text_samples:
        words.extend(text.split())

    # Count occurrences of words
    word_counts = Counter(words)

    # Keep only words that appear multiple times (filters out OCR mistakes)
    merged_text = " ".join(word for word, count in word_counts.items() if count > 1)

    return merged_text if merged_text else max(text_samples, key=len, default="")

def capture_and_combine():
    """ Capture multiple frames & merge results for higher accuracy """
    text_samples = []

    for _ in range(3):  # Capture 3 frames
        subtitle = capture_subtitle()
        if subtitle:  # Only store non-empty results
            text_samples.append(subtitle)
        time.sleep(0.02)  # Small delay between captures

    if text_samples:
        merged_text = merge_text_variants(text_samples)  # Merge best results
        print("Detected Subtitle:", merged_text)
    else:
        print("No subtitle detected.")

# Run subtitle extraction every 2 seconds
while True:
    capture_and_combine()
    time.sleep(2)
