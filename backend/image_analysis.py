import pytesseract
from PIL import Image
import io

def analyze_image(file):
    try:
        # Read the image file
        image = Image.open(file)
        
        # Extract text using OCR
        text = pytesseract.image_to_string(image)
        
        # Basic analysis of the extracted text
        lines = text.split('\n')
        chart_type = "Unknown"
        key_points = []
        
        # Simple analysis to detect chart type and key points
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect chart type
            if any(keyword in line.lower() for keyword in ['candlestick', 'candle']):
                chart_type = "Candlestick Chart"
            elif any(keyword in line.lower() for keyword in ['line', 'trend']):
                chart_type = "Line Chart"
            elif any(keyword in line.lower() for keyword in ['bar', 'histogram']):
                chart_type = "Bar Chart"
                
            # Collect key points (lines with numbers or percentages)
            if any(char.isdigit() for char in line):
                key_points.append(line)
        
        return {
            'chart_type': chart_type,
            'key_points': key_points[:5],  # Limit to top 5 points
            'raw_text': text,
            'analysis': f"Detected a {chart_type.lower()}. Found {len(key_points)} key data points in the image."
        }
        
    except Exception as e:
        return {
            'error': f'Error analyzing image: {str(e)}'
        }