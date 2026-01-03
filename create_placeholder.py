from PIL import Image, ImageDraw, ImageFont
import os

def create_placeholder():
    # Create a white image
    width = 800
    height = 400
    img = Image.new('RGB', (width, height), color='white')
    
    # Draw text
    draw = ImageDraw.Draw(img)
    text = "Model Architecture Diagram\n(Placeholder)\nPlease replace with actual diagram."
    
    # Calculate text position (roughly centered)
    # We don't have easy font metrics without loading a font, so we'll guess
    x = width // 2 - 100
    y = height // 2 - 20
    
    # Draw text in black
    draw.text((x, y), text, fill='black')
    
    # Save
    save_path = os.path.join(r"C:\Users\samet\.gemini\antigravity\brain\a084d933-668e-4f51-937e-dee8dcfb96e6", "model_architecture_placeholder.png")
    img.save(save_path)
    print(f"Created placeholder at {save_path}")

if __name__ == "__main__":
    create_placeholder()
