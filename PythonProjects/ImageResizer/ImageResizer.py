#Program to resize the image
from PIL import Image

# Step 1: Open the image (JPEG format)
image_path = "OIP.jpeg"  # Replace this with your JPEG image path
img = Image.open(image_path)

# Step 2: Display the original size
print(f"Original size: {img.size}")  # Output example: (1920, 1080)

# Step 3: Define new size (resizing to 400x400 pixels as an example)
new_width = 400
new_height = 400
new_size = (new_width, new_height)

# Step 4: Resize the image
resized_img = img.resize(new_size)

# Step 5: Save the resized image as PNG format
png_image_path = "resized_image.png"  # Name of the output PNG file
resized_img.save(png_image_path, "PNG")  # Save as PNG format
print(f"Resized image saved as '{png_image_path}' (in PNG format)")