import torch
import cv2
import pytesseract
from pathlib import Path
from matplotlib import pyplot as plt
import tempfile 
from img2table.ocr import TesseractOCR
from img2table.document import Image
import cv2


weights_path = 'C:/shyam/invoice_reader/yolov5/runs/train/my_model/weights/best.pt'
image_path = "C:/shyam/invoice_reader/invoice.jpg"
output_path = "C:/shyam/invoice_reader/out"
text_output_path = 'C:/shyam/invoice_reader/out/extracted.txt'

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)

# Load image
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Inference
results = model(img_rgb)

# Results
results.print()  # Print results to console

# Extract bounding boxes and labels
boxes = results.xyxy[0].cpu().numpy()  # xyxy format

# Class names (adjust these based on your dataset)
class_names = ['logo', 'table', 'text']

# Ensure the output directory exists
Path(output_path).mkdir(parents=True, exist_ok=True)

# Convert image back to BGR for saving
output_image = results.render()[0]
output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

# Save the image with predictions
output_image_path = Path(output_path) / Path(image_path).name
cv2.imwrite(str(output_image_path), output_image)

# Prepare to save extracted text
extracted_texts = []

# Iterate through each detected box
for box in boxes:
    xmin, ymin, xmax, ymax, conf, cls = map(int, box[:6])
    class_name = class_names[cls]

    # Crop the image based on the bounding box coordinates
    cropped_img = img[ymin:ymax, xmin:xmax]

    # Use pytesseract to extract text from the cropped image
    text = pytesseract.image_to_string(cropped_img, config='--psm 6')

    # For the table class, perform additional processing
    if class_name == 'table':
        temp_file_path = tempfile.NamedTemporaryFile(suffix=".png").name
        cv2.imwrite(temp_file_path, cropped_img)

        # Instantiation of OCR
        ocr = TesseractOCR(n_threads=1, lang="eng")

        # Instantiation of document from the saved image file
        doc = Image(temp_file_path)

        # Table extraction
        extracted_tables = doc.extract_tables(ocr=ocr,
                                              implicit_rows=False,
                                              borderless_tables=False,
                                              min_confidence=50)
        # Here, you might want to retrieve the extracted table data and append it to the text variable
        doc.to_xlsx(dest="C:/shyam/invoice_reader/extract.xlsx",
                    ocr=ocr,
                    implicit_rows=False,
                    borderless_tables=False,
                    min_confidence=50)

    # Save extracted text with class name and bounding box coordinates
    # You might want to append the extracted table data to the text variable here

# After processing all boxes, you can remove the temporary file


# Write all extracted texts to a text file
with open(text_output_path, 'w') as f:
    f.writelines(extracted_texts)

# Plot the results
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

print(f"Extracted text saved to: {text_output_path}")
