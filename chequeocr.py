import cv2
from ultralytics import YOLO
import easyocr
import matplotlib.pyplot as plt

# Path to your trained YOLOv8 model
MODEL_PATH = 'best.pt'

# Class names as per your YAML
CLASS_NAMES = ['Amount_In_Numbers', 'Amount_In_Words', 'Date', 'MICR', 'Payee_Name', 'Sign']

def run_cheque_ocr(image_path):
    # Load YOLOv8 model
    model = YOLO(MODEL_PATH)
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found!")
        return
    
    # Run detection
    results = model(image)
    detections = results[0].boxes.data.cpu().numpy()
    cheque_detected = False  # Flag to check if any cheque field is detected

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if conf < 0.5:
            continue
        cls = int(cls)
        label = CLASS_NAMES[cls]
        if label not in ['Amount_In_Numbers', 'Amount_In_Words', 'Date', 'MICR', 'Payee_Name']:
            continue
        cheque_detected = True  # Set flag if any relevant field is detected
        # Draw bounding box and label on the image
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Print results
    if cheque_detected:
        print("The uploaded document is a cheque.")
    else:
        print("No cheque fields detected. The uploaded document may not be a cheque.")
    
    # Show and save the output image
    output_path = 'output_detected_cheque.jpg'
    cv2.imwrite(output_path, image)

    # Display using matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.title('Detected Cheque')
    plt.axis('off')
    plt.show()

    print(f"Output image saved as {output_path}") 

if __name__ == "__main__":
    # Example usage: replace with your cheque image path
    # cheque_image_path = 'Cheques.v1i.yolov8/valid/images/X_021_jpeg_jpg.rf.4c6d6c58c78213ac24e2f1cb921e4c2c.jpg'
    # cheque_image_path = 'Cheques.v1i.yolov8/valid/images/X_063_jpeg_jpg.rf.47223ccae72f02eb1b2ed45e958e8ad9.jpg'
    cheque_image_path = 'Cheques.v1i.yolov8/train/images/X_090_jpeg_jpg.rf.18fa920172c8347419b656da996e4d10.jpg'
    # cheque_image_path = 'C:\\Users\\pranjal.prabhu\\Downloads\\IDRBT_Cheque_Image_Dataset\\IDRBT Cheque Image Dataset\\300\\Cheque 100830.tif'
    # cheque_image_path = 'C:\\Users\\pranjal.prabhu\\Desktop\\cheque\\cimage.webp'
    run_cheque_ocr(cheque_image_path)