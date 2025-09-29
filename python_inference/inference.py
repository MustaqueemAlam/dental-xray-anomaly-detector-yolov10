import sys
import os
import cv2
import pickle
from ultralytics import YOLO

# **NEW**: Added a main function with try-except block for robust error handling
def main(image_path):
    try:
        # Define paths relative to this script
        script_dir = os.path.dirname(__file__)
        model_path = os.path.join(script_dir, '..', 'model_files', 'dental_yolov10_best.pt')
        mapping_path = os.path.join(script_dir, '..', 'model_files', 'class_mapping.pkl')
        output_dir = os.path.join(script_dir, '..', 'public', 'processed')
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model and class mapping
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model file not found at {model_path}")
        model = YOLO(model_path)
        
        if not os.path.exists(mapping_path):
             raise FileNotFoundError(f"Class mapping file not found at {mapping_path}")
        with open(mapping_path, 'rb') as f:
            class_to_id = pickle.load(f)
        id_to_class = {v: k for k, v in class_to_id.items()}

        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            raise IOError(f"Could not read the image file at {image_path}")

        # Perform prediction
        results = model.predict(source=img, verbose=False)
        
        # Draw bounding boxes
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                class_name = id_to_class.get(cls, 'Unknown')
                label = f'{class_name} {conf:.2f}'
                
                # Draw rectangle and label
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Save the processed image
        output_filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, img)
        
        # Print the output path to stdout for the Node.js server
        print(f"Processed image saved to: {output_path}")

    except Exception as e:
        # **CRITICAL**: Print any error to stderr so Node.js can catch it
        print(f"Error in inference.py: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        input_image_path = sys.argv[1]
        main(input_image_path)
    else:
        print("Error: No image path provided.", file=sys.stderr)
        sys.exit(1)

