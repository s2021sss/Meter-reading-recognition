import argparse
from Meter_reading_recognition import MeterImageRecognizer  

def main(image_path):
    recognizer = MeterImageRecognizer(
        yolo_model_path='models/best.pt',
        cnn_model_path_upside_down_img = 'models/CustomCNNForUpsideDownImages.pth',
        cnn_model_path_digit_recogn='models/CustomCNNForDigits.pth',
        cropped_dir='data/CroppedImagesForRecognition'
    )
    
    single_image_name = image_path.split("/")[-1]
    recognized_number = recognizer.process_single_image(image_path)
    print(f"Recognized Number {single_image_name}: {recognized_number}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recognize meter readings from an image.")
    parser.add_argument(
        "--image-path", 
        required=True, 
        help="Path to the image to be processed."
    )
    args = parser.parse_args()
    main(args.image_path)
