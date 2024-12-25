from YoloCrop import YOLOCropper
from Train_CNN_for_digit_recognition import CustomCNNForDigits

import os
import torch
from torchvision import transforms
from PIL import Image

class MeterImageRecognizer:
    """Класс для распознавания значений счетчиков с помощью YOLO и CNN."""

    def __init__(self, yolo_model_path, cnn_model_path_upside_down_img, cnn_model_path_digit_recogn, cropped_dir="data/CroppedImagesForRecognition"):
        """
        Инициализация класса распознавателя изображений счетчиков.

        Args:
            yolo_model_path (str): Путь к модели YOLO.
            cnn_model_path_upside_down_img (str): Путь к модели CNN для распознавания перевернутых изображений.
            cnn_model_path_digit_recogn (str): Путь к модели CNN для распознавания цифр.
            cropped_dir (str): Директория для сохранения обрезанных изображений.
        """
        self.cropped_dir = cropped_dir

        self.cropper = YOLOCropper(yolo_model_path=yolo_model_path, cnn_model_path_upside_down_img=cnn_model_path_upside_down_img, obb_model=True)

        self.model = CustomCNNForDigits()
        self.model.load_state_dict(torch.load(cnn_model_path_digit_recogn))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((50, 50)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def predict_image(self, image_path):
        """
        Предсказывает цифру на одном изображении.

        Args:
            image_path (str): Путь к изображению.

        Returns:
            int: Распознанная цифра.
        """
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(image)
            predicted_num = outputs.argmax(dim=1).item()

        return predicted_num

    def process_single_image(self, image_path):
        """
        Обрабатывает одно изображение, обрезает его и распознает число.

        Args:
            image_path (str): Путь к изображению.

        Returns:
            str: Распознанное число.
        """
        cropped_image_dir = os.path.join(self.cropped_dir, os.path.splitext(os.path.basename(image_path))[0])
        os.makedirs(cropped_image_dir, exist_ok=True)

        self.cropper.crop_image(image_path=image_path, output_dir=cropped_image_dir)
        self.cropper.cut_image(input_dir=cropped_image_dir)

        digits_dir = os.path.join(cropped_image_dir, "digits")
        recognized_number = ""
        for digit_name in sorted(os.listdir(digits_dir)):
            digit_path = os.path.join(digits_dir, digit_name)
            predicted_digit = self.predict_image(digit_path)
            recognized_number += str(predicted_digit)

        if len(recognized_number)>5:
            recognized_number = recognized_number[:-3]+"."+recognized_number[-3:]

        return recognized_number

    def process_directory(self, image_dir):
        """
        Обрабатывает все изображения в указанной директории.

        Args:
            image_dir (str): Путь к папке с изображениями.

        Returns:
            dict: Словарь с именами изображений и распознанными числами.
        """
        results = {}

        for image_name in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_name)

            if not image_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
                continue

            recognized_number = self.process_single_image(image_path)
            results[image_name] = recognized_number

        return results
    
    @staticmethod
    def count_mismatched_characters(part1, part2):
        max_length = max(len(part1), len(part2))
        part1 = part1.zfill(max_length)
        part2 = part2.zfill(max_length)
        return sum(c1 != c2 for c1, c2 in zip(part1, part2))
    
    @staticmethod
    def compare_strings(image_name, recognized_number):
        if '.' in recognized_number:
            recognized_number_value1, recognized_number_value2 = recognized_number.split('.', 1)
        else:
            recognized_number_value1, recognized_number_value2 = recognized_number, ''

        image_parts = image_name.split('_')
        
        image_parts = image_name.split('value_')
        image_name_value1 = image_parts[1].split('_')[0]
        image_name_value2 = (image_parts[1].split('_')[1]).split('.')[0]


        if image_name_value2 != "0":
            padded_value = image_name_value1.zfill(8-max(len(image_name_value2), 3)) + image_name_value2.ljust(max(len(image_name_value2), 3),'0')
        else:
            padded_value = image_name_value1.zfill(5)

        
        if len(padded_value)==5:
            return MeterImageRecognizer.count_mismatched_characters(image_name_value1, recognized_number_value1)
        else:
            return MeterImageRecognizer.count_mismatched_characters(image_name_value1, recognized_number_value1) + \
                MeterImageRecognizer.count_mismatched_characters(image_name_value2, recognized_number_value2)


if __name__ == "__main__":
    recognizer = MeterImageRecognizer(
        yolo_model_path='models/weights/best.pt',
        cnn_model_path_upside_down_img = 'models/CustomCNNForUpsideDownImages.pth',
        cnn_model_path_digit_recogn='models/CustomCNNForDigits.pth',
        cropped_dir='data/CroppedImagesForRecognition'
    )

    #Пример распознавания показания с одного изображения
    single_image_path = 'data/ImagesForRecognition/id_924_value_625_593.jpg' 
    # single_image_path = 'data/CroppedImagesForRecognition/id_963_value_242_778.jpg' # не распознанное изображение yolo
    single_image_name = single_image_path.split("/")[-1]
    recognized_number = recognizer.process_single_image(single_image_path)
    print(f"Recognized Number {single_image_name}: {recognized_number}")

    #Пример распознавания показаний со всех изображений в папке
    results = recognizer.process_directory('data/ImagesForRecognition')
    compare_results = []

    for image_name, recognized_number in results.items():
        if recognized_number!="":
            compare_results.append(MeterImageRecognizer.compare_strings(image_name, recognized_number))
        print(f"Image: {image_name}, Recognized Number: {recognized_number}" + \
            (f", number of mismatched digits: {compare_results[-1] if recognized_number!='' else ''}" if recognized_number!="" else ""))