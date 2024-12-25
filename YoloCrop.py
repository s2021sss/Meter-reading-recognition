import cv2
import os
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import glob
import shutil
import torch
from PIL import Image
from torchvision import transforms
from Train_CNN_for_upside_down_images import CustomCNNForUpsideDownImages

class YOLOCropper:
    """
    Класс для обрезки изображений с помощью модели YOLO.
    """
    def __init__(self, yolo_model_path, cnn_model_path_upside_down_img, img_size=640, iou_thresh=0.7, conf_thresh=0.75, obb_model=False):
        """
        Инициализация класса YOLOCropper.

        Args:
            yolo_model_path (str): Путь к модели YOLO.
            cnn_model_path_upside_down_img (str): Путь к модели CNN для распознавания перевернутых изображений.
            img_size (int): Размер изображений для обработки.
            iou_thresh (float): Порог пересечения (IoU) для фильтрации рамок.
            conf_thresh (float): Порог уверенности для предсказания объектов.
            obb_model (bool): Использовать ли модель с обобщенными ограничивающими рамками (OBB).
        """
        self.model = YOLO(yolo_model_path)
        self.img_size = img_size
        self.iou_thresh = iou_thresh # порог для фильтрации перекрывающихся ограничивающих рамок 
        # (если на одном изображении предсказывается несколько рамок, то при пересечении двумя рамками порога iou_thresh, 
        # одна из них будет отброшена)

        self.conf_thresh = conf_thresh # минимальное значение уверенности для того, чтобы модель считала, что объект обнаружен на изображении

        self.obb_model = obb_model


        self.modelUpsideDownImages = CustomCNNForUpsideDownImages()
        self.modelUpsideDownImages.load_state_dict(torch.load(cnn_model_path_upside_down_img))
        self.modelUpsideDownImages.eval()
        
        self.transform = transforms.Compose([
        transforms.Resize((80, 300)),
        transforms.ToTensor()
        ])

    def is_image_flipped(self, warped):
        """
        Проверяет, перевернуто ли изображение, и корректирует его.

        Args:
            warped (numpy.ndarray): Изображение для проверки.

        Returns:
            numpy.ndarray: Скорректированное изображение.
        """
        hsv_image = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV) # пространство HSV
    
        red_mask = cv2.inRange(hsv_image, (0, 50, 50), (10, 255, 255)) | cv2.inRange(hsv_image, (350, 50, 50), (360, 255, 255))

        height, width = red_mask.shape
        
        left_half = red_mask[:, :width // 2]
        right_half = red_mask[:, width // 2:]

        red_count_left = cv2.countNonZero(left_half)
        red_count_right = cv2.countNonZero(right_half)

        if red_count_left > red_count_right:
            warped = cv2.rotate(warped, cv2.ROTATE_180)

        return warped

    def flip_upside_down_images(self, image_path):
        """
        Переворачивает изображние на основе CustomCNNForUpsideDownImages.

        Args:
            input_dir (str): Путь к изображениям для обработки.
        """
        
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.modelUpsideDownImages(image)
            predicted_class = int(np.round(torch.sigmoid(outputs)).item())
        if predicted_class == 1:
            img = cv2.imread(image_path)
            warped = cv2.rotate(img, cv2.ROTATE_180)
            cv2.imwrite(image_path, warped)

    def crop_image(self, image_path, output_dir='data/CroppedImages'):
        """
        Обрезает изображение на основе предсказаний модели YOLO.

        Args:
            image_path (str): Путь к изображению для обработки.
            output_dir (str): Путь для сохранения обрезанных изображений.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
                
        image_name = image_path.split(".")[0]

        img = cv2.imread(image_path)
        if img is None:
            print(f"Не удалось загрузить изображение {image_path}")
            return
        
        res = self.model(img, imgsz=self.img_size, iou=self.iou_thresh, conf=self.conf_thresh, verbose=False)
        number_of_WaterMeters = 0
        object_found = False  

        for result in res:
            img = result.orig_img  
            
            if self.obb_model:
                preds = result.obb  
                if preds is not None:

                    for polygon in preds:
                        points = np.array(polygon.xyxyxyxy[0].tolist(), dtype=np.float32) 
                        digit_class = int(polygon[0].cls[0])
                        # print (points)

                        points[[0, 2]] = points[[2, 0]]
                        width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
                        height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
                        
                        dst_points = np.array([[0, 0], 
                                            [width - 1, 0], 
                                            [width - 1, height - 1], 
                                            [0, height - 1]], dtype=np.float32)
                        
                        matrix = cv2.getPerspectiveTransform(points, dst_points)
                        warped = cv2.warpPerspective(img, matrix, (width, height))

                        warped = self.is_image_flipped(warped)

                        image_path_new = os.path.join(output_dir, os.path.basename(f'{image_name}_cropped_{digit_class}_{number_of_WaterMeters}.jpg'))

                        cv2.imwrite(image_path_new, warped)
                        
                        self.flip_upside_down_images(image_path_new)
 
                        number_of_WaterMeters += 1
                        object_found = True


            else:
                preds = result.boxes   
 
                if preds!=None:

                    for box in preds:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  
                        cropped = img[y1:y2, x1:x2]  

                        cv2.imwrite(os.path.join(output_dir, os.path.basename(f'{image_path}_cropped_{number_of_WaterMeters}.jpg')), cropped)
                        number_of_WaterMeters += 1
                        object_found = True

        if not object_found:
            cv2.imwrite(os.path.join(output_dir, os.path.basename(image_path)), img)

        # print(f"Обработка изображения {image_path} завершена.")

    def crop_images_from_folder(self, input_dir, output_dir='data/CroppedImages', max_images=None, imagesForRecognition_dir=None):
        """
        Обрезает изображения из указанной папки.

        Args:
            input_dir (str): Папка с изображениями для обработки.
            output_dir (str): Папка для сохранения обрезанных изображений.
            max_images (int, optional): Максимальное количество изображений для обрезки.
            imagesForRecognition_dir (str, optional): Папка для сохранения оставшихся изображений.
        """
        image_files_all = [f for f in sorted(os.listdir(input_dir)) if f.endswith(('.jpg', '.png'))]
    
        if max_images!=None:
             image_files_for_crop = image_files_all[:max_images]
             
             if imagesForRecognition_dir!=None:
                os.makedirs(imagesForRecognition_dir, exist_ok=True)

                for file in image_files_all[max_images:]:
                    src_path = os.path.join(input_dir, file)
                    dst_path = os.path.join(imagesForRecognition_dir, file)
                    shutil.copy(src_path, dst_path)
        else:
            image_files_for_crop = image_files_all


        if not image_files_for_crop:
            print(f"В папке {input_dir} нет изображений для обработки.")
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for image_file in tqdm(image_files_for_crop, desc="Обрезка изображений", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
            image_path = os.path.join(input_dir, image_file)
            self.crop_image(image_path, output_dir)
        
    def cut_images_into_numbers(self, input_dir, output_dir='digits'):
        """
        Разделяет изображения счетчиков на отдельные цифры и сохраняет их в нужных папках для обучения нейронной сети  

        Args:
            input_dir (str): Папка с изображениями счетчиков.
            output_dir (str): Папка для сохранения отдельных цифр.
        """
        for filename in glob.glob(input_dir+"/*"):
            try:
                image_name = os.path.basename(filename).split('.')[0]

                parts = filename.split('value_')
                if len(parts) > 1 and 'cropped' in parts[1]:
                    value_part1 = parts[1].split('_')[0]
                    value_part2 = parts[1].split('_')[1]
                    digit_class = int(parts[1].split('_')[-2])
                    
                    if digit_class == 1:
                        padded_value = value_part1.zfill(5)

                        image = cv2.imread(filename)

                        if image is None:
                            raise ValueError(f"Не удалось загрузить изображение: {filename}")
                        
                        height, width = image.shape[:2]
                        column_width = width // 5
                        columns = [] 

                        for i in range(5):
                            x_start = i * column_width
                            x_end = x_start + column_width
                            column = image[:, x_start:x_end]
                            columns.append(column)

                        
                        for i, j in enumerate(padded_value):
                            digit_folder = f'{output_dir}/{int(j)}'
                            os.makedirs(digit_folder, exist_ok=True)  
                            
                            output_filename = f'{digit_folder}/{image_name}_digit_{i}.jpg'
                            
                            cv2.imwrite(output_filename, columns[i])     

                    else:
                        if value_part2 != "0":
                            padded_value = value_part1.zfill(8-max(len(value_part2), 3)) + value_part2.ljust(max(len(value_part2), 3),'0')
                        else:
                            padded_value = value_part1.zfill(8)

                        image = cv2.imread(filename)

                        if image is None:
                            raise ValueError(f"Не удалось загрузить изображение: {filename}")
                        
                        height, width = image.shape[:2]
                        column_width = width // 8
                        columns = []

                        for i in range(8):
                            x_start = i * column_width
                            x_end = x_start + column_width
                            column = image[:, x_start:x_end]
                            columns.append(column)
                        
                        for i, j in enumerate(padded_value):
                            digit_folder = f'{output_dir}/{int(j)}'
                            os.makedirs(digit_folder, exist_ok=True)  
                            
                            output_filename = f'{digit_folder}/{image_name}_digit_{i}.jpg'
                            
                            cv2.imwrite(output_filename, columns[i])
                else:
                    print (f"Изображение {filename} не обработано")
            
            except Exception as e:
                print(f"Ошибка: {e}")
                return None



    def cut_image(self, input_dir):
        """
        Разделяет одно изображение на цифры для распознавания цифр на изображении

        Args:
            input_dir (str): Путь к папке, содержащей изображение для обработки.
        """
        image_path = glob.glob(input_dir+"/"+input_dir.split("/")[-1][:1]+"*")[0]
        
        try:
            image_name = os.path.basename(image_path).split('.')[0]
            output_dir = input_dir + "/digits"
            os.makedirs(output_dir, exist_ok=True)  

            parts = image_name.split('cropped_')
            if len(parts) > 1 and 'cropped' in image_name:
                digit_class = int(parts[1].split('_')[-2])

                
                if digit_class == 1:

                    image = cv2.imread(image_path)
                    
                    if image is None:
                        raise ValueError(f"Не удалось загрузить изображение: {image_name}")
                    
                    height, width = image.shape[:2]
                    column_width = width // 5

                    for i in range(5):
                        x_start = i * column_width
                        x_end = x_start + column_width
                        column = image[:, x_start:x_end]

                        output_filename = f'{output_dir}/{image_name}_digitN_{i}.jpg'
                        cv2.imwrite(output_filename, column)     


                else:


                    image = cv2.imread(image_path)

                    if image is None:
                        raise ValueError(f"Не удалось загрузить изображение: {image_name}")
                    
                    height, width = image.shape[:2]
                    column_width = width // 8


                    for i in range(8):
                        x_start = i * column_width
                        x_end = x_start + column_width
                        column = image[:, x_start:x_end]

                        output_filename = f'{output_dir}/{image_name}_digitN_{i}.jpg'
                        cv2.imwrite(output_filename, column)
                    

            else:
                print (f"Изображение {image_name} не обработано")
        
        except Exception as e:
            print(f"Ошибка: {e}")
            return None


if __name__ == "__main__":
    cropper = YOLOCropper(yolo_model_path='models/weights/best_v1.pt', 
                          cnn_model_path_upside_down_img='models/CustomCNNForUpsideDownImages.pth',
                          obb_model=True) #runs/detect/train/weights/best.pt

    # cropper.crop_image(image_path='data/TlkWaterMeters/images/id_9_value_18_724.jpg', output_dir='data/CroppedImages')
    # cropper.crop_image(image_path='data/TlkWaterMeters/images/id_553_value_65_475.jpg', output_dir='data/CroppedImages')
    # cropper.crop_image(image_path='data/TlkWaterMeters/images/id_823_value_797_0.jpg', output_dir='data/CroppedImages')

    cropper.crop_images_from_folder(input_dir='data/TlkWaterMeters/images', 
                                    output_dir='data/CroppedImages', 
                                    max_images=1100, 
                                    imagesForRecognition_dir='data/ImagesForRecognition'
                                    )
    cropper.cut_images_into_numbers(input_dir='data/CroppedImages', output_dir='data/digits')