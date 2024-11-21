import cv2
import os
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from scipy.stats import kurtosis
import glob

class YOLOCropper:
    def __init__(self, model_path, img_size=640, iou_thresh=0.4, conf_thresh=0.75, obb_model=False):
        # Инициализация модели YOLO и ее параметров
        self.model = YOLO(model_path)
        self.img_size = img_size
        self.iou_thresh = iou_thresh # порог для фильтрации перекрывающихся ограничивающих рамок 
        # (если на одном изображении предсказывается несколько рамок, то при пересечении двумя рамками порога iou_thresh, 
        # одна из них будет отброшена)

        self.conf_thresh = conf_thresh # минимальное значение уверенности для того, чтобы модель считала, что объект обнаружен на изображении

        self.obb_model = obb_model

    def is_image_flipped(self, warped):
        # Обработка перевернутого изображения
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

    def crop_image(self, image_path, output_dir='CroppedImages'):
        # Функция обрезки изображения на основе предсказаний модели YOLO
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
                
        image_name = image_path.split(".")[0]

        img = cv2.imread(image_path)
        if img is None:
            print(f"Не удалось загрузить изображение {image_path}")
            return
        
        res = self.model(img, imgsz=self.img_size, iou=self.iou_thresh, conf=self.conf_thresh)
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

                        cv2.imwrite(os.path.join(output_dir, os.path.basename(f'{image_name}_cropped_{digit_class}_{number_of_WaterMeters}.jpg')), warped)

 
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

        print(f"Обработка изображения {image_path} завершена.")

    def crop_images_from_folder(self, input_dir, output_dir='CroppedImages', max_images=None):
        # Функция для обрезки всех изображений в папке на основе предсказаний модели YOLO
        image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))]
    
        if max_images!=None:
             image_files = image_files[:max_images]

        if not image_files:
            print(f"В папке {input_dir} нет изображений для обработки.")
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for image_file in tqdm(image_files, desc="Обрезка изображений"):
            image_path = os.path.join(input_dir, image_file)
            self.crop_image(image_path, output_dir)

    def cut_images_into_numbers(self, input_dir, output_dir='digits'):
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
                        # print (padded_value)

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
                        # print (padded_value)

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



cropper = YOLOCropper(model_path='Yolo_project/weights/best_v1.pt', obb_model=True) #runs/detect/train/weights/best.pt
# cropper.crop_image(image_path='TlkWaterMeters/images/id_9_value_18_724.jpg', output_dir='CroppedImages')
# cropper.crop_image(image_path='TlkWaterMeters/images/id_553_value_65_475.jpg', output_dir='CroppedImages')
# cropper.crop_image(image_path='TlkWaterMeters/images/id_823_value_797_0.jpg', output_dir='CroppedImages')
cropper.crop_images_from_folder(input_dir='Yolo_project/TlkWaterMeters/images', output_dir='Yolo_project/CroppedImages', max_images=500)
cropper.cut_images_into_numbers(input_dir='Yolo_project/CroppedImages', output_dir='Yolo_project/digits')
