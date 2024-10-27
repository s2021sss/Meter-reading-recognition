import pandas as pd
from sklearn.model_selection import train_test_split
import os
import glob
import shutil
import ast
from tqdm import tqdm
import cv2
from ultralytics import YOLO
import numpy as np
from scipy.spatial import ConvexHull

class YOLOTrainer:
    def __init__(self, data_path, image_dir, tags_dir, model_name='YOLOv8m.pt', obb_model=False):
        # Инициализация класса, загрузка данных и разделение на обучающую и валидационную выборки
        self.data = pd.read_csv(data_path, sep='\t')
        # self.train_data, self.val_data = train_test_split(self.data, test_size=0.2, random_state=42)

        self.read_tags()
        filtered_data = self.data[self.data['photo_name'].isin(self.tags.keys())]
        self.data = self.data[~self.data['photo_name'].isin(self.tags.keys())][:50]

        self.train_data_without_filtered, self.val_data_without_filtered = train_test_split(self.data, test_size=0.2, random_state=42)
        self.train_filtered, self.val_filtered = train_test_split(filtered_data, test_size=0.2, random_state=42)

        self.train_data = pd.concat([self.train_data_without_filtered, self.train_filtered])
        self.val_data = pd.concat([self.val_data_without_filtered, self.val_filtered])

        self.image_dir = image_dir
        self.tags_dir = tags_dir
        self.model_name = model_name
        self.setup_directories()
        self.obb_model = True if "-obb.pt" in model_name or obb_model else False

    def setup_directories(self):
        # Создание директорий для хранения данных для обучения и валидации
        os.makedirs('datasets/images/train', exist_ok=True)
        os.makedirs('datasets/labels/train', exist_ok=True)
        os.makedirs('datasets/images/val', exist_ok=True)
        os.makedirs('datasets/labels/val', exist_ok=True)
        os.makedirs('datasets/images_obb/train', exist_ok=True)
        os.makedirs('datasets/images_obb/val', exist_ok=True)

    def convert_polygon_to_bbox(self, polygon_data, img_width, img_height):
        # Преобразование полигональных данных в координаты ограничивающей рамки (bbox)
        xs = [point['x'] * img_width for point in polygon_data]
        ys = [point['y'] * img_height for point in polygon_data]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        x_center = x_min + bbox_width / 2
        y_center = y_min + bbox_height / 2

        return x_center / img_width, y_center / img_height, bbox_width / img_width, bbox_height / img_height

    def convert_polygon_to_obb_points(self, polygon_data, img_width, img_height):
        # Преобразование полигональных данных в координаты ориентированной ограничивающей рамки (OBB)
        points = np.array([[point['x'] * img_width, point['y'] * img_height] for point in polygon_data])
        
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        
        hull_points = np.array(hull_points, dtype=np.float32)
        
        rect = cv2.minAreaRect(hull_points)
        box = cv2.boxPoints(rect)  

        normalized_points = [(x / img_width, y / img_height) for x, y in box]
        return normalized_points

    def get_image_size(self, image_path):
        # Получение размеров изображения
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        return width, height
    
    def draw_obb_on_image(self, image_path, obb_points, img_width, img_height, save_path):
        # Сохранение исходных заново размеченных изображений 
        image = cv2.imread(image_path)
        
        obb_points = np.array([[x * img_width, y * img_height] for x, y in obb_points], dtype=np.int32)
        
        cv2.polylines(image, [obb_points], isClosed=True, color=(0, 255, 0), thickness=10)
        
        cv2.imwrite(save_path, image)

    def move_images_and_create_labels(self, data, img_dest, label_dest, obb_img_dest):
        # Скопирование изображений и создание меток для YOLO
        for _, row in tqdm(data.iterrows(), total=data.shape[0]):
            image_name = row['photo_name']
            image_path = os.path.join(self.image_dir, image_name)
            
            shutil.copy(image_path, os.path.join(img_dest, image_name))
            
            label_name = os.path.splitext(image_name)[0] + '.txt'
            label_path = os.path.join(label_dest, label_name)
            
            img_width, img_height = self.get_image_size(image_path)
            polygon_data = ast.literal_eval(row['location'])["data"]
            if self.obb_model:
                coords = self.convert_polygon_to_obb_points(polygon_data, img_width, img_height)
            else:
                coords = self.convert_polygon_to_bbox(polygon_data, img_width, img_height)
            
            with open(label_path, 'w') as f:
                tag = self.tags[image_name] if image_name in self.tags.keys() else "0"
                f.write(tag+" "+" ".join([f"{var[0]} {var[1]}" for var in coords]) + "\n")

            save_obb_img_path = os.path.join(obb_img_dest, image_name)
            self.draw_obb_on_image(image_path, coords, img_width, img_height, save_obb_img_path)

    def read_tags(self):
        # Добавление отдельных классов для нестандартных счетчиков
        file_names = [f for f in glob.glob("TlkWaterMeters/tags/*.txt")]

        self.tags = {}

        for file_name in file_names:
            with open(file_name, 'r', encoding='utf-8') as file:
                img_names = file.read().splitlines()
            for img_name in img_names:
                self.tags[img_name] = file_name.split("/")[-1][:-4]

    def prepare_data(self):
        # Подготовка данных: создание меток и копирование изображений
        self.move_images_and_create_labels(self.train_data, 'datasets/images/train', 'datasets/labels/train', 'datasets/images_obb/train')
        self.move_images_and_create_labels(self.val_data, 'datasets/images/val', 'datasets/labels/val', 'datasets/images_obb/val')

    def train_model(self, epochs=5, imgsz=640, batch_size=16):
        # Обучение модели YOLO
        model = YOLO(self.model_name)
        model.train(data='dataset.yaml', epochs=epochs, imgsz=imgsz, batch=batch_size)


# trainer = YOLOTrainer(data_path='TlkWaterMeters/data.tsv', image_dir='TlkWaterMeters/images', tags_dir='TlkWaterMeters/tags', model_name='YOLOv8m.pt')
trainer = YOLOTrainer(data_path='TlkWaterMeters/data.tsv', image_dir='TlkWaterMeters/images', tags_dir='TlkWaterMeters/tags', model_name='yolo11n-obb.pt', obb_model=True)

# trainer.prepare_data() 
# trainer.train_model(epochs=3, imgsz=640, batch_size=16) 
