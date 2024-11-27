from os import path
import pandas as pd
from PIL import Image
import os
import glob
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score, confusion_matrix



class CustomCNNForUpsideDownImages(nn.Module):
    """Класс для определения кастомной сверточной нейронной сети (CNN) для определения перевернуто ли изображение."""

    def __init__(self):
        """Инициализация архитектуры CNN."""
        super().__init__()
        
        self.conv_layer1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv_layer2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv_layer3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 10 * 37, 256) 
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1) 
        self.activ = nn.ReLU()

    def forward(self, x):
        """
        Определяет прямое прохождение через слои сети.

        Args:
            x (torch.Tensor): Входные данные.

        Returns:
            torch.Tensor: Выходные данные после прохождения через сеть.
        """
        x = self.pool1(self.activ(self.bn1(self.conv_layer1(x))))
        x = self.pool2(self.activ(self.bn2(self.conv_layer2(x))))
        x = self.pool3(self.activ(self.bn3(self.conv_layer3(x))))
        x = self.flatten(x)
        x = self.dropout(self.activ(self.fc1(x)))
        x = self.fc2(x)
        return x


class ImageDataset(Dataset):
    """
    Класс для создания кастомного датасета для загрузки изображений и их меток.
    """
    def __init__(self, image_folder: str, targets: pd.Series, transform):
        self.image_folder = image_folder
        self.transform = transform
        self.images = self._load_images()
        self.targets = targets.values

    def _load_images(self):
        """
        Загрузка всех путей к изображениям из указанной папки.
        """
        folders = os.listdir(self.image_folder)
        images_nested = [glob.glob(os.path.join(self.image_folder, folder, '*')) for folder in folders]
        return [img for sublist in images_nested for img in sublist]

    def __len__(self):
        """
        Возвращает общее количество изображений.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Возвращает изображение и соответствующую метку по индексу.
        """
        image_path = self.images[idx]
        img = self.transform(Image.open(image_path))
        target = self.targets[idx]
        return img, target




class ImageClassificationPipeline:
    """
    Класс для обучения, оценки и сохранения модели по распознавания перевернутых изображений.
    """
    def __init__(self, data_path, image_folder, epochs=5, batch_size=32, learning_rate=0.01):
        """
        Инициализация пайплайна.

        Args:
            data_path (str): Путь к файлу с метками перевернутых изображений.
            image_folder (str): Корневая папка с изображениями.
            epochs (int): Количество эпох.
            batch_size (int): Размер батча.
            learning_rate (float): Скорость обучения.
        """
        self.data_path = data_path
        self.image_folder = image_folder
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = CustomCNNForUpsideDownImages()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()

        self.transform = transforms.Compose([
            transforms.Resize((80, 300)),
            transforms.ToTensor()
        ])


    def prepare_datasets(self):
        """Подготовка обучающего и тестового датасетов."""
        data = pd.read_csv(self.data_path)
        dataset = ImageDataset(self.image_folder, data['target'], self.transform)
        train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def train(self):
        """Обучение модели на тренировочном датасете."""
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            progress_bar = tqdm(self.train_loader, desc=f"Эпоха {epoch + 1}/{self.epochs}")
            for imgs, targets in progress_bar:
                imgs, targets = imgs, targets.float().unsqueeze(1)
                self.optimizer.zero_grad()
                outputs = self.model(imgs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() / len(self.train_loader)
                progress_bar.set_postfix(loss=running_loss)
            print(f"Эпоха {epoch + 1}/{self.epochs}, Потери: {running_loss:.4f}")

    def evaluate(self):
        """Оценка модели на тестовом датасете.

        Returns:
            tuple: Метрики оценки (точность, F1-метрика, средняя абсолютная ошибка).
        """
        self.model.eval()
        preds, actuals = [], []
        total_loss = 0.0
        progress_bar = tqdm(self.test_loader, desc="Оценка")
        with torch.no_grad():
            for imgs, targets in progress_bar:
                imgs, targets = imgs, targets.float().unsqueeze(1)
                outputs = self.model(imgs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                preds.extend((torch.sigmoid(outputs) > 0.5).cpu().numpy())
                actuals.extend(targets.cpu().numpy())

        accuracy = accuracy_score(actuals, preds)
        f1 = f1_score(actuals, preds, average='weighted')
        mae = mean_absolute_error(actuals, preds)
        cm = confusion_matrix(actuals, preds)

        print(f"Потери на валидации: {total_loss / len(self.test_loader):.4f}, Точность: {accuracy:.4f}, F1-скор: {f1:.4f}, MAE: {mae:.4f}")
        print("Матрица ошибок:")
        print(cm)

        return accuracy, f1, mae

    def save_model(self, path='models/CustomCNNForUpsideDownImages.pth'):
        """Сохранение обученной модели.

        Args:
            path (str): Путь для сохранения модели.
        """
        torch.save(self.model.state_dict(), path)
        print(f"Model saved as {path}")


if __name__ == "__main__":
    trainer = ImageClassificationPipeline(
        data_path='data/ImagesForTrainUpsideDownImages/WaterMeters(index=False).csv',
        image_folder='data/ImagesForTrainUpsideDownImages',
        epochs=20,
        batch_size=32,
        learning_rate=0.01
    )
    trainer.prepare_datasets()
    trainer.train()
    trainer.evaluate()
    # trainer.save_model()