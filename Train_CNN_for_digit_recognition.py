import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm



class CustomCNNForDigits(nn.Module):
        """Класс для определения кастомной сверточной нейронной сети (CNN) для классификации цифр."""

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
            self.fc1 = nn.Linear(128 * 6 * 6, 256)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(256, 10)
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
        

class DigitsDataset(Dataset):
        """Класс для работы с датасетом цифр со счетчиков."""
        
        def __init__(self, images, transform=None, augment_transform=None, augment_save_dir=None):
            """
            Инициализация датасета.

            Args:
                images (list): Список изображений и меток классов.
                transform (callable, optional): Трансформации для изображений.
                augment_transform (callable, optional): Аугментации для изображений.
                augment_save_dir (str, optional): Папка для сохранения аугментированных изображений.
            """
            self.images = images
            self.transform = transform
            self.augment_transform = augment_transform
            self.augment_save_dir = augment_save_dir

            if self.augment_save_dir:
                os.makedirs(self.augment_save_dir, exist_ok=True)

        def __len__(self):
            """
            Возвращает количество элементов в датасете.

            Returns:
                int: Количество элементов.
            """
            return len(self.images) * (2 if self.augment_transform else 1)

        def __getitem__(self, idx):
            """
            Возвращает элемент датасета по индексу.

            Args:
                idx (int): Индекс элемента.

            Returns:
                tuple: Кортеж из изображения и метки класса.
            """
            original_idx = idx // 2 if self.augment_transform else idx
            image_path, label = self.images[original_idx]

            if self.augment_transform and idx % 2 == 1:
                image = Image.open(image_path).convert('RGB')
                image = self.augment_transform(image)

                if self.augment_save_dir:
                    augmented_image_path = os.path.join(
                        self.augment_save_dir, 
                        f"aug_{os.path.basename(image_path)}"
                    )
                    save_image = transforms.ToPILImage()(image).convert('RGB')
                    save_image.save(augmented_image_path)

            else:

                image = Image.open(image_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)

            return image, label
        


class DigitsClassificationPipeline:
    """Класс для классификации цифр со счетчиков с поддержкой аугментации и обучения модели."""

    def __init__(self, root_dir, test_size=0.2, batch_size=32, lr=0.001, epochs=3, augment_save_dir=None):
        """
        Инициализация пайплайна.

        Args:
            root_dir (str): Корневая папка с изображениями.
            test_size (float): Доля данных для тестирования.
            batch_size (int): Размер батча.
            lr (float): Скорость обучения.
            epochs (int): Количество эпох.
            augment_save_dir (str, optional): Папка для сохранения аугментированных изображений.
        """
        self.root_dir = root_dir
        self.test_size = test_size
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.augment_save_dir = augment_save_dir

        self.model = CustomCNNForDigits()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.transform = transforms.Compose([
            transforms.Resize((50, 50)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.augment_transform = transforms.Compose([
            transforms.Resize((50, 50)),
            transforms.RandomRotation(10),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def prepare_datasets(self):
        """Подготовка обучающего и тестового датасетов."""
        all_images = []
        for label in range(10):
            folder_path = os.path.join(self.root_dir, str(label))
            if os.path.isdir(folder_path):
                for image_name in os.listdir(folder_path):
                    if image_name.endswith(('.png', '.jpg', '.jpeg')):
                        all_images.append((os.path.join(folder_path, image_name), label))
        
        train_images, test_images = train_test_split(all_images, test_size=self.test_size, random_state=42)

        self.train_dataset = DigitsDataset(
            train_images, 
            transform=self.transform, 
            augment_transform=self.augment_transform, 
            augment_save_dir=self.augment_save_dir
        )
        self.test_dataset = DigitsDataset(test_images, transform=self.transform)


        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def train(self):
        """Обучение модели на тренировочном датасете."""
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            for img, y in progress_bar:
                self.optimizer.zero_grad()
                outputs = self.model(img)
                y = y.long()
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() / len(self.train_loader)
                progress_bar.set_postfix(loss=running_loss)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {running_loss:.4f}")

    def evaluate(self):
        """Оценка модели на тестовом датасете.

        Returns:
            tuple: Метрики оценки (точность, F1-метрика, средняя абсолютная ошибка).
        """
        self.model.eval()
        pred, actuals = [], []
        total_loss = 0.0

        progress_bar = tqdm(self.test_loader, desc="Evaluating")
        with torch.no_grad():
            for img, y in progress_bar:
                outputs = self.model(img)
                y_pred = outputs.argmax(dim=1)
                total_loss += self.criterion(outputs, y.long()).item()
                pred.extend(y_pred.tolist())
                actuals.extend(y.tolist())
        
        accuracy = accuracy_score(actuals, pred)
        f1 = f1_score(actuals, pred, average='weighted') 
        cm = confusion_matrix(actuals, pred)  
        mae = mean_absolute_error(actuals, pred)

        print(f"Validation Loss: {total_loss/len(self.test_loader):.4f}, Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}, MAE: {mae:.4f}")
        print("Confusion Matrix:")
        print(cm)

        return accuracy, f1, mae

    def save_model(self, path='models/CustomCNNForDigits.pth'):
        """Сохранение обученной модели.

        Args:
            path (str): Путь для сохранения модели.
        """
        torch.save(self.model.state_dict(), path)
        print(f"Model saved as {path}")


if __name__ == "__main__":
    pipeline = DigitsClassificationPipeline(root_dir="data/digits", epochs=10)
    pipeline.prepare_datasets()
    pipeline.train()
    pipeline.evaluate()
    # pipeline.save_model()