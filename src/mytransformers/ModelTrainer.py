import numpy as np
from tqdm import tqdm
import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import torch.nn as nn
from torch.utils.data import Dataset
from numpy.typing import NDArray
import matplotlib.pyplot as plt


class ModelTrainer:
    """
    тренирует модель, сохраняет и визуализирует потери, проводит тесты
    """
    def __init__(self,
                 num_epochs: int,
                 batch_size: int,
                 pad_token_id: int = 0,
                 device: torch.device | None = None):
        
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id
        self.device = device
        
        if self.device is None:
            self.device = torch.device('cuda' if cuda.is_available() else 'cpu')
            
        self.model = None
        self.loss_history = None
        self.criterion = None
        
    @property
    def model_(self):
        return self.model
            
    def train(self,
              data: Dataset,
              model: nn.Module,
              optimizer: Optimizer,
              criterion: nn.Module,
              ) -> NDArray[np.float64]:
        """
        Обучает модель на датасете, где каждый элемент — словарь с 'source' и 'target'
        Пример: {'source': tensor([...]), 'target': tensor([...])}
        """
        dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=True)
        model = model.to(self.device)
        self.criterion = criterion
        
        model.train()
        loss_history = []

        with tqdm(total=len(dataloader) * self.num_epochs, unit="batch", position=0, leave=True) as pbar:
            for epoch in range(self.num_epochs):
                running_loss = 0.0

                for batch_num, batch in enumerate(dataloader):
                    source = batch['source'][:, :-1].to(self.device)
                    target = batch['target'].to(self.device)

                    target_input = target[:, :-1]
                    target_labels = target[:, 1:]

                    optimizer.zero_grad()
                    
                    outputs = model(source, target_input)

                    loss = criterion(
                        outputs.contiguous().view(-1, outputs.size(-1)),
                        target_labels.contiguous().view(-1)
                    )

                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()

                    running_loss += loss.item()
                    pbar.set_postfix({
                        "Epoch": f"{epoch + 1}/{self.num_epochs}",
                        "Loss": f"{loss.item():.4f}"
                    })
                    pbar.update()

                avg_epoch_loss = running_loss / len(dataloader)
                loss_history.append(avg_epoch_loss)
                pbar.set_postfix({"Epoch Loss": f"{avg_epoch_loss:.4f}"})

            pbar.close()
        
        self.model = model
        self.loss_history = np.array(loss_history)
        
        # Возвращаем тестовые потери
        test_losses = self.test(data)
        return test_losses
    
    def test(self, data: Dataset) -> NDArray[np.float64]:
        """
        Оценка модели на всём датасете
        """
        if self.model is None:
            raise ValueError("Модель не обучена или не установлена.")
        
        self.model.eval()
        dataloader = DataLoader(data, batch_size=self.batch_size)
        test_losses = []

        with torch.no_grad():
            for batch in dataloader:
                source = batch['source'][:, :-1].to(self.device)
                target = batch['target'].to(self.device)

                target_input = target[:, :-1]
                target_labels = target[:, 1:]

                outputs = self.model(source, target_input)
                loss = self.criterion(
                    outputs.contiguous().view(-1, outputs.size(-1)),
                    target_labels.contiguous().view(-1)
                )
                test_losses.append(loss.item())

        return np.array(test_losses)
    
    def print_loss_history(self, path_save: str | None = None) -> None:
        """
        Визуализация истории лосса по эпохам
        """
        if self.loss_history is None:
            raise ValueError("Нет данных о loss. Сначала вызовите train().")

        plt.figure(figsize=(8, 5))
        plt.plot(self.loss_history, label="Train Loss")
        plt.title("Training Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        if path_save:
            plt.savefig(f"{path_save}.png", dpi=200, bbox_inches='tight')
            print(f"График сохранён: {path_save}.png")
        else:
            plt.show()