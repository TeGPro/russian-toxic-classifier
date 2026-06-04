import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from dataset import ToxicDataset
from model import BertClassifier

def train(model, train_loader, optimizer, loss_fn, device, epochs=3):
    print(f"Начинаем обучение на {device}...")
    history = {'loss': [], 'acc': []}
    
    for epoch in range(epochs):
        print(f"\n--- Эпоха {epoch + 1}/{epochs} ---")
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Эпоха {epoch+1}")

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            pred = outputs.argmax(dim=1)
            batch_correct = (pred == labels).sum().item()
            batch_total = labels.size(0)
            batch_acc = (batch_correct / batch_total) * 100

            total_loss += batch_loss
            correct += batch_correct
            total += batch_total

            progress_bar.set_postfix({
                'batch_loss': f"{batch_loss:.4f}",
                'batch_acc': f"{batch_acc:.2f}%"
            })
            
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = (correct / total) * 100
        print(f"-> Итог Эпохи {epoch+1} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%\n")
        
        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_acc)
        
    return history

def plot_history(history):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], marker='o', color='crimson', label='Train Loss')
    plt.title('Динамика потерь (Loss)')
    plt.xlabel('Эпоха')
    plt.ylabel('Значение Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['acc'], marker='o', color='royalblue', label='Train Accuracy')
    plt.title('Динамика точности (Accuracy)')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    print("Графики обучения успешно сохранены в файл 'learning_curves.png'")

if __name__ == "__main__":
    # Настройки гиперпараметров
    EPOCHS = 3
    BATCH_SIZE = 32
    LR = 2e-5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Загрузка исходного датасета...")
    raw_dataset = load_dataset('Mnwa/russian-toxic')
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
    
    print("Подготовка тренировочных данных...")
    train_dataset = ToxicDataset(raw_dataset['train'], tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = BertClassifier(n_classes=2).to(device)
    optimizer = AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    
    # Запуск
    history = train(model, train_loader, optimizer, loss_fn, device, epochs=EPOCHS)
    
    # Сохраняем веса
    torch.save(model.state_dict(), 'bert_toxic_classifier.pt')
    print("Веса модели успешно сохранены в 'bert_toxic_classifier.pt'")
    
    # Строим графики
    plot_history(history)