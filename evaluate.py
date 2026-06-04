import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from dataset import ToxicDataset
from model import BertClassifier

def evaluate(model, data_loader, loss_fn, device):
    print(f"Начинаем проверку модели на отложенной выборке...")
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Валидация")

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix({
                'val_loss': f"{loss.item():.4f}"
            })

    final_loss = total_loss / len(data_loader)
    final_acc = (correct / total) * 100
    
    print("\n" + "="*30)
    print("🏆 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ ТЕСТА:")
    print(f"Средний Loss: {final_loss:.4f}")
    print(f"Точность (Accuracy): {final_acc:.2f}%")
    print("="*30 + "\n")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Загрузка проверочного датасета...")
    raw_dataset = load_dataset('Mnwa/russian-toxic')
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
    
    test_key = 'test' if 'test' in raw_dataset.keys() else 'validation'
    test_dataset = ToxicDataset(raw_dataset[test_key], tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = BertClassifier(n_classes=2)
    try:
        model.load_state_dict(torch.load('bert_toxic_classifier.pt', map_location=device))
        print("Обученные веса успешно загружены в модель!")
    except FileNotFoundError:
        print("Ошибка: Файл весов 'bert_toxic_classifier.pt' не найден. Сначала запусти train.py")
        exit()
        
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    
    evaluate(model, test_loader, loss_fn, device)