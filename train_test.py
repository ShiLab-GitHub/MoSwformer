import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import data_set
from model import formerClassifier, MOG
from sklearn.metrics import precision_recall_fscore_support, f1_score

# 训练函数
def train(model, device, train_loader, criterion, optimizer):
    model.train()  # 将模型设置为训练模式
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, sample in enumerate(train_loader):
        # 传送到GPU或CPU
        # data, target = sample['data'].to(device), sample['label'].to(device)
        data1, data2, data3, target = sample['data1'].to(device), sample['data2'].to(device), sample['data2'].to(device), sample['label'].to(device)
        # 重置梯度
        optimizer.zero_grad()
        # 前向传播
        output = model(data1, data2, data3)
        # 计算损失
        loss = criterion(output, target)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

        # for accuracy
        running_loss += loss.item() * data1.size(0)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    train_loss = running_loss / len(train_loader.dataset)
    accuracy = correct / total
    print(f'Train Loss: {train_loss:.6f}, Train Accuracy: {accuracy:.4f}')


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    all_targets = []
    all_predicted = []

    with torch.no_grad():
        for sample in test_loader:
            # data, target = sample['data'].to(device), sample['label'].to(device)
            data1, data2, data3, target = sample['data1'].to(device), sample['data2'].to(device), sample['data2'].to(device), sample['label'].to(device)
            output = model(data1, data2, data3)
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
            all_targets.extend(target.view_as(predicted).cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    # Calculate Macro Precision, Recall, and F1 Score and convert to percentages
    precision, recall, f1m, _ = precision_recall_fscore_support(all_targets, all_predicted, average='macro')
    f1w = f1_score(all_targets, all_predicted, average='weighted')

    print(f'Test Loss: {test_loss:.6f}, '
          f'Test Accuracy: {accuracy:.4f}, '
          f'Macro Precision: {precision:.4f}, '
          f'Macro Recall: {recall:.4f}, '
          f'Macro F1: {f1m:.4f}, '
          f'weighted F1: {f1w:.4f}')
    # 创建DataFrame并保存到CSV
    df_predictions = pd.DataFrame({'Predicted_Label': all_predicted})
    df_predictions.to_csv('predicted_labels_BLCA.csv', index=False)

    return accuracy, f1w, f1m


if __name__ == '__main__':
    input_size = 2503
    output_size = 5

    batch_size = 32
    data_folder = './BRCA_split/BRCA'

    train_dataset = data_set(data_folder, train_flag = True)
    test_dataset = data_set(data_folder, train_flag = False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义模型
    # model = formerClassifier(input_size, output_size).to(device)
    model = MOG(d_model=64, nhead=8, dim_feedforward=256, num_layers=1).to(device)
    # 损失函数（交叉熵损失通常用于多分类)
    criterion = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    # optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 定义要经过的迭代次数
    num_epochs = 200

    # 训练和测试模型
    best_acc = -1
    f1w_with_bestacc = -1
    f1m_with_bestacc = -1
    best_epoch = 1
    early_stop_count = 0
    for epoch in range(1, num_epochs + 1):
        print(f'\nEpoch {epoch}:')
        train(model, device, train_loader, criterion, optimizer)
        acc, f1w, f1m = test(model, device, test_loader, criterion)

        if acc >= best_acc:
            best_acc = acc
            f1w_with_bestacc = f1w
            f1m_with_bestacc = f1m
            best_epoch = epoch
            early_stop_count = 0
        else:
            early_stop_count = early_stop_count + 1
        print("Test BEST epoch:", best_epoch)
        print("Test BEST ACC: {:.3f}".format(best_acc))
        print("Test F1 weighted: {:.3f}".format(f1w_with_bestacc))
        print("Test F1 macro: {:.3f}".format(f1m_with_bestacc))

        if early_stop_count >= 30:
            print('Early Stop.')
            break