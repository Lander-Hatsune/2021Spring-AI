import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from model_cnn import cnn

from dataset import isear

torch.manual_seed(814)

def test():
    with torch.no_grad():
        correct = 0
        for i, data in enumerate(valid_loader):
            sentence, label = data[0].to(device), data[1].to(device)
            ans = model(sentence).max(1)[1]
            if (ans == label.max(1)[1]):
                correct += 1
        valid_acc = correct / i
        print('Validation done, acc: {:.3}'.format(valid_acc))

        correct = 0
        for i, data in enumerate(test_loader):
            sentence, label = data[0].to(device), data[1].to(device)
            ans = model(sentence).max(1)[1]
            if (ans == label.max(1)[1]):
                correct += 1
        test_acc = correct / i
        print('Test done, acc: {:.3}'.format(test_acc))

        return (valid_acc, test_acc)
            
train_accs, valid_accs, test_accs = [], [], []
def plot(train_acc, valid_acc, test_acc):
    train_accs.append(train_acc)
    valid_accs.append(valid_acc)
    test_accs.append(test_acc)
    plt.clf()
    x = np.arange(0, len(train_accs))
    plt.plot(x, train_accs, label='train_acc')
    plt.plot(x, valid_accs, label='valid_acc')
    plt.plot(x, test_accs, label='test_acc')
    plt.legend()
    plt.pause(0.01)

BATCH_SIZE = 50
EPOCH = 50

train_set = isear("train")
valid_set = isear("validation")
test_set = isear("test")

train_loader = torch.utils.data.DataLoader(train_set, shuffle=True,
                                           batch_size=BATCH_SIZE)
valid_loader = torch.utils.data.DataLoader(valid_set, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device: ', device)

model = cnn()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters())

for epoch in range(EPOCH):
    train_loss = .0
    total = 0
    correct = 0
    for i, data in enumerate(train_loader):
        sentences, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(sentences)
        correct += sum(outputs.max(1)[1] == labels.max(1)[1]).cpu()
        loss = criterion(outputs, labels.max(1)[1])
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        total += len(labels)

    train_acc = correct / total
    print('epoch {:3}, batch {:3}, loss {:.3}, acc {:.3}'.format(
        epoch + 1, i + 1, train_loss / 100, train_acc))
    train_loss = .0

    valid_acc, test_acc = test()
    plot(train_acc, valid_acc, test_acc)

plt.savefig('cnn.png')
plt.show()


torch.save(model.state_dict(), 'cnn.pth')
torch.save(model, 'cnn-full.pth')

