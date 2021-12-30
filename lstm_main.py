import torch
from torch import nn, optim
from data_preprocess import PADDING, read_data
from LSTM.model import LSTM_MODEL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PADDING = '[PAD]'
BATCH_SIZE = 64

train_loader, test_loader, labelmap, vocab_size, padd_idx = read_data()
label_size = len(labelmap)

model = LSTM_MODEL(vocab_size, 200, 100, label_size, padd_idx, device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model.train()
for epoch in range(1):
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        pred = model(batch_x)
        loss = criterion(pred, batch_y)  #batch_y类标签就好，不用one-hot形式   
        
        if (epoch + 1) % 10 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

test_acc_list = []
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

# test_loss /= len(test_loader.dataset)
# test_loss_list.append(test_loss)
test_acc_list.append(100. * correct / len(test_loader.dataset))
print('Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))