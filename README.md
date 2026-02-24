# Stock-Price-Prediction


## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset
### Problem Statement:

Predicting stock prices is a complex task due to market volatility. Using historical closing prices, we aim to develop a Recurrent Neural Network (RNN) model that can analyze time-series data and generate future price predictions.

### Dataset:

The dataset consists of historical stock closing prices from trainset.csv and testset.csv. The data is normalized using MinMax scaling, and sequences of 60 past values are used as input features. The model learns patterns from training data to predict upcoming prices, helping traders and investors make informed decisions.
## Train Dataset
<img width="522" height="427" alt="image" src="https://github.com/user-attachments/assets/a2d14b92-ad0e-4ac3-94ad-c83abfb6456d" />

## Test Dataset

<img width="485" height="500" alt="image" src="https://github.com/user-attachments/assets/d540df32-39a6-4063-886d-0789873559e2" />

## Design Steps
## Step 1:
Data Collection & Preprocessing: Load historical stock prices, normalize using MinMaxScaler, and create sequences for time-series input.
## Step 2:
Model Design: Build an RNN with two layers, define input/output sizes, and set activation functions.
## Step 3:
Training Process: Train the model using MSE loss and Adam optimizer for 20 epochs with batch-size optimization.
## Step 4:
Evaluation & Prediction: Test on unseen data, inverse transform predictions, and compare with actual prices.
## Step 5:
Visualization & Interpretation: Plot training loss and predictions to analyze performance and potential improvements.

## Program
#### Name: HARISH KUMAR S
#### Register Number: 212224230091

```Python 
# Define RNN Model
class RNNModel(nn.Module):
    def __init__(self,input_size=1,hidden_size=64,num_layers=2,output_size=1):
      super(RNNModel,self).__init__()
      self.rnn=nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
      self.fc=nn.Linear(hidden_size,output_size)

    def forward(self, x):
      out, _ = self.rnn(x)
      out = self.fc(out[:, -1, :])
      return out


model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Train the Model

epochs = 20
model.train()
train_losses = []
for epoch in range(epochs):
  epoch_loss = 0
  for x_batch, y_batch in train_loader:
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    optimizer.zero_grad()
    outputs = model(x_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
  train_losses.append(epoch_loss / len(train_loader))
  print(f"Epoch [{epoch+1}/{epochs}], Loss:{train_losses[-1]:.4f}")

```

## Output

### True Stock Price, Predicted Stock Price vs time
<img width="926" height="517" alt="image" src="https://github.com/user-attachments/assets/3d80e95e-f7ba-401f-b7e0-e5f7c861f039" />




### Predictions 
<img width="1161" height="652" alt="image" src="https://github.com/user-attachments/assets/50a52637-1e67-4d72-bb56-ad864c4e3e0b" />



## Result

Thus, a Recurrent Neural Network model for stock price prediction has successfully been devoloped successfully.
