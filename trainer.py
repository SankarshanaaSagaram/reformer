import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.autograd
from .dataloader import class_weights_tensor
from .model import model_fin

optimizer = torch.optim.AdamW(model_fin.parameters(),lr=lr)

loss = nn.CrossEntropyLoss(weight=class_weights_tensor)

train_sampler = RandomSampler(trainReformer)

train_dataloader = Dataloader(train, sampler=train_sampler, batch_size = batch_size)

#epochs = 10

def training(config):

    train(config['epochs'], config['batch_size'], config['device'], config['lr'], config['beta'])

    model_fin.train()#
    total_loss, total_accuracy = 0, 0

    total_preds = []

    for step, batch in enumerate(train_dataloader):

        if step%50 == 0 and not step == 0:
            print('  Batch {:>5} of {:>5}.'.format(step, len(train_dataloader)))

        batch = [r.to(device) for r in batch]

        sent_id, mask, labels = batch

        # clear previously calculated gradients 
        model_fin.zero_grad()        

        # get model predictions for the current batch
        preds = model_fin(sent_id, mask)

        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        nn.utils.clip_grad_norm_(model_fin.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds=preds.detach().cpu().numpy()

    # append the model predictions
    total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)
  
      # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    #returns the loss and predictions
    return avg_loss, total_preds

# empty lists to store training and validation loss of each epoch
train_losses=[]

#for each epoch
for epoch in range(epochs):
     
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    
    #train model
    train_loss,_ = training()
    
    # append training and validation loss
    train_losses.append(train_loss)
    
    print(f'\nTraining Loss: {train_loss:.3f}')
    
    wandb.log({'train_loss': train_loss})
    
    #load weights of best model
path = 'saved_weights.pt'
model.load_state_dict(torch.load(path))