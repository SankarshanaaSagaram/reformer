





# get predictions for test data
with torch.no_grad():
    preds = model(test_dataset.to(device), test_mask.to(device))
    preds = preds.detach().cpu().numpy()

# model's performance
preds = np.argmax(preds, axis = 1)
print(classification_report(test_y, preds))