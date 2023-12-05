from sklearn.isotonic import IsotonicRegression
import torch

def isotonic(dataloader, model):
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for input, label in dataloader:
            input = input
            logits = torch.sigmoid(model(input))
            logits_list.append(logits)
            labels_list.append(label)
        logits = torch.flatten(torch.cat(logits_list))
        labels = torch.flatten(torch.cat(labels_list))
    
    iso_reg = IsotonicRegression().fit(logits, labels)
    return iso_reg