import torch
from torch import nn, Tensor


class CosineSimilarityLoss(nn.Module):
    def __init__(self, model):
        super(CosineSimilarityLoss, self).__init__()
        self.model = model
    
    def forward(self, input_features, labels):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in input_features]
        rep_a, rep_b = reps
        
        output = torch.cosine_similarity(rep_a, rep_b)
        loss_fct = nn.MSELoss()
        
        if labels is not None:
            loss = loss_fct(output, labels.view(-1))
            return loss
        else:
            return reps, output