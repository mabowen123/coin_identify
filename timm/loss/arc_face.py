import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, margin=0.5):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        
        # Initialize class centers as model parameters
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, targets):
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        
        # Normalize class centers
        weight = F.normalize(self.weight, dim=1)
        
        # Compute cosine similarity between embeddings and class centers
        cosine_similarity = F.linear(embeddings, weight)
        
        # Get the ground truth class score for each sample
        ground_truth_scores = cosine_similarity[torch.arange(embeddings.size(0)), targets].unsqueeze(1)
        
        # Compute the margin-based loss
        loss = torch.acos(ground_truth_scores) + self.margin
        cos_m = torch.cos(loss)
        sin_m = torch.sin(loss)
        ground_truth_scores = ground_truth_scores * cos_m - sin_m * (1 - ground_truth_scores)
        
        # Scale the logits
        ground_truth_scores *= self.scale
        
        # Compute the final loss
        loss = F.cross_entropy(ground_truth_scores, targets)
        return loss
