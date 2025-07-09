import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import itertools
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE

class ACLoss(nn.Module):
    def __init__(self, temperature=1.0, ignore_index=-100):
        super().__init__()
        self.temperature = temperature
        self.ignore_index = ignore_index
        
    def forward(self, logits, target):
      
        B, C = logits.shape
        
        target_mask = F.one_hot(target, num_classes=C).to(logits.dtype)
        
        masked_logits = logits - 9999 * target_mask 
        _, res_max = torch.max(masked_logits, dim=1)  
        resmax_mask = F.one_hot(res_max, num_classes=C).to(logits.dtype) 
        
        adjusted_logits = logits.clone()
        probs = torch.softmax(logits.detach(), dim=1) 
        
        non_target_mask = 1.0 - target_mask 
        intensity = (1.0 - probs.gather(1, target.unsqueeze(1)))  
        adjusted_logits += self.temperature * non_target_mask * intensity
        
        final_mask = 1.0 - target_mask - resmax_mask 
        final_logits = adjusted_logits - 9999 * final_mask 
        
        loss = F.cross_entropy(final_logits, target, 
                              reduction='mean',
                              ignore_index=self.ignore_index)
        
        return loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def plot_confusion_matrix(cm, labels_name, title, acc, output_path=None):
    print('*********************************')
    cm = cm / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    num_class = np.array(range(len(labels_name)))
    plt.xticks(num_class, labels_name, rotation=90)
    plt.yticks(num_class, labels_name)
    plt.ylabel('Target')
    plt.xlabel('Prediction')
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.tight_layout()
    if output_path is None:
        output_path = os.path.join('./Confusion_matrix', title)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    plt.savefig(os.path.join(output_path, "acc" + str(acc) + ".png"), format='png')
    plt.show()


def load_pretrained_weights(model, checkpoint):
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    # new_state_dict.requires_grad = False
    model_dict.update(new_state_dict)

    model.load_state_dict(model_dict)
    print('load_weight', len(matched_layers))
    return model

def plot_tsne(features, labels, labels_name, dataset_name, save_path='tsne_visualization.png'):
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(12, 10))
    num_classes = len(labels_name)
    colors = plt.cm.get_cmap('tab10', num_classes)
    
    for i in range(num_classes):
        mask = labels == i
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                    c=[colors(i)], 
                    label=labels_name[i], 
                    alpha=0.6,
                    edgecolors='w',
                    linewidth=0.5)
    
    plt.title(f't-SNE Visualization ({dataset_name})')
    plt.xlabel('TSNE-1')
    plt.ylabel('TSNE-2')
    plt.legend(title='Categories')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()