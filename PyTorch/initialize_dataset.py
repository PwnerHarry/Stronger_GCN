from utils import *
import torch


adj, features, labels = load_data('cora')
dense_adj = adj.to_dense()
torch.save(dense_adj, "cora_dense_adj.pt")
torch.save(features, "cora_features.pt")
torch.save(labels, "cora_labels.pt")

adj, features, labels = load_data('citeseer')
dense_adj = adj.to_dense()
torch.save(dense_adj, "citeseer_dense_adj.pt")
torch.save(features, "citeseer_features.pt")
torch.save(labels, "citeseer_labels.pt")

adj, features, labels = load_data('pubmed')
dense_adj = adj.to_dense()
torch.save(dense_adj, "pubmed_dense_adj.pt")
torch.save(features, "pubmed_features.pt")
torch.save(labels, "pubmed_labels.pt")

del adj, dense_adj, features, labels
