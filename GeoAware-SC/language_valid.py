import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss

# Define Embedding layer
vocab_size = 32000
embedding_dim = 4096
embedding_layer = torch.load("/mnt/bn/shijiaynas/clip_embed_layer.pt")

# Get the token embeddings
token_embeddings = embedding_layer.weight.data.cpu().to(torch.float32)[6635]

# Example image features (replace with actual features)
image_features = torch.load("/mnt/bn/shijiaynas/cat.pt").to(torch.float32).cpu()

# Normalize embeddings and image features
normalized_image_features = F.normalize(image_features, p=2, dim=1).cpu()
normalized_token_embeddings = F.normalize(token_embeddings.unsqueeze(0), p=2, dim=1).cpu()

# Convert tensors to numpy arrays for FAISS
# image_features_np = normalized_image_features.to(torch.float32).cpu().numpy()
# token_embeddings_np = normalized_token_embeddings.to(torch.float32).cpu().numpy()

# Initialize FAISS index
# print(token_embeddings.shape)
# exit()
# index = faiss.IndexFlatL2(embedding_dim)
# index.add(token_embeddings)

cosine_sim = torch.matmul(normalized_image_features, normalized_token_embeddings.T)
most_similar_index = torch.argmax(cosine_sim)
most_similar_score = cosine_sim[most_similar_index].item()
print("Most similar index:", most_similar_index.item())
print("Similarity score:", most_similar_score)

# # Perform search
# print(image_features_np.shape)
# _, nearest_token_ids = index.search(image_features, 1)

# # # Convert result back to tensor
# nearest_token_ids = torch.tensor(nearest_token_ids).squeeze()

# similarity = torch.matmul(image_features, token_embeddings.T)
# nearest_token_ids = torch.argmax(similarity, dim=1)

# print(nearest_token_ids.shape)
# print(nearest_token_ids)