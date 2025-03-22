import numpy as np

def load_embeddings(file_path):
    """
    Загружает эмбеддинги из файла.
    
    Формат файла:
      Первая строка: "<vocab_size> <d>"
      Далее: каждая строка содержит слово и его d координат через пробел.
      
    Возвращает:
      embeddings: матрица эмбеддингов (numpy.ndarray shape=(vocab_size, d))
      word2index: словарь, отображающий слово -> индекс
      index2word: список слов, где index2word[i] соответствует i-му эмбеддингу.
    """
    with open(file_path, encoding='utf-8') as f:
        header = f.readline().strip().split()
        vocab_size = int(header[0])
        d = int(header[1])
        embeddings = np.zeros((vocab_size, d), dtype=np.float32)
        word2index = {}
        index2word = []
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) != d + 1:
                continue  # Пропускаем строки с неверным числом элементов
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            embeddings[i] = vector
            word2index[word] = i
            index2word.append(word)
    return embeddings, word2index, index2word

# ближайшие слова для выбранного слова
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
  
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import random

d = 100

embeddings_file = 'embeddings/harrypotter_embeddings_{}_bpe.txt'
embeddings_file_norm = 'embeddings/harrypotter_embeddings_{}_bpe_norm.txt'
embeddings, word2index, index2word = load_embeddings(embeddings_file.format(d))

query_word = random.choice(list(word2index.keys()))
# query_word = 'война'
for d in [100]:
    embeddings, word2index, index2word = load_embeddings(embeddings_file.format(d))
    if query_word in word2index:
        query_index = word2index[query_word]
        query_vec = embeddings[query_index]
        similarities = {}
        for idx in range(embeddings.shape[0]):
            word = index2word[idx]
            vec = embeddings[idx]
            similarities[word] = cosine_similarity(query_vec, vec)
        # Сортируем по убыванию сходства
        nearest = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"[d={d}] Слова, ближайшие к '{query_word}':")
        for word, sim in nearest:
            print(f"{word}: {sim:.4f}")
    else:
        print(f"Слово '{query_word}' не найдено в словаре.")


# Reduce the number of words to visualize for clarity
num_words_to_visualize = min(3000, len(index2word))
selected_indices = random.sample(range(len(index2word)), num_words_to_visualize)

embeddings, word2index, index2word = load_embeddings(embeddings_file.format(d))

# Extract the embeddings and corresponding words
selected_embeddings = embeddings[selected_indices]
selected_words = [index2word[i] for i in selected_indices]

# Perform t-SNE to reduce dimensions to 3D
tsne = TSNE(n_components=3, random_state=42, perplexity=30, max_iter=300)
embeddings_3d = tsne.fit_transform(selected_embeddings)

# Plot the 3D visualization
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(121, projection='3d')

plt.title(f"3D t-SNE Visualization of Word Embeddings, d={d} (not normalized)")

# Scatter plot
ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], c='blue', marker='o')

# # Annotate points with words
# for i, word in enumerate(selected_words):
#     ax.text(embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2], word, fontsize=8)

embeddings, word2index, index2word = load_embeddings(embeddings_file_norm.format(d))

# Extract the embeddings and corresponding words
selected_embeddings_norm = embeddings[selected_indices]
selected_words_norm = [index2word[i] for i in selected_indices]

# Perform t-SNE to reduce dimensions to 3D
tsne_norm = TSNE(n_components=3, random_state=42, perplexity=30, max_iter=300)
embeddings_3d_norm = tsne_norm.fit_transform(selected_embeddings_norm)

# Plot the 3D visualization
# fig2 = plt.figure(figsize=(10, 10))
ax_norm = fig.add_subplot(122, projection='3d')

# Scatter plot
ax_norm.scatter(embeddings_3d_norm[:, 0], embeddings_3d_norm[:, 1], embeddings_3d_norm[:, 2], c='red', marker='o')

# # Annotate points with words
# for i, word in enumerate(selected_words):
#     ax2.text(embeddings_3d2[i, 0], embeddings_3d2[i, 1], embeddings_3d2[i, 2], word, fontsize=8)

plt.title(f"3D t-SNE Visualization of Word Embeddings, d={d} (normalized)")

plt.show()