
from rag_pipeline import embedding_model

print("\n===== Testing Embedding Model =====")

sentences = [
    "What is an INNER JOIN in SQL?",
    "INNER JOIN returns matching records from two tables.",
    "Python is a programming language."
]

# Generate embeddings
embeddings = embedding_model.encode(
    sentences,
    normalize_embeddings=True
)

print("\nEmbedding model loaded successfully!")

print("Number of sentences:", len(sentences))
print("Embedding shape:", embeddings.shape)

# Display first embedding
print("\nFirst embedding:")
print(embeddings[0])

# Check similarity between sentences
similarity = embeddings[0] @ embeddings[1]

print("\nSimilarity between sentence 1 and 2:")
print(round(float(similarity), 4))

similarity2 = embeddings[0] @ embeddings[2]

print("\nSimilarity between sentence 1 and 3:")
print(round(float(similarity2), 4))

print("\nEmbedding test completed!")
