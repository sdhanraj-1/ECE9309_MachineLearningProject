import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import pearsonr

"""
Computes the mean and median of consine similarity matrix.

Parameter:
- similarity_matrix: Similarity matrix that was returned from consine_similarity calculation.

Returns:
- similarity_mean: Mean value of top 10 recommended papers' similarity score.
- similarity_median: Median value of top 10 recommended papers' similarity score.
"""
def cosine_similarity_mean_median(similarity_matrix):
    # Sort the similarity matrix in reverse order and select the top 10 data points
    top_n = 10
    top_indices = np.argsort(-similarity_matrix, axis=1)[:, :top_n]  

    # Store the similarity values of all top 10 recommended papers for all test papers
    similarity_matrix_top10 = []  
    for i, test_idx in enumerate(top_indices):
        similarity_row_top10 = []
        for j, train_idx in enumerate(test_idx):
            similarity_row_top10.append(similarity_matrix[i, train_idx])
        similarity_matrix_top10.append(similarity_row_top10)

    similarity_mean = np.mean(similarity_matrix_top10, axis=1)
    similarity_median = np.median(similarity_matrix_top10, axis=1)

    return similarity_mean, similarity_median

"""
Computes TF-IDF representations for test papers and top 10 recommended papers.

Parameters:
- test_texts: List of strings (test papers).
- top10_texts: List of lists (each inner list contains 10 recommended papers as strings).

Returns:
- tfidf_matrix: TF-IDF matrix for all papers.
- test_vectors: TF-IDF vectors for test papers.
- top10_vectors: TF-IDF vectors for top 10 recommendations per test paper.
"""
def compute_tfidf_similarity(test_texts, top10_texts):
    # Flatten all texts to fit TF-IDF on the entire dataset
    all_texts = test_texts + [paper for sublist in top10_texts for paper in sublist]

    # Initialize and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Split TF-IDF matrix into test papers and recommended papers
    num_test_papers = len(test_texts)
    test_vectors = tfidf_matrix[:num_test_papers]  # First 'num_test_papers' rows
    top10_vectors = tfidf_matrix[num_test_papers:].toarray().reshape(num_test_papers, 10, -1)  # Reshape for 10 recommendations

    return tfidf_matrix, test_vectors, top10_vectors

"""
Computes Pearson’s correlation coefficient between each test paper and each of its top 10 recommended papers.

Parameters:
- test_embeddings: A numpy array of shape (num_test_papers, embedding_dim), where each row is the vector representation of a test paper.
- top10_embeddings: A numpy array of shape (num_test_papers, 10, embedding_dim), where each row contains embeddings for the top 10 recommended papers corresponding to each test paper.

Returns:
- correlation_mean: The mean Pearson correlation of the top 10 recommendations per test paper.
- correlation_median: The median Pearson correlation per test paper.
"""
def compute_pearson_correlation(test_embeddings, top10_embeddings):
    num_test_papers = test_embeddings.shape[0]
    
    correlation_mean = []
    correlation_median = []

    for i in range(num_test_papers):
        test_paper_vector = test_embeddings[i]  # Vector for the test paper
        top10_vectors = top10_embeddings[i]  # Vectors for the top 10 recommendations

        correlations = []
        for recommended_vector in top10_vectors:
            # Compute Pearson correlation between test paper and recommended paper
            corr, _ = pearsonr(test_paper_vector, recommended_vector)
            correlations.append(corr)
        
        # Compute mean and median Pearson correlation for the top 10 papers
        correlation_mean.append(np.mean(correlations))
        correlation_median.append(np.median(correlations))
    
    return correlation_mean, correlation_median