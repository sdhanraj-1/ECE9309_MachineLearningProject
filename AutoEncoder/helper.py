import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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
Plots the Mean and Median of Cosine Similarity on Test Set

Parameters:
- similarity_mean: The output from cosine_similarity_mean_median
- similarity_median: The output from cosine_similarity_mean_median

Returns:
"""
def plot_cosine_similarity(similarity_mean, similarity_median):
    # Plot each array as a separate line
    plt.figure(figsize=(30, 5))
    plt.plot(similarity_mean, linestyle="-", label="Mean")
    plt.plot(similarity_median, linestyle="-", label="Median")
    plt.xlabel("Test Paper")
    plt.ylabel("Cosine Similiarity")
    plt.title("Mean and Median of Cosine Similarity on Test Set")
    plt.legend()
    #plt.savefig("plot.png", dpi=300, bbox_inches="tight")
    plt.show()

"""
Plots the Mean and Median of Pearson Correlation on Test Set

Parameters:
- correlation_mean: The output from cosine_similarity_mean_median
- correlation_median: The output from cosine_similarity_mean_median

Returns:
"""
def plot_pearson_correlation(correlation_mean, correlation_median):
    # Plot each array as a separate line
    plt.figure(figsize=(30, 5))
    plt.plot(correlation_mean, linestyle="-", label="Mean")
    plt.plot(correlation_median, linestyle="-", label="Median")
    plt.xlabel("Test Paper")
    plt.ylabel("Pearson Correlation")
    plt.title("Mean and Median of Pearson Correlation on Test Set")
    plt.legend()
    #plt.savefig("plot.png", dpi=300, bbox_inches="tight")
    plt.show()
    
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
    #all_texts = test_texts + [paper for sublist in top10_texts for paper in sublist]

    # Flatten all texts to fit TF-IDF on the entire dataset
    #all_texts = list(map(str, test_texts)) + [str(paper) for sublist in top10_texts for paper in sublist]

    # Ensure inputs are lists of strings
    test_texts = list(test_texts) if isinstance(test_texts, np.ndarray) else test_texts
    top10_texts = list(top10_texts) if isinstance(top10_texts, np.ndarray) else top10_texts

    all_texts = test_texts + top10_texts  # Combine test and recommended texts
    
    # Ensure `all_texts` is a list of strings
    all_texts = [str(text) for text in all_texts]
    # Initialize and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Split TF-IDF matrix into test papers and recommended papers
    num_test_papers = len(test_texts)
    test_vectors = tfidf_matrix[:num_test_papers]  # First 'num_test_papers' rows
    top10_vectors = tfidf_matrix[num_test_papers:].toarray().reshape(num_test_papers, 10, -1)  # Reshape for 10 recommendations

    return tfidf_matrix, test_vectors, top10_vectors

def compute_tfidf_similarity_v(test_texts, top10_texts):
    # Flatten all texts to fit TF-IDF on the entire dataset
    #all_texts = test_texts + [paper for sublist in top10_texts for paper in sublist]

    # Flatten all texts to fit TF-IDF on the entire dataset
    #all_texts = list(map(str, test_texts)) + [str(paper) for sublist in top10_texts for paper in sublist]

    # Ensure inputs are lists of strings
    test_texts = list(test_texts) if isinstance(test_texts, np.ndarray) else test_texts
    top10_texts = list(top10_texts) if isinstance(top10_texts, np.ndarray) else top10_texts

    all_texts = test_texts + top10_texts  # Combine test and recommended texts
    
    # Ensure `all_texts` is a list of strings
    all_texts = [str(text) for text in all_texts]
    # Initialize and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Split TF-IDF matrix into test papers and recommended papers
    num_test_papers = len(test_texts)
    test_vectors = tfidf_matrix[:num_test_papers]  # First 'num_test_papers' rows
    top10_vectors = tfidf_matrix[num_test_papers:].toarray().reshape(num_test_papers, 10, -1)  # Reshape for 10 recommendations

    return tfidf_matrix, test_vectors, top10_vectors, vectorizer
    
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
"""
Creates a word cloud from the top 5 recommended papers

Parameters:
- paper_ids: A list of ids of the top 5 recommended papers
- papers: The cleaned papers dataset from database_clean.csv as a pandas DataFrame

Returns:
- A word cloud plot of the top 5 recommended papers
"""

def createWCloud(paper_ids:list, papers:pd.DataFrame):
    # Concatenate into a single string
    text_list = papers.loc[
        papers["id"].isin(paper_ids), ["title", "abstract"]
    ].apply(
        lambda row: f"{row['title']} {row['abstract']}", axis=1
    ).tolist()
    cloud_text = " ".join(text_list)

    # Generate and display word cloud from top 5 paper content
    if cloud_text.strip():
        wcObj = WordCloud(width=800, height=400, background_color='white', max_words=20).generate(cloud_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wcObj, interpolation='bilinear')
        plt.axis("off")
        plt.title("Word Cloud of Top Recommended Papers")
        plt.savefig("plot.png", dpi=300, bbox_inches="tight")
        plt.show()
        
        return wcObj