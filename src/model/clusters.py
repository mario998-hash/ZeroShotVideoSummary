from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator  # For elbow method
from collections import Counter
from scipy import stats


class ClusterFrame():

    def __init__(self, max_clusters=10):
        self.max_clusters = max_clusters

    # Function to find the optimal number of clusters using the elbow method
    def find_optimal_clusters(self, embeddings):
        wcss = []  # Within-cluster sum of squares
        for k in range(2, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(embeddings)
            wcss.append(kmeans.inertia_)
        
        # Use the "knee" or "elbow" point to find the optimal number of clusters
        elbow = KneeLocator(range(2, self.max_clusters + 1), wcss, curve="convex", direction="decreasing")
        if elbow.knee is None:
            return 2
        return elbow.knee  # Optimal number of clusters

    # Function to cluster and evaluate results
    def cluster_data(self, embeddings, n_clusters=None):
        # Use k-means if n_clusters is provided
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)

        labels = kmeans.fit_predict(embeddings)
        
        return labels

    # Main function to automate the pipeline
    def automate_clustering(self, embeddings):
        n_clusters = self.find_optimal_clusters(embeddings)
        labels = self.cluster_data(embeddings, n_clusters=n_clusters)

        return labels
    
    #
    def segment_contribution(self, labels_for_segment, segment_embeddings):
        """
        Calculate the contribution score of a segment based on its consistency and dissimilarity.
        
        Parameters:
        - labels_for_segment: List of cluster labels for each frame in the segment.
        - segment_embeddings: Embedding vectors for the frames in the segment.
        - w_consistency: Weight for the consistency (homogeneity) score.
        - w_dissimilarity: Weight for the dissimilarity score.
        
        Returns:
        - weighted contribution score for the segment.
        """
        
        # Calculate homogeneity (consistency) score
        mode_label, count = stats.mode(labels_for_segment)
        consistency_score = count / len(labels_for_segment)  # Homogeneity score
        
        # Calculate intra-segment dissimilarity score
        distances = np.linalg.norm(segment_embeddings - np.mean(segment_embeddings, axis=0), axis=1)
        dissimilarity_score = np.mean(distances).item()  # Intra-segment dissimilarity


        return consistency_score[0], dissimilarity_score

    #
    def segment_labels(self, labels, n_frames, frames_per_segment):
        segments_labels = []
        segment_indcies = []
        for i in range(0, n_frames, frames_per_segment):
            curr_segment_labels = labels[i : min(n_frames, i + frames_per_segment)]
            segment_indcies.append((i,min(n_frames, i + frames_per_segment)))
            segments_labels.append(curr_segment_labels)

        return segments_labels, segment_indcies 






