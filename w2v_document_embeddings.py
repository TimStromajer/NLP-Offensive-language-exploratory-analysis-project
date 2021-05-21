import gensim.downloader
import gensim
import pandas as pd
import os
import pickle
import numpy as np
import torch
from lazy_load import lazy
from scipy.spatial import distance
from nltk.corpus import stopwords
from sentence_transformers.util import pytorch_cos_sim

from text_processing import tokenize, remove_links, remove_ats, remove_consecutive_phrases, remove_retweets
from speech_classes import SPEECH_CLASSES
from dense_plotting import plotPCA, plotMDS, plotTSNE, plotDistanceMatrix

stopwords = stopwords.words('english')


def read_document(file_path):
    return pd.read_csv(file_path, dtype=str, usecols=[1, 2])


def embed_document(model, tokens):
    total_vector = np.zeros([model.vector_size], dtype=np.float32)
    normalization = 0
    for token in tokens:
        if token in stopwords:
            # print(f"skipped: {token}")
            continue
        if token in model:
            embedding = model[token]
            total_vector += embedding
            normalization += 1
    if normalization == 0:
        return total_vector
    return total_vector / normalization


def embed_documents(model, posts):
    embedded_posts = list()
    for i, post in enumerate(posts):
        print(100*i/len(posts))
        post = remove_links(post)
        post = remove_ats(post)
        posts = remove_retweets(post)
        tokens = tokenize(post)
        tokens = remove_consecutive_phrases(tokens)
        embedded_post = embed_document(model, tokens)
        embedded_posts.append(embedded_post)
    embedded_posts.reverse()
    return embedded_posts


def load_or_create(path, action, recreate=False):
    if not recreate and os.path.exists(path):
        with open(path, "rb") as f:
            print(f"Loaded from {path}")
            output = pickle.load(f)
    else:
        output = action()
        with open(path, "wb") as f:
            pickle.dump(output, f)
            print(f"Generated and saved to {path}")
    return output


# Calculate distances between two groups of embeddings (lists or arrays of embeddings)
# Returns the distance plus indexes of the closest from a to b and closest from b to a
def document_group_similarities(a_embeddings, b_embeddings):
    distances = pytorch_cos_sim(a_embeddings, b_embeddings)
    a_to_b_averages = torch.mean(distances, dim=1)
    b_to_a_averages = torch.mean(distances, dim=0)
    top_a_to_b = torch.argsort(a_to_b_averages, descending=True)
    top_b_to_a = torch.argsort(b_to_a_averages, descending=True)
    # print([a_to_b_averages[i.item()] for i in top_a_to_b[:5]])
    # print([b_to_a_averages[i.item()] for i in top_b_to_a[:5]])
    a_to_b = torch.mean(a_to_b_averages)
    b_to_a = torch.mean(b_to_a_averages)
    return a_to_b, b_to_a, top_a_to_b, top_b_to_a


intermediate_location = f"{os.path.basename(__file__)}-intermediate_data"
if not os.path.exists(intermediate_location):
    os.mkdir(intermediate_location)


if __name__ == '__main__':
    regenerate = False
    def load_w2v():
        print("Loading model...")
        w2v_model = gensim.downloader.load('word2vec-google-news-300')
        print("Loaded.")
        return w2v_model
    model = lazy(load_w2v)
    input_files = ['9.csv',
                   '21.csv',
                   '25.csv',
                   '26.csv',
                   '31.csv',
                   '32.csv',
                   'jigsaw-toxic.csv'
                   ]

    # Load the data into these variables
    all_posts = list()
    all_labels = list()
    all_embeddings = list()

    for file_name in input_files:
        data_path = 'data'
        file_path = os.path.join(data_path, file_name)
        if not os.path.exists(file_path):
            print(f"Skipping {file_path}, because source is not available")
            continue
        data = read_document(file_path)
        data = data[data['class'] != '0']
        posts = list(data['text'])
        labels = list(data['class'])
        embedding_file_path = os.path.join(intermediate_location, f"{file_name}-embeddings.p")
        post_embeddings = load_or_create(embedding_file_path, lambda: embed_documents(model, posts), recreate=regenerate)

        all_posts.extend(posts)
        all_labels.extend(labels)
        all_embeddings.extend(post_embeddings)
        print(len(post_embeddings))

    # Move data into dictionaries {label: list of embeddings}, {label: list of texts}
    labels_embeddings = dict()
    label_posts = dict()
    for i in range(len(all_labels)):
        label = all_labels[i]
        embedding = all_embeddings[i]
        if label not in labels_embeddings:
            labels_embeddings[label] = list()
            label_posts[label] = list()
        labels_embeddings[label].append(np.array(all_embeddings[i]))
        label_posts[label].append(all_posts[i])

    labels = list(labels_embeddings.keys())
    labels = [SPEECH_CLASSES[int(label)] for label in labels]

    top_count_print = 5
    top_count_visualize = 30

    # Combined based representative posts
    # Finds posts closest to the combined vector of the class
    embedding_totals = list()
    embedding_top_similar = list()
    posts_top_similar = list()

    for label in labels_embeddings:
        combined = sum(labels_embeddings[label])
        combined = combined/len(labels_embeddings[label])
        embedding_totals.append(combined)
        count = len(labels_embeddings[label])
        centroid = combined/count
        distances = distance.cdist([centroid], labels_embeddings[label], "cosine")[0]
        distances = [e for e in distances if e < 1]
        sort = [i[0] for i in sorted(enumerate(distances), key=lambda x:x[1])]
        print("==============")
        print(SPEECH_CLASSES[int(label)])
        for i in range(top_count_print):
            no_links = remove_links(label_posts[label][sort[i]])
            no_ats = remove_ats(no_links)
            remove_repeating = remove_consecutive_phrases(no_ats.split())
            remove_repeating = " ".join(remove_repeating)
            print(f"{i+1}: {remove_repeating}")

        label_embedding_top = list()
        label_post_top = list()
        for i in range(top_count_visualize):
            label_embedding_top.append(labels_embeddings[label][sort[i]])
            label_post_top.append(label_posts[label][sort[i]])
        embedding_top_similar.append(label_embedding_top)
        posts_top_similar.append(label_post_top)



    # Similarity of the combined embeddings of each label
    # similarity = pytorch_cos_sim(embedding_totals, embedding_totals)
    # similarity = similarity.numpy()
    #
    # plotMDS("MDS Totals", labels, [[tot] for tot in embedding_totals])
    # plotPCA("PCA Totals", labels, [[tot] for tot in embedding_totals])
    # plotDistanceMatrix("Document Similarity", labels, similarity)

    # Distance based representative
    # Finds the posts closest based on the lowest distance to all posts withn the class

    print()
    print()
    print()
    print("DISTANCE BASED")
    for i, (label, embeddings) in enumerate(labels_embeddings.items()):
        label_distance, _, indexs, _ = document_group_similarities(embeddings, embeddings)
        print("==============")
        print(SPEECH_CLASSES[int(label)])
        print(label_distance)
        for j in range(top_count_print):
            no_links = remove_links(label_posts[label][indexs[j].item()])
            # no_ats = remove_ats(no_links)
            # no_rts = remove_retweets(no_ats)
            remove_repeating = remove_consecutive_phrases(no_links.split())
            remove_repeating = " ".join(remove_repeating)
            print(f"{j+1}: {remove_repeating}")
