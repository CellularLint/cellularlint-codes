import pickle
import seaborn as sns
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from gensim.models import Word2Vec
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import pandas as pd
import sys


def tokenizer_comparison(TYPE):
    with open("./Data/cp_corpus_4G.txt", "rb") as fp:   # CP: Context Preserving
        corpus_4G = pickle.load(fp)

    with open("./Data/cp_corpus_5G.txt", "rb") as fp:   # CP: Context Preserving
        corpus_5G = pickle.load(fp)


    SEP_IDX_4G = 720 #Index where 4G Security specification ends
    SEP_IDX_5G = 1233 #Index where 5G Security specification ends
    
    
    if TYPE == '4G':
        sentences = corpus_4G
        SEP_IDX = SEP_IDX_4G
    elif TYPE == '5G':
        sentences = corpus_5G
        SEP_IDX = SEP_IDX_5G
    else:
        print("ERROR")
        
     
    # Preprocess the sentences and create TaggedDocument objects
    tagged_data = [TaggedDocument(words=sentence.lower().split(), tags=[str(i)]) for i, sentence in enumerate(sentences)]

    # Doc2Vec model training
    model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4, epochs=20)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    # Get embeddings for the sentences
    embeddings = [model.infer_vector(doc.words) for doc in tagged_data]

    # Calculate cosine similarity matrix
    cosine_sim_matrix_doc2vec = cosine_similarity(embeddings)
    
    
    # Load the Universal Sentence Encoder (USE)
    model_name = "sentence-transformers/paraphrase-mpnet-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize and get embeddings for the sentences
    tokenized_sentences = [tokenizer.encode(sentence, padding='max_length', truncation=True) for sentence in sentences]
    input_ids = torch.tensor(tokenized_sentences, dtype=torch.long).to(device)

    # Get embeddings for all sentences in batches
    batch_size = 32
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(input_ids), batch_size):
            batch_inputs = input_ids[i:i+batch_size]
            outputs = model(input_ids=batch_inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
            embeddings.append(batch_embeddings)

    # Concatenate embeddings from all batches
    embeddings = np.concatenate(embeddings, axis=0)


    # Calculate cosine similarity matrix
    cosine_sim_matrix_use = cosine_similarity(embeddings)
    
    
    # Load the Sentence-BERT model
    model_name = "sentence-transformers/distilbert-base-nli-stsb-mean-tokens"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize and get embeddings for the sentences
    tokenized_sentences = [tokenizer.encode(sentence, padding='max_length', truncation=True) for sentence in sentences]
    input_ids = torch.tensor(tokenized_sentences, dtype=torch.long).to(device)

    # Get embeddings for all sentences in batches
    batch_size = 32
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(input_ids), batch_size):
            batch_inputs = input_ids[i:i+batch_size]
            outputs = model(input_ids=batch_inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
            embeddings.append(batch_embeddings)

    # Concatenate embeddings from all batches
    embeddings = np.concatenate(embeddings, axis=0)

    # Calculate cosine similarity matrix
    cosine_sim_matrix_sbert = cosine_similarity(embeddings)
    
    # Tokenize the sentences
    tokenized_sentences = [sentence.lower().split() for sentence in sentences]

    # Word2Vec model training
    model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4, sg=0)

    # Get embeddings for the sentences
    embeddings = np.array([np.mean([model.wv[word] for word in sentence], axis=0) for sentence in tokenized_sentences])

    # Calculate cosine similarity matrix
    cosine_sim_matrix_word2vec = cosine_similarity(embeddings)
    
    
    vect = TfidfVectorizer(min_df=1, stop_words="english")                                                                                                                                                                                                   
    tfidf = vect.fit_transform(sentences)                                                                                                                                                                                                                       
    cosine_sim_matrix_tfidf = tfidf * tfidf.T
    
    plt.imshow(cosine_sim_matrix_tfidf.toarray(), cmap='hot')
    plt.colorbar()
    plt.title(f"{TYPE}",fontsize = 20)
    plt.savefig(f"heatmap_{TYPE}.pdf", dpi = 300)
    
    sbert = cosine_sim_matrix_sbert.flatten()
    doc2vec = cosine_sim_matrix_doc2vec.flatten()
    use = cosine_sim_matrix_use.flatten()
    word2vec = cosine_sim_matrix_word2vec.flatten()
    
    tfidf = cosine_sim_matrix_tfidf.toarray().flatten()
    
    flag = []
    for i in range(cosine_sim_matrix_sbert.shape[0]):
        for j in range(cosine_sim_matrix_sbert.shape[1]):
            if i<= SEP_IDX and j<= SEP_IDX:
                flag.append("Security") #only security
            elif (i<= SEP_IDX and j > SEP_IDX) or (i > SEP_IDX and j <= SEP_IDX):
                flag.append("Inter-Document") #mixed
            else:
                flag.append("NAS")
    
    
    df = pd.DataFrame(list(zip(sbert, doc2vec, use, word2vec, tfidf, flag)),
               columns =['sbert', 'doc2vec', 'use', 'word2vec', 'tfidf', 'source'])
    
    
    
    df2 = df.sample(frac = 0.008)
    
    
    long_df = df2.melt(id_vars=['source'], var_name='Embeddings', value_name='Sim_Score')
    
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    
    plt.figure(figsize=(10, 6))
    ax = sns.stripplot(data=long_df, x=f'Embeddings', y='Sim_Score', hue='source', size = 0.8, palette='magma')

    ax.set(title=f'{TYPE}')
    ax.xaxis.tick_top()
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"{TYPE}_embedding_times.png", dpi = 80)
    

    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 tokenizer_and_sim_matrix.py <TYPE>")
        print("<TYPE> = [4G] [5G]")
        sys.exit(1)
    
    argument = sys.argv[1]
    tokenizer_comparison(argument)
        
        
        
        
        
        
        
       
    
