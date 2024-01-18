import os
import re
import numpy as np
import pandas as pd
import spacy
from functools import lru_cache
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances
import syllapy
import tabulate

nlp = spacy.load("en_core_web_sm")

class TextAnalyzer:
    def __init__(self, texts):
        self.texts = texts
        self.dfms = {}

    @staticmethod
    def read_texts_from_folder(folder_path):
        return {os.path.splitext(filename)[0]: open(os.path.join(folder_path, filename), 'r', encoding='utf-8').read().lower().replace("-", ' ').replace('\n', ' ')
                for filename in os.listdir(folder_path) if filename.endswith(".txt")}

    def _vectorize(self, ngram_range=(1, 1), stop_words=None, dfm_key='base'):
        vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words)
        dfm = vectorizer.fit_transform(self.texts.values())
        self.dfms[dfm_key] = pd.DataFrame(dfm.toarray(), columns=vectorizer.get_feature_names_out(), index=self.texts.keys())

    def tokenize(self, ngram_range=(1, 1), dfm_key='base', remove_stopwords=True):
        self._vectorize(ngram_range=ngram_range, stop_words='english' if remove_stopwords else None, dfm_key=dfm_key)

    def display_data(self, data, headers, header, tablefmt="grid", numalign="center"):
        table = tabulate.tabulate(data, headers=headers, tablefmt=tablefmt, numalign=numalign)
        separator = "=" * len(header)
        print(f"\n{header}\n{separator}\n{table}\n{separator}\n")

    def display_token_counts(self, dfm_key='base', header="Token Counts"):
        dfm_selected = self.dfms[dfm_key]
        data = ((key, total, unique) for key, total, unique in zip(self.texts.keys(), dfm_selected.sum(axis=1), dfm_selected.astype(bool).sum(axis=1)))
        self.display_data(data, ["Text", "Total Tokens", "Unique Tokens"], header)

    def display_specific_token_counts(self, tokens_to_count, dfm_key='base', header="Specific Token Counts"):
        token_counts_per_text = self.dfms[dfm_key][tokens_to_count].copy()
        token_counts_per_text.loc['Total'] = token_counts_per_text.sum(numeric_only=True)
        data = token_counts_per_text.reset_index().values.tolist()
        self.display_data(data, ["Text"] + tokens_to_count, header)

    def fog_index(self, text, use_spacy=False):
        if use_spacy:
            doc = nlp(text)
            num_sentences = len(list(doc.sents))
            words = [token.text for token in doc]
        else:
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            num_sentences = len(sentences)

        num_words = len(words)
        num_complex = sum(1 for word in words if syllapy.count(word) >= 3)
        avg_sentence_length = num_words / num_sentences
        percent_complex = num_complex / num_words * 100

        return 0.4 * (avg_sentence_length + percent_complex)

    def compute_fog_indexes(self, use_spacy=False):
        return {name: self.fog_index(text, use_spacy=use_spacy) for name, text in self.texts.items()}
    
    def cosine_similarity_analysis(self, dfm_key='base'):
        pairwise_distances = pd.DataFrame(cosine_distances(self.dfms[dfm_key]), index=self.texts.keys(), columns=self.texts.keys())
        
        np.fill_diagonal(pairwise_distances.values, np.inf)
        most_similar_texts = pairwise_distances.idxmin().to_dict()
        
        np.fill_diagonal(pairwise_distances.values, -np.inf)
        least_similar_texts = pairwise_distances.idxmax().to_dict()
        
        return most_similar_texts, least_similar_texts

    def generate_freq_matrix(self, dfm_key='base'):
        dfm = self.dfms[dfm_key]
        freq_matrix = pd.DataFrame(dfm.transpose(), columns=self.texts.keys())
        freq_matrix['Total'] = freq_matrix.sum(axis=1)
        return freq_matrix.transpose()

    def top_features(self, dfm_key='base', n=5):
        freq_matrix = self.generate_freq_matrix(dfm_key=dfm_key)
        top_features_per_text = {text: freq_matrix.loc[text].nlargest(n) for text in self.texts.keys()}
    
        data = []
        for text, features in top_features_per_text.items():
            data.append((text, ", ".join([f"{feature} ({int(count)})" for feature, count in features.items()])))
    
        header = "(j). Top 5 Features for Each Text"
        self.display_data(data, ["Text", "Top 5 Features"], header)
    
    def display_overall_top_features(self, dfm_key='base', n=5):
        freq_matrix = self.generate_freq_matrix(dfm_key=dfm_key)
        overall_top_features = freq_matrix.loc['Total'].nlargest(n)
        print("\nTop 5 Overall Features\n" + "="*25)
        print(", ".join([f"{feature} ({int(count)})" for feature, count in overall_top_features.items()]))
        print("="*25 + "\n")
