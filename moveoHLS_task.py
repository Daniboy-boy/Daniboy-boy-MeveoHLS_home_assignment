import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from gensim.models import LdaModel


def get_soup_from_url(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.content, "html.parser")
            return soup
        else:
            print("Failed to retrieve the webpage:", response.status_code)
            return None
    except Exception as e:
        print("An error occurred:", e)
        return None


def extract_claims(soup, claims_list):
    """
    I assume here that google patents page has a specific structure for the claims section.
    """
    # Find the claims section
    claims_section = soup.find('section', itemprop='claims')
    #print(claims_section)
    # If claims section is found
    # Find the h2 tag within the claims section
    h2_tag = claims_section.find('h2')

    # Find the span tag within the h2 tag
    span_tag = h2_tag.find('span', itemprop='count')
    num_claims = int(span_tag.text.strip())
    print("Total claims found:", span_tag.text.strip())
    # Find all claim text divs
    claim_divs = soup.find_all("div", class_="claim")
    if claims_section:
        for i in range(0, num_claims):
            raw_claim = claim_divs[i].get_text(strip=True)
            if raw_claim not in claims_list:
                claims_list.append(remove_prefix_and_dot(raw_claim))


def remove_prefix_and_dot(string):
    # Find the index of the first dot
    dot_index = string.find(".")

    if dot_index != -1:  # If dot is found
        # Extract the substring after the dot
        rest_of_string = string[dot_index + 1:].strip()
        return rest_of_string
    else:
        return string  # If dot is not found, return the original string


def infer_topics(sentences, num_topics=3, topn_words=5):
    """
    One more challnge was to infer a topic for each cluster we got from the model.
    I used LDA model to infer the topic for each cluster.

    Note1: I wanted to genarete labels for each cluster using an OpenAI model but I couldn't get a key to use the API.
    example of the code I wanted to use can be find here: https://huggingface.co/MaartenGr/BERTopic_ArXiv

    Note2: I tried to use this model: https://huggingface.co/cristian-popa/bart-tl-all
    but it generated not quite accurate labels for the clusters.
    """
    # Combine all sentences into one document
    nltk.download('stopwords')
    nltk.download('punkt')

    combined_document = ' '.join(sentences)

    # Tokenize and preprocess the combined document
    stop_words = set(stopwords.words('english'))
    tokenized_text = word_tokenize(combined_document.lower())
    tokenized_text = [word for word in tokenized_text if word.isalnum() and word not in stop_words]

    # Create a dictionary and corpus for LDA
    dictionary = Dictionary([tokenized_text])
    corpus = [dictionary.doc2bow(tokenized_text)]

    # Apply LDA
    lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)

    # Get the most representative words for each topic
    topic_words = [lda_model.show_topic(topic_id, topn=topn_words) for topic_id in range(lda_model.num_topics)]

    # Infer the topic for the combined document
    bow_vector = dictionary.doc2bow(tokenized_text)
    topic_distribution = lda_model.get_document_topics(bow_vector)
    inferred_topic_id = max(topic_distribution, key=lambda x: x[1])[0]
    inferred_topic_label = ', '.join(word for word, _ in topic_words[inferred_topic_id])

    return inferred_topic_label

def group_claims(patent_path, num_clusters):
    # Get the claims from the patent
    claims = []
    soup = get_soup_from_url(patent_path)
    extract_claims(soup, claims)

    df = pd.DataFrame(claims, columns=['Strings'])

    model = SentenceTransformer('all-MiniLM-L6-v2')
    df['encode_transformer'] = df['Strings'].apply(lambda text: model.encode(text, convert_to_numpy=True).flatten())
    features_transformers = np.vstack(df['encode_transformer'])

    agg = AgglomerativeClustering(n_clusters=num_clusters)
    agg.fit(features_transformers)
    clusters = agg.labels_

    clusters_result_name = f'cluster_'
    df[clusters_result_name] = clusters

    grouped_df = df.groupby(clusters_result_name)['Strings'].apply(lambda x: '\n'.join(x)).reset_index()
    output = ""
    # create array out of the strings in each cluster
    for index, row in grouped_df.iterrows():
        sentences = row['Strings'].split('\n')
        cluster_topic = infer_topics(sentences)
        output += f"Group name: {cluster_topic}: {len(sentences)} claims.  "
        df.loc[df[clusters_result_name] == row[clusters_result_name], clusters_result_name] = f"{row[clusters_result_name]}: {cluster_topic}"

    return output


if __name__ == "__main__":
    # test
    site = "https://patents.google.com/patent/US9980046B2/en?oq=US9980046B2"
    claims = []
    soup = get_soup_from_url(site)
    extract_claims(soup, claims)
    num = 0
    group_claims(site, 3)

