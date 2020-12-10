import streamlit as st
import pandas as pd
from app_store_scraper import AppStore
from google_play_scraper import Sort, reviews_all
import nltk
nltk.download('stopwords')
from nltk.tokenize import  word_tokenize
from wordcloud import WordCloud
from nltk.metrics import ConfusionMatrix
from sklearn.model_selection import train_test_split


def main():
    st.title("XRate_App - Seu app de análise de sentimentos do Google Play e AppStore(future)")
    ## get data from AppStore
    #xp = AppStore(country="br", app_name="xp-investimentos")
    #xp_appstore = xp.review()
    #xp_app_reviews_df = pd.DataFrame(xp_appstore)
    #xp_app_reviews_df.to_csv('xp_app_store_reviews.csv', index=None, header=True)

    ## Get Data from Google Play
    #br.com.xp.carteira
    xp_google_play = reviews_all(
        'br.com.xp.carteira',
        sleep_milliseconds=0, # defaults to 0
        lang='pt_BR', # defaults to 'en'
        country='br'#, # defaults to 'us'
        #sort=Sort.NEWEST # defaults to Sort.MOST_RELEVANT
    )

    app_reviews_df = pd.DataFrame(xp_google_play)
    #app_reviews_df.to_csv('xp_google_play_reviews.csv', index=None, header=True)

    ## create columns in pandas for year_month, day and sentiment
    app_reviews_df['year_month'] = app_reviews_df['at'].dt.strftime('%Y-%m')
    app_reviews_df['day'] = app_reviews_df['at'].dt.strftime('%d')

    def gplay_sentiment(df):
    if df['score'] < 3:
        return 'Negative'
    elif df['score'] == 3:
        return 'Neutral'
    elif df['score'] > 3:
        return 'Positive'
    else:
        return 'Undefined'

    app_reviews_df['manual_sentiment'] = app_reviews_df.apply(gplay_sentiment, axis=1)

    ## Create of StopWords
    stopwords_list = nltk.corpus.stopwords.words('portuguese')
    ## Reduced DataFrame
    df_reduzido = app_reviews_df[['content','manual_sentiment']]

    ## Spliting Data from Train and Test
    treino, teste = train_test_split(df_reduzido, test_size=0.3)

    ## WordCloud 
    def get_word_clouds(df):
        words = []
        for i in df.content:
            for p in i.lower().split():
                if p not in stopwords_list:
                    words.append(p)
        words = str(words)            
        return words        

    #word_cloud = WordCloud(width=1000, height=800, margin=0).generate(get_word_clouds(df_reduzido))
    #plt.figure(figsize=(20,11))
    #plt.imshow(word_cloud, interpolation='bilinear')
    #plt.axis('off')
    #plt.margins(x=0,y=0)

    ## Stemming

    def stemmer_aplied(text):
        stemmer = nltk.stem.SnowballStemmer("portuguese")
        phrases_without_stemmer = []
        for (words, sentiment) in text:
            with_stemmer = [str(stemmer.stem(p)) for p in words.lower().split() if p not in stopwords_list]
            phrases_without_stemmer.append((with_stemmer, sentiment))
        return phrases_without_stemmer

    ## Steming de treino e teste
    treino = [tuple(x) for x in treino.values]
    frases_com_stem_treinamento = stemmer_aplied(treino)
    teste = [tuple(x) for x in teste.values]
    frases_com_stem_teste = stemmer_aplied(teste)

    ## Busca Palavras

    def search_words(phrases):
        all_words = []
        for (words, sentiment) in phrases:
            all_words.extend(words)
        return all_words

    def frequency_search(words):
        words = nltk.FreqDist(words)
        return words


    palavras_treinamento = search_words(frases_com_stem_treinamento)
    palavras_teste = search_words(frases_com_stem_teste)

    ## Frequencia das palavras 
    frequencia_treinamento = frequency_search(palavras_treinamento)
    frequencia_teste = frequency_search(palavras_teste)

    ## Extrator de palavras
    def search_unique_words(frequency):
        freq = frequency.keys()
        return freq

    unique_words_train = search_unique_words(frequencia_treinamento)
    unique_words_test = search_unique_words(frequencia_teste)

    def words_extractor(document):
        doc = set(document)
        characteristics = {}
        for words in unique_words_train:
            characteristics['%s' % words] = (words in doc)
        return characteristics

    def words_extractor_test(document):
        doc = set(document)
        characteristics = {}
        for words in unique_words_test:
            characteristics['%s' % words] = (words in doc)
        return characteristics

    # Base completa
    base_completa_treinamento = nltk.classify.apply_features(words_extractor, frases_com_stem_treinamento)
    base_completa_teste = nltk.classify.apply_features(words_extractor_test, frases_com_stem_teste)

    ## Classificando a Bodega
    classificador = nltk.NaiveBayesClassifier.train(base_completa_treinamento)

    ## Pegando erros e a matrix de confusão
    def get_errors():
        erros = []
        for (frase, classe) in base_completa_teste:
            resultado = classificador.classify(frase)
            if resultado != classe:
                erros.append((classe, resultado, frase))
        return erros

    def get_confusion_matrix():
        esperado = []
        previsto = []
        for (frase, classe) in base_completa_teste:
            resultado = classificador.classify(frase)
            previsto.append(resultado)
            esperado.append(classe)
        matrix = ConfusionMatrix(esperado, previsto)
        print(matrix)

    ## Sentiment Tester
    def sentiment_tester(phrase):
        stemmer = nltk.stem.SnowballStemmer("portuguese")
        test_stemming = []
        for (palavras_treinamento) in phrase.split():
            with_stem = [p for p in palavras_treinamento.split()]
            test_stemming.append(str(stemmer.stem(with_stem[0])))
        new_job = words_extractor(test_stemming)
        distribution = classificador.prob_classify(new_job)
        print('##======================##\n')
        for classe in distribution.samples():
            print('%s: %f' % (classe, distribution.prob(classe)))
            if classe == 'Positive':
                print('Probabilidade de ser positiva')
            if classe == 'Neutral':
                print('Probabilidade de ser neutra')
            if classe == 'Negative':
                print('Probabilidade de ser negativa')
        print('##======================##\n')


### Start of Streamlit App

def if __name__ == "__main__":
    main()