import pandas as pd
import streamlit as st
#from app_store_scraper import AppStore
from google_play_scraper import Sort, reviews_all
import nltk
from nltk.tokenize import  word_tokenize
from wordcloud import WordCloud
from nltk.metrics import ConfusionMatrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import resample
from PIL import Image

def main():
    st.set_page_config(page_title="XRate_App", page_icon=":rocket:", layout="wide", initial_sidebar_state="expanded")

    st.title(':rocket: XRate_App - Seu app de análise de sentimentos do Google Play e AppStore(future) :rocket:')
    page = st.sidebar.selectbox("Choose a page", ["Home", "Exploration", "Sentiment Analiser"])
    app_reviews_df = pd.read_csv("Data/xp_google_play_reviews.csv")
    app_reviews_df['at'] = pd.to_datetime(app_reviews_df['at'], errors='coerce')
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

    app_reviews_df['sentiment'] = app_reviews_df.apply(gplay_sentiment, axis=1).reset_index(drop=True)
    ## StopWords
    stopwords_list = nltk.corpus.stopwords.words('portuguese')
    ## Reduced DataFrame
    df_reduzido = app_reviews_df[['content','sentiment']]

    df_positive = df_reduzido[df_reduzido['sentiment']=='Positive']
    df_negative = df_reduzido[df_reduzido['sentiment']=='Negative']
    df_neutral = df_reduzido[df_reduzido['sentiment']=='Neutral']
    maj_class1 = resample(df_positive, replace=True, n_samples=1736, random_state=123) 
    maj_class2 = resample(df_negative, replace=True, n_samples=1736, random_state=123)

    df_final=pd.concat([df_neutral,maj_class1,maj_class2])

    ## Deletando o que não for necessário
    del maj_class1, maj_class2, df_positive, df_neutral, df_negative

    if page == "Home":

        st.header("Bem vindo ao Xrate App!")
        
        image = Image.open('Data/XDATA.jpeg')
        st.image(image, caption='XDATA - XP INC.', use_column_width=True)

        st.text("O Xrate App veio para melhorar suas decisões baseadas em reviews da AppStore e do GooglePlay.")
        st.text("Existem três paginas até o Momento: Home, Exploration and Sentiment Analiser")

        st.text("Home: Pagina Inicial")
        st.text("Exploration: Pagina com pequenas informações sobre os dados capturados")
        st.text("Sentiment Analiser: Pagina para testar o classificador de analise de sentimentos")

        st.text("Fique a Vontade para dar seu FeedBack")

    if page == "Exploration":
        st.header("This is your data explorer.")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        col1, col2 = st.beta_columns(2)
        with col1:
            st.header('Quantidade de sentimentos positivos, negativos e neutros')
        #sentiment_chart = app_reviews_df.groupby(['sentiment']).sentiment.count().reset_index(drop=True)
            sentiment_table = app_reviews_df.groupby(['sentiment']).sentiment.count()
            st.table(sentiment_table)
        with col2:
            st.header('Quantidade de sentimentos (positivos, negativos e neutros) ao longo do tempo')
            line_sentiment = app_reviews_df.groupby(['year_month']).agg({'sentiment': 'count'})#.reset_index(drop=True)
            st.line_chart(line_sentiment)

        st.header('Alguns dos comentários:')
        filtro_df = st.selectbox('Filtre os dados por Score:', ('Entre 4 e 5', '3', 'Entre 2 e 1', 'todos'))
        coments = app_reviews_df[['score', 'content']]

        if (filtro_df == 'Entre 4 e 5'):
            coments = coments[coments['score']>3]
            coments = coments.sample(20)
            st.table(coments)
        elif (filtro_df == 'Entre 2 e 1'):
            coments = coments[coments['score']<3]
            coments = coments.sample(20)
            st.table(coments)
        elif (filtro_df == '3'):
            coments = coments[coments['score']==3]
            coments = coments.sample(20)
            st.table(coments)
        else:
            coments = coments.sample(20)
            st.table(coments)

        

        st.header('Quantidade de Notas do APP (de 1 a 5)')
        sentiment_chart = app_reviews_df.groupby(['score']).agg({'score': 'count'}).reset_index(drop=True)
        st.bar_chart(sentiment_chart)

        col3, col4 = st.beta_columns(2)
        with col3:
            ## Media de notas este mês
            st.header('Media das notas por mês')
            notas_media_agg = app_reviews_df.groupby(['year_month']).agg({'score': 'mean'}).reset_index()
            st.table(notas_media_agg)
            #notas_media = notas_media_agg.loc[notas_media_agg['year_month'] == '2020-12']
            #st.header('Media das notas por mês')
            #st.dataframe(notas_media)
        with col4:
            ## Quantidade de notas 5
            st.header('Quantidade de notas 5')
            notas_5 = app_reviews_df.query("score==5")["year_month"]
            fig, ax = plt.subplots()
            ax.hist(notas_5, bins=20, color='#33FFE9')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        ## WordCloud
        def get_word_clouds(df):
            words = []
            for i in df.content:
                for p in i.lower().split():
                    if p not in stopwords_list:
                        words.append(p)
            words = str(words)            
            return words        

        st.header('Nuvem de palavras mais utilizadas')
        word_cloud = WordCloud(width=1100, height=900, margin=0).generate(get_word_clouds(df_reduzido))
        plt.figure(figsize=(20,11))
        plt.imshow(word_cloud, interpolation='bilinear')
        plt.axis('off')
        plt.margins(x=0,y=0)
        plt.show()
        st.pyplot()

        ## Tabelao 
        st.header('Tabelão - Filtre conforme necessidade')
        cols = ["reviewId","content", "score", "thumbsUpCount", "reviewCreatedVersion", "at", "replyContent", "repliedAt"]
        num_rows = st.sidebar.slider('Selecione a quantidade de linhas para o Tabelão', min_value=1, max_value=50)
        st_multiselect = st.sidebar.multiselect("Selecione as colunas para o tabelão", app_reviews_df.columns.tolist(), default=cols)
        tabelao_df = app_reviews_df[st_multiselect].head(num_rows)
        st.table(tabelao_df)


    ##  Rodando a bodega toda 
    elif page == "Sentiment Analiser":

        ## Spliting Data from Train and Test
        treino, teste = train_test_split(df_final, test_size=0.3)
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
        #classificador = nltk.NaiveBayesClassifier.train(base_completa_treinamento)
        import pickle
        f = open('Data/my_classifier.pickle', 'rb')
        classificador = pickle.load(f)
        f.close()

        def sentiment_tester(phrase):
            phrase
            stemmer = nltk.stem.SnowballStemmer("portuguese")
            test_stemming = []
            retorno_infos = []
            for (palavras_treinamento) in phrase.split():
                with_stem = [p for p in palavras_treinamento.split()]
                test_stemming.append(str(stemmer.stem(with_stem[0])))
            new_job = words_extractor(test_stemming)
            distribution = classificador.prob_classify(new_job)
            
            for classe in distribution.samples():
                retorno_infos.append((phrase, classe, distribution.prob(classe)))
            df_final = pd.DataFrame(retorno_infos,columns=['frase','classe', 'probablidade da classe'])
            return df_final
        st.header('Abaixo alguns exemplos de frases que podem ser utilizadas:')    
        st.text('Exemplo de Frase Positiva: Simples , fácil e intuitivo	')
        st.text('Exemplo de Frase Positiva: Gostei do App, pois ele facilita a vida!')
        st.text('Exemplo de Frase Neutra: O aplicativo está abrindo sozinho, do nada! É só para mim que isso acontece?')
        st.text('Exemplo de Frase Neutra: Bom o app Porém o atendimento pessoal ainda a margens para melhoras') 	
        st.text('Exemplo de Frase Negativa: Não gostei,ruim, péssima experiência, toda hora pede o token que expira rapidamente, lixo de app.')
        st.text('Exemplo de Frase Negativa: O app não carrega, já troquei de celular já modifiquei meu plano de internet más o app não carrega principalmente a aba bolsa. So carrega fora do horário de pregão quando não preciso mais, fora as atualizações que dizem que é para melhorar más só piora a compreensão, estou muito decepcionado. Se vcs se consideram a maior corretora de investimentos do país façam ao menos um app descente para seus usuários pois suas taxas são altíssimas e qualidade não corresponde.')
        
        st.header('Digite uma frase abaixo para testar o classificador')
        user_input = st.text_input("Insira uma frase de uma linha para testar o classificador: ")
        user_text = sentiment_tester(user_input)
        st.table(user_text)

if __name__ == "__main__":
    main()
