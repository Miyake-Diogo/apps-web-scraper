import pandas as pd
#import nltk
#nltk.download('all')
import streamlit as st
from app_store_scraper import AppStore
from google_play_scraper import Sort, reviews_all
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image

def main():
    st.set_page_config(page_title="Apps Web Scraper", page_icon=":rocket:", layout="wide", initial_sidebar_state="expanded")

    st.title(':rocket: Apps Web Scraper - Seu app de exploração de comentários e feedbacks do Google Play e AppStore :rocket:')
    page = st.sidebar.selectbox("Choose a page", ["Home", "Exploration"])
    
    #user_app_input = st.text_input("Digite o endereço do app a buscar os comentários (Gplay)", 'com.facebook.katana')

    #app_to_Review = AppStore(country="br", app_name="xp-investimentos")
    #review_appstore = app_to_Review.review()
    #appstore_reviews_df = pd.DataFrame(review_appstore)
    
    ## Google Play App Store
    #user_app_input = 'com.amazon.avod.thirdpartyclient' 
    #review_google_play = reviews_all(
    #user_app_input,
    #sleep_milliseconds=0, # defaults to 0
    #lang='pt_BR', # defaults to 'en'
    #country='br'#, # defaults to 'us'
    #sort=Sort.NEWEST # defaults to Sort.MOST_RELEVANT
    #)

    #app_reviews_df = pd.DataFrame(review_google_play)
    
    app_reviews_df = pd.read_csv("Data/google_play_reviews.csv")
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
    #stopwords_list = nltk.corpus.stopwords.words('portuguese')
    ## Reduced DataFrame
    
    #df_reduzido = app_reviews_df[['content','sentiment']]

    #df_positive = df_reduzido[df_reduzido['sentiment']=='Positive']
    #df_negative = df_reduzido[df_reduzido['sentiment']=='Negative']
    #df_neutral = df_reduzido[df_reduzido['sentiment']=='Neutral']
    #maj_class1 = resample(df_positive, replace=True, n_samples=1736, random_state=123) 
    #maj_class2 = resample(df_negative, replace=True, n_samples=1736, random_state=123)

    #df_final=pd.concat([df_neutral,maj_class1,maj_class2])

    ## Deletando o que não for necessário
    #del maj_class1, maj_class2, df_positive, df_neutral, df_negative

    if page == "Home":

        st.header("Bem vindo ao Apps Web Scraper!")
        
        #image = Image.open('Data/XDATA.jpeg')
        #st.image(image, caption='XDATA - XP INC.', use_column_width=True)

        st.text("O Apps Web Scraper veio para melhorar suas decisões baseadas em reviews da AppStore e do GooglePlay.")

        st.text("Para este protótipo foi adicionado os dados de um app (Lojas Americanas) para teste.")

        st.text("Existem duas páginas até o Momento: Home e Exploration")

        st.text("Home: Pagina Inicial")
        st.text("Exploration: Pagina com pequenas informações sobre os dados capturados")
        #st.text("Sentiment Analiser: Pagina para testar o classificador de analise de sentimentos")

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
        #def get_word_clouds(df):
        #    words = []
        #    for i in df.content:
        #        for p in i.lower().split():
        #            if p not in stopwords_list:
        #                words.append(p)
        #    words = str(words)            
        #    return words        

        #st.header('Nuvem de palavras mais utilizadas')
        #word_cloud = WordCloud(width=1100, height=900, margin=0).generate(get_word_clouds(df_reduzido))
        #plt.figure(figsize=(20,11))
        #plt.imshow(word_cloud, interpolation='bilinear')
        #plt.axis('off')
        #plt.margins(x=0,y=0)
        #plt.show()
        #st.pyplot()

        ## Tabelão 
        st.header('Tabelão - Filtre conforme necessidade')
        cols = ["reviewId","content", "score", "thumbsUpCount", "reviewCreatedVersion", "at", "replyContent", "repliedAt"]
        num_rows = st.sidebar.slider('Selecione a quantidade de linhas para o Tabelão', min_value=1, max_value=50)
        st_multiselect = st.sidebar.multiselect("Selecione as colunas para o tabelão", app_reviews_df.columns.tolist(), default=cols)
        tabelao_df = app_reviews_df[st_multiselect].head(num_rows)
        st.table(tabelao_df)


if __name__ == "__main__":
    main()
