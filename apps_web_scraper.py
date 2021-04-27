import pandas as pd
from app_store_scraper import AppStore
from google_play_scraper import Sort, reviews_all

class AppsWebScrapper:
    def __init__(self, app_name, store_name, country_name):
        self.app_name = app_name
        self.store_name = store_name
        self.country_name = country_name

    def get_gplay_data(self, app_name, country_name):
        user_app_input = self.app_name 
        review_google_play = reviews_all(
        user_app_input,
        sleep_milliseconds=0, # defaults to 0
        lang='pt_BR', # defaults to 'en'
        country=self.country_name#, # defaults to 'us'
        sort=Sort.NEWEST # defaults to Sort.MOST_RELEVANT
        )


    def get_appstore_data(self, app_name, country_name):
        app_to_Review = AppStore(country=self.country_name, app_name=self.app_name)
        review_appstore = app_to_Review.review()
        return review_appstore
    
    def get_reviews_data(self, store_name, app_name, country_name):
        if (self.store_name == "gplay"):
            results = self.get_gplay_data(self.app_name, self.country_name)
        elif(self.store_name == "appstore"):
            results = self.get_gplay_data(self.app_name, self.country_name)
        else:
            print("Exception: Please pass gplay or appstore")

    def make_dataframe_from_scrap(self, scrap_data):
        appstore_reviews_df = pd.DataFrame(scrap_data)
    