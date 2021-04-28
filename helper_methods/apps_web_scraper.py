import pandas as pd
from google_play_scraper import Sort, reviews

class AppsWebScrapper:
    def __init__(self, app, country_name):
        self.app = app
        self.country_name = country_name

    def get_gplay_data_as_dataframe(self):
        user_app_input = self.app
        review_google_play, continuation_token= reviews(
        user_app_input,
        lang='pt_BR', # defaults to 'en'
        country=self.country_name, # defaults to 'us'
        sort=Sort.NEWEST,
        count = 200 # defaults to Sort.MOST_RELEVANT
        )
        review_google_play, _ = reviews(
        user_app_input,
        continuation_token=continuation_token # defaults to None(load from the beginning)
        )
        review_google_play_df = pd.DataFrame(review_google_play)
        return review_google_play_df

    def get_lot_of_app_reviews(self, app_packages):
        app_reviews = []
        for ap in app_packages:
            for score in list(range(1, 6)):
                for sort_order in [Sort.MOST_RELEVANT, Sort.NEWEST]:
                    rvs, _ = reviews(
                        ap,
                        lang='pt_BR',
                        country=self.country_name,
                        sort=sort_order,
                        count= 200,
                        filter_score_with=score
                    )
                    for r in rvs:
                        r['sortOrder'] = 'most_relevant' if sort_order == Sort.MOST_RELEVANT else 'newest'
                        r['appId'] = ap
                    app_reviews.extend(rvs)
        app_reviews_df = pd.DataFrame(app_reviews)
        return app_reviews_df
    
    
    