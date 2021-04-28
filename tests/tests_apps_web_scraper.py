import unittest
import os, sys
from helper_methods.apps_web_scraper import *
# Google Play 'com.mercadolibre','com.alibaba.aliexpresshd', 'com.shopee.ph','com.contextlogic.wish'


class Test_Apps_WebScrapper(unittest.TestCase):

    def test_if_variable_gplay_is_not_none(self):
        gpst = AppsWebScrapper("com.mercadolibre", "br")
        gplay_var = gpst.get_gplay_data_as_dataframe()
        self.assertIsNotNone(gplay_var)

    def test_if_gplay_df_has_length_equals_200(self):
        gpst = AppsWebScrapper("com.mercadolibre", "br")
        gplay_var = gpst.get_gplay_data_as_dataframe()
        self.assertEqual(gplay_var.shape[0], 200)

    def test_if_gplaystore_file_exists(self):
        filename="download_data/gplaystore.csv"
        gpst = AppsWebScrapper("com.mercadolibre", "br")
        gplay_df = gpst.get_gplay_data_as_dataframe()
        gplay_df.to_csv(filename, sep=';')
        try:
            f = open(filename)
            self.assertTrue(f, True)
            # Do something with the file
        except IOError:
            print("File not accessible")
        finally:
            f.close()

    def test_if_list_of_apps_exists(self):
        list_of_apps = ['com.mercadolibre','com.alibaba.aliexpresshd']
        gpst = AppsWebScrapper("com.mercadolibre", "br")
        gplay_df = gpst.get_lot_of_app_reviews(list_of_apps)
        self.assertIsNotNone(gplay_df)
