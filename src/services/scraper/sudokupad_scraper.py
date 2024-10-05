from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options
from webdriver_manager.firefox import GeckoDriverManager
from time import sleep

class SudokupadScraper:

    binary_location = '/snap/firefox/current/usr/lib/firefox/firefox'

    def __init__(self):
        self.web_driver = self.__setup_web_driver()
        self.soup = None

    def __del__(self):
        self.web_driver.quit()

    def scrape_url(self, url: str) -> dict:
        self.web_driver.get(url)
        sleep(1)

        html = self.web_driver.page_source
        self.soup = BeautifulSoup(html, 'html.parser')

        return {
            'rules': self.__get_rules()
        }

    def __setup_web_driver(self):
        options = Options()
        options.add_argument("--headless")
        options.binary_location = self.binary_location
        options.profile = webdriver.FirefoxProfile()

        gecko_driver_manager = GeckoDriverManager()
        gecko_driver = gecko_driver_manager.install()
        service = FirefoxService(gecko_driver)

        web_driver = webdriver.Firefox(service=service, options=options)
        return web_driver

    def __get_rules(self) -> str:
        rules_section = self.soup.find('div', class_='puzzle-rules selectable')
        return rules_section.text if rules_section else 'No rules available'

