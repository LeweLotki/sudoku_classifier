import requests
from bs4 import BeautifulSoup


class LogicMasterScraper:

    def __init__(self):
        self.soup = None
        self.url = None

    def scrape_url(self, url: str) -> dict:
        self.url = url
        response = requests.get(url)

        if response.status_code != 200:
            print(f"Failed to access {url}")
            return {}

        self.soup = BeautifulSoup(response.text, 'html.parser')

        return {
            'logic_master_ref': url,
            'sudoku_pad_ref': self.__get_sudoku_pad_ref(),
            'difficulty': self.__get_difficulty()
        }

    def __get_sudoku_pad_ref(self) -> str:
        sudoku_pad_ref = None
        sudoku_pad_ref_tag = self.soup.find('a', href=lambda href: href and ('crackingthecryptic' in href or 'sudokupad' in href))
        if sudoku_pad_ref_tag:
            sudoku_pad_ref = sudoku_pad_ref_tag['href']
        return sudoku_pad_ref

    def __get_difficulty(self) -> str:
        difficulty_row = self.soup.find(lambda tag: tag.name == 'td' and ('Schwierigkeit' in tag.get_text() or 'Difficulty' in tag.get_text()))
        difficulty = None
        if difficulty_row:
            difficulty = difficulty_row.find('img').get('title')
        return difficulty or 'Unknown'

