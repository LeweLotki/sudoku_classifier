import requests
import logging
from urllib.parse import urljoin
from bs4 import BeautifulSoup

from .logic_master_scraper import LogicMasterScraper
# from .sudokupad_scraper import SudokupadScraper
from .code_generator import alphanumeric_code_generator

from sqlalchemy.orm import Session
from ..database import SessionLocal
from ..tables.puzzles import Puzzle


class Scraper:
    logic_master_base_url = 'https://logic-masters.de/Raetselportal/Raetsel/zeigen.php'

    def __init__(self):
        logging.basicConfig(filename='services.log', level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logic_master_scraper = LogicMasterScraper()
        # self.sudokupad_scraper = SudokupadScraper()
        self.code_generator = alphanumeric_code_generator('000000')  

    def scrape_puzzles(self, number_of_puzzles: int = 100) -> None:
        for _ in range(number_of_puzzles):
            puzzle_code = next(self.code_generator)
            puzzle_url = f"{self.logic_master_base_url}?id={puzzle_code}"
            
            try:
                logic_master_data = self.logic_master_scraper.scrape_url(url=puzzle_url)
                # if 'sudoku_pad_ref' in logic_master_data and logic_master_data['sudoku_pad_ref']:
                #    sudoku_pad_data = self.sudokupad_scraper.scrape_url(logic_master_data['sudoku_pad_ref'])
                #    logic_master_data['rules'] = sudoku_pad_data.get('rules', 'No rules available')
                
                print(self.__is_data_complete(logic_master_data))
                if self.__is_data_complete(logic_master_data):
                    self.__save_to_db(puzzle_code, logic_master_data)
                    self.logger.info(f"Successfully scraped and saved puzzle: {puzzle_code}")
                    print(f"Successfully scraped and saved puzzle: {puzzle_code}")
                else:
                    self.logger.info(f"Skipping incomplete puzzle: {puzzle_code}")

            except Exception as e:
                self.logger.error(f"Error scraping puzzle with code {puzzle_code}: {e}")

    def __is_data_complete(self, data: dict) -> bool:
        required_fields = ['difficulty', 'rules'] 

        for field in required_fields:
            if not data.get(field) or data[field] in ['Unknown', 'No rules available', None, '']:
                return False

        return True

    def __save_to_db(self, puzzle_code: str, data: dict) -> None:
        session: Session = SessionLocal()
        try:
            new_puzzle = Puzzle(
                code=puzzle_code,
                rules=data.get('rules', 'No rules available'),
                difficulty=data.get('difficulty', 'Unknown')
            )
            session.add(new_puzzle)
            session.commit()
            self.logger.info(f"Data saved successfully for puzzle: {puzzle_code}")
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to save puzzle data: {e}")
        finally:
            session.close()

    def scrape_url(self, url: str) -> None:
        try:
            logic_master_data = self.logic_master_scraper.scrape_url(url=url)

            if 'sudoku_pad_ref' in logic_master_data and logic_master_data['sudoku_pad_ref']:
                sudoku_pad_data = self.sudokupad_scraper.scrape_url(logic_master_data['sudoku_pad_ref'])
                logic_master_data['rules'] = sudoku_pad_data.get('rules', 'No rules available')

            if self.__is_data_complete(logic_master_data):
                puzzle_code = url.split('=')[-1]  
                self.__save_to_db(puzzle_code, logic_master_data)
                print(f"Successfully scraped and saved puzzle: {puzzle_code}")
            else:
                print(f"Skipping incomplete puzzle from URL: {url}")

        except Exception as e:
            self.logger.error(f"Error scraping puzzle from URL {url}: {e}")
            print(f"Error scraping puzzle from URL {url}: {e}")

