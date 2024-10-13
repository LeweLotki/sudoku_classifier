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
        
        try:
            return {
                'rules': self.__get_rules(),
                'difficulty': self.__get_difficulty(),
                'puzzle_type': self.__get_puzzle_type(),
                'comments': self.__get_comments()      
            }
        except Exception as e:
            print(e)
            return {}

    def __get_difficulty(self) -> str:
        difficulty_row = self.soup.find(lambda tag: tag.name == 'td' and ('Schwierigkeit' in tag.get_text() or 'Difficulty' in tag.get_text()))
        difficulty = None
        if difficulty_row:
            difficulty = difficulty_row.find('img').get('title')
        return difficulty or 'Unknown'

    def __get_rules(self) -> str:
        rules_section = self.soup.find('div', class_='rp_html')
        if not rules_section:
            return 'No rules available'

        rules = ' '.join([p.get_text(strip=True) for p in rules_section.find_all('p')])
        return rules

    def __get_puzzle_type(self) -> str:
        type_section = self.soup.find('div', class_='rp_tags')
        if not type_section:
            return 'Unknown'

        puzzle_types = [tag.get_text(strip=True) for tag in type_section.find_all('a')]
        
        return ', '.join(puzzle_types) if puzzle_types else 'Unknown'

    def __get_comments(self) -> str:
        """Extracts all comments from the puzzle page."""
        # Find the heading for comments, usually <h3> with text 'Kommentare' or 'Comments'
        comments_heading = self.soup.find(lambda tag: tag.name == 'h3' and ('Kommentare' in tag.get_text() or 'Comments' in tag.get_text()))

        if not comments_heading:
            print("Comments heading not found")
            return 'No comments available'
        
        # Attempt to locate the comments section immediately following the heading
        comments_section = comments_heading.find_next_sibling('div')
        
        if not comments_section:
            print("Comments section not found after the heading")
            return 'No comments available'
        
        comments = []
        
        # Iterate through the <div> tags immediately after the comments heading
        for comment_div in comments_section.find_all('div', recursive=False):
            # Extract all paragraphs <p> inside each <div> as individual comments
            comment_text = ' '.join([p.get_text(separator=' ', strip=True) for p in comment_div.find_all('p', recursive=False)])
            if comment_text:  # If the comment is not empty, add it
                comments.append(comment_text)
        
        # If no <div> found inside comments_section, let's try directly accessing <p> tags inside comments_section
        if not comments:
            comments = [p.get_text(separator=' ', strip=True) for p in comments_section.find_all('p', recursive=False)]
        
        # Print the raw comments section for debugging
        print("Extracted comments:", comments)
        
        return ' '.join(comments) if comments else 'No comments available'

