import sys

from services.database import initialize_database
from services.parser import initialize_parser
from services.scraper.scraper import Scraper

def main():

    initialize_database()
    args, parser = initialize_parser()

    if args.scrape:
        scraper = Scraper()
        scraper.scrape_puzzles(number_of_puzzles=10000)
    elif args.analysis:
        pass
    elif args.train:
        pass
    elif args.predict:
        pass
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()

