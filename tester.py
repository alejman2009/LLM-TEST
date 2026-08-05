import requests
from bs4 import BeautifulSoup
import os
from tqdm import tqdm

# Function to download a book
def download_book(url, folder, pbar):
    response = requests.get(url)
    filename = os.path.join(folder, url.split('/')[-1])
    with open(filename, 'wb') as file:
        file.write(response.content)
    pbar.set_description(f"Downloaded: {filename}")
    pbar.update(1)

# Main function
def main():
    base_url = "https://www.gutenberg.org/"
    spanish_books_url = "https://www.gutenberg.org/browse/languages/es"

    # Make a GET request to the Spanish books page
    response = requests.get(spanish_books_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all book links
    book_links = soup.find_all('a', href=True)

    # Create a directory to store downloaded books
    if not os.path.exists('spanish_books'):
        os.makedirs('spanish_books')

    # Count the number of books to download
    total_books = sum(1 for link in book_links if link['href'].startswith('/ebooks/'))

    # Initialize tqdm progress bar
    with tqdm(total=total_books) as pbar:
        # Loop through book links and download
        for link in book_links:
            href = link['href']
            if href.startswith('/ebooks/'):
                download_book(base_url + href + '.txt.utf-8', 'spanish_books', pbar)

if __name__ == "__main__":
    main()
