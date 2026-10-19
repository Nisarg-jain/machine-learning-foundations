import requests
from bs4 import BeautifulSoup
import pandas as pd

print("=" * 65)
print("1. INGESTING TABULAR DATA VIA REST API (JSON -> DATAFRAME)")
print("=" * 65)

# Example: Pulling live JSON placeholder data from a REST API endpoint
api_url = "https://jsonplaceholder.typicode.com/posts"
response = requests.get(api_url, timeout=10)

if response.status_code == 200:
    posts_data = response.json()
    df_api = pd.DataFrame(posts_data)
    print(f"Successfully retrieved {len(df_api)} records from API.")
    print(df_api[['userId', 'id', 'title']].head(3))
else:
    print(f"API request failed with status code: {response.status_code}")

print("\n" + "=" * 65)
print("2. WEB SCRAPING TABULAR DATA VIA BEAUTIFULSOUP (HTML DOM -> CSV)")
print("=" * 65)

# Example: Scraping a public quotes page with custom browser headers
scrape_url = "https://quotes.toscrape.com/"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

scrape_response = requests.get(scrape_url, headers=headers, timeout=10)
soup = BeautifulSoup(scrape_response.text, "html.parser")

extracted_quotes = []
quote_blocks = soup.find_all("div", class_="quote")

for block in quote_blocks:
    text = block.find("span", class_="text").get_text(strip=True)
    author = block.find("small", class_="author").get_text(strip=True)
    tags = [tag.get_text(strip=True) for tag in block.find_all("a", class_="tag")]
    
    extracted_quotes.append({
        "quote": text,
        "author": author,
        "tags": ", ".join(tags)
    })

df_scraped = pd.DataFrame(extracted_quotes)
df_scraped.to_csv("scraped_quotes.csv", index=False)

print(f"Successfully scraped {len(df_scraped)} quotes from HTML DOM.")
print(df_scraped.head(3))
print("\nExported extracted data to 'scraped_quotes.csv'!")