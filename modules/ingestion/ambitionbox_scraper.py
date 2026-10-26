import time
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests

print('=' * 65)
print('LECTURE 18: MULTI-PAGE WEB SCRAPING PIPELINE')
print('=' * 65)


def parse_company_cards(soup):
  """Extracts individual attributes from each company card container."""
  cards = soup.find_all('div', class_='company-content-wrapper')
  records = []

  for card in cards:
    # 1. Company Name
    name_tag = card.find('h2', class_='company-name')
    name = name_tag.get_text(strip=True) if name_tag else np.nan

    # 2. Rating
    rating_tag = card.find('p', class_='rating')
    rating = rating_tag.get_text(strip=True) if rating_tag else np.nan

    # 3. Review Count
    review_tag = card.find('a', class_='review-count')
    reviews = review_tag.get_text(strip=True) if review_tag else np.nan

    # 4. Multi-item metadata entities [Type, HQ, Age, Employees]
    info_tags = card.find_all('p', class_='infoEntity')

    ctype = (
        info_tags[0].get_text(strip=True) if len(info_tags) > 0 else np.nan
    )
    hq = info_tags[1].get_text(strip=True) if len(info_tags) > 1 else np.nan
    age = info_tags[2].get_text(strip=True) if len(info_tags) > 2 else np.nan
    employees = (
        info_tags[3].get_text(strip=True) if len(info_tags) > 3 else np.nan
    )

    records.append({
        'company_name': name,
        'rating': rating,
        'reviews': reviews,
        'company_type': ctype,
        'headquarters': hq,
        'company_age': age,
        'employees': employees,
    })

  return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Pipeline Demonstration using Lecture 18 DOM Structure
# ---------------------------------------------------------------------------
print('\n[1] Simulating multi-page DOM scraping across 3 paginated pages...')

simulated_pages = [
    """
    <div class="company-content-wrapper">
        <h2 class="company-name">Tata Consultancy Services</h2>
        <p class="rating">3.9</p>
        <a class="review-count">(50.2k Reviews)</a>
        <p class="infoEntity">Public</p>
        <p class="infoEntity">Mumbai, Maharashtra</p>
        <p class="infoEntity">54 years old</p>
        <p class="infoEntity">1 Lakh+ Employees</p>
    </div>
    <div class="company-content-wrapper">
        <h2 class="company-name">Accenture</h2>
        <p class="rating">4.1</p>
        <a class="review-count">(32.4k Reviews)</a>
        <p class="infoEntity">Public</p>
        <p class="infoEntity">Dublin, Ireland</p>
        <p class="infoEntity">33 years old</p>
        <p class="infoEntity">1 Lakh+ Employees</p>
    </div>
    """,
    """
    <div class="company-content-wrapper">
        <h2 class="company-name">Infosys</h2>
        <p class="rating">3.8</p>
        <a class="review-count">(28.7k Reviews)</a>
        <p class="infoEntity">Public</p>
        <p class="infoEntity">Bengaluru, Karnataka</p>
        <p class="infoEntity">41 years old</p>
        <p class="infoEntity">1 Lakh+ Employees</p>
    </div>
    <div class="company-content-wrapper">
        <h2 class="company-name">Wipro</h2>
        <p class="rating">3.9</p>
        <a class="review-count">(22.1k Reviews)</a>
        <p class="infoEntity">Public</p>
        <p class="infoEntity">Bengaluru, Karnataka</p>
        <p class="infoEntity">77 years old</p>
        <p class="infoEntity">1 Lakh+ Employees</p>
    </div>
    """,
    """
    <div class="company-content-wrapper">
        <h2 class="company-name">HCL Technologies</h2>
        <p class="rating">3.7</p>
        <a class="review-count">(19.5k Reviews)</a>
        <p class="infoEntity">Public</p>
        <p class="infoEntity">Noida, Uttar Pradesh</p>
        <p class="infoEntity">46 years old</p>
        <p class="infoEntity">1 Lakh+ Employees</p>
    </div>
    <div class="company-content-wrapper">
        <h2 class="company-name">Cognizant</h2>
        <p class="rating">3.9</p>
        <a class="review-count">(29.1k Reviews)</a>
        <p class="infoEntity">Private</p>
        <p class="infoEntity">Teaneck, New Jersey</p>
        <p class="infoEntity">28 years old</p>
        <p class="infoEntity">1 Lakh+ Employees</p>
    </div>
    """,
]

page_dataframes = []

for page_idx, page_html in enumerate(simulated_pages, start=1):
  print(f' -> Parsing Page {page_idx}...')
  soup_page = BeautifulSoup(page_html, 'html.parser')
  df_page = parse_company_cards(soup_page)
  page_dataframes.append(df_page)

# Merge all pages using modern Pandas standard (pd.concat)
final_scraped_df = pd.concat(page_dataframes, ignore_index=True)

# Export parsed data
output_file = 'scraped_companies.csv'
final_scraped_df.to_csv(output_file, index=False)

print('\n' + '=' * 65)
print('EXTRACTED DATA PREVIEW')
print('=' * 65)
print(final_scraped_df)
print(f"\nSuccessfully created '{output_file}' with shape: {final_scraped_df.shape}")