import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import os

def scrape_skytrax_reviews(airline_name, base_url, num_pages=50):  # increased to 50 to gather more data
    """
    Scrapes airline reviews from Skytrax.

    Args:
        airline_name (str): The name of the airline (used for file naming).
        base_url (str): The base URL for the airline's Skytrax reviews (e.g., 'https://www.airlinequality.com/airline-reviews/british-airways').
        num_pages (int): The number of pages of reviews to scrape.

    Returns:
        pandas.DataFrame: A DataFrame containing the scraped reviews.
    """
    all_reviews = []

    for page in range(1, num_pages + 1):
        url = f"{base_url}/page/{page}?sortby=post_date%3ADesc&pagesize=10"  # Modified URL for Skytrax sorting
        print(f"Scraping page: {page}")
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            soup = BeautifulSoup(response.content, "html.parser")

            reviews = soup.find_all("article", itemprop="review") # The review entries are present in an article
            for review in reviews:
                try:
                    # Extract data from each review
                    title = review.find("h2").text.strip() if review.find("h2") else None
                    content = review.find("div", class_="text_content").text.strip() if review.find("div", class_="text_content") else None

                    # Extract the rating from the star rating system, if available
                    rating_element = review.find("div", class_="rating-10")
                    rating = int(rating_element.find("span").text) if rating_element and rating_element.find("span") else None

                    date = review.find("time", itemprop="datePublished").text.strip() if review.find("time", itemprop="datePublished") else None
                    author = review.find("span", itemprop="name").text.strip() if review.find("span", itemprop="name") else None

                    # Extracting recommended status
                    recommended_element = review.find("span", class_="recommend-value")
                    recommended = recommended_element.text.strip() == "yes" if recommended_element else None

                    all_reviews.append({
                        "title": title,
                        "content": content,
                        "rating": rating,
                        "date": date,
                        "author": author,
                        "recommended": recommended
                    })
                except Exception as e:
                    print(f"Error extracting data from a review: {e}")

        except requests.exceptions.RequestException as e:
            print(f"Request error on page {page}: {e}")
            break  # Stop scraping if there's a persistent request error
        except Exception as e:
            print(f"Error processing page {page}: {e}")

    df = pd.DataFrame(all_reviews)
    return df


# Example Usage
airline_name = "british-airways"  # Replace with the airline you want to scrape
base_url = f"https://www.airlinequality.com/airline-reviews/{airline_name}"
num_pages_to_scrape = 50 # Adjust the number of pages
df = scrape_skytrax_reviews(airline_name, base_url, num_pages_to_scrape)

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Save the data to a CSV file
if not df.empty:
    df.to_csv(f"data/{airline_name}_reviews.csv", index=False)  # Save to the 'data' folder
    print(f"Scraped {len(df)} reviews and saved to data/{airline_name}_reviews.csv")
else:
    print("No reviews were scraped.")