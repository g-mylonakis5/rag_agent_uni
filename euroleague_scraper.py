# Αρχείο: euroleague_scraper.py
# Περιγραφή: Αυτόνομο εργαλείο web scraping για Euroleague (Manual & Batch mode)

import os
import time
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Φάκελος αποθήκευσης
ARTICLES_DIR = './basketball_articles'

class EuroleagueScraper:
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless") # Τρέχει στο παρασκήνιο χωρίς να ανοίγει παράθυρο
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--log-level=3")
        # Fake User-Agent για να μην καταλάβει το site ότι είμαστε bot
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        self.service = Service(ChromeDriverManager().install())
        self.options = chrome_options

        if not os.path.exists(ARTICLES_DIR):
            os.makedirs(ARTICLES_DIR)

    def scrape_and_save(self, url: str):
        print(f"\n[Scraper] Starting connection to: {url}")
        driver = webdriver.Chrome(service=self.service, options=self.options)
        
        try:
            driver.get(url)
            
            # Αναμονή και Scroll για να φορτώσουν τα στατιστικά (lazy loading)
            wait = WebDriverWait(driver, 20)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            driver.execute_script("window.scrollTo(0, 500);")
            time.sleep(4)
            
            try:
                content = driver.find_element(By.TAG_NAME, "main").text
            except:
                content = driver.find_element(By.TAG_NAME, "body").text
            
            # Καθαρισμός από JavaScript, cookies και διαφημίσεις
            lines = content.split("\n")
            clean_lines = []
            noise_words = ['javascript', 'cookie', 'subscribe', 'interest', 'goal3', 'about:', 'data,']
            
            for line in lines:
                line = line.strip()
                if not any(noise.lower() in line.lower() for noise in noise_words):
                    if len(line) > 3:
                        clean_lines.append(line)
            
            final_text = "\n".join(clean_lines)
            
            # Δημιουργία ονόματος αρχείου από το URL
            match = re.search(r'/([^/]+)/E\d+/', url)
            if match:
                game_name = match.group(1).replace("-", "_")
            else:
                game_name = f"euroleague_game_{int(time.time())}"
                
            filename = f"{game_name}_box.txt"
            filepath = os.path.join(ARTICLES_DIR, filename)
            
            # Αποθήκευση στο .txt αρχείο
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(final_text)
                
            print(f"[Scraper] SUCCESS! Data saved to: {filepath}")
            print(f"[Scraper] Total characters extracted: {len(final_text)}")
            
        except Exception as e:
            print(f"[Scraper] ERROR: {e}")
        finally:
            driver.quit()

if __name__ == "__main__":
    print("="*40)
    print("🏀 EUROLEAGUE DATA SCRAPER")
    print("="*40)
    scraper = EuroleagueScraper()
    
    print("Options:")
    print("1. Manual URL entry")
    print("2. Batch scrape from URLs file (urls.txt)")
    choice = input("\nSelect (1 or 2): ").strip()
    
    if choice == '2':
        if not os.path.exists("urls.txt"):
            with open("urls.txt", "w") as f:
                f.write("# Paste Euroleague URLs here, one per line\n")
            print("Created 'urls.txt'. Please open it, paste your links inside, save it, and run the script again.")
        else:
            with open("urls.txt", "r") as f:
                urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            
            print(f"Found {len(urls)} URLs to scrape.")
            for i, url in enumerate(urls):
                print(f"\n--- Scraping {i+1}/{len(urls)} ---")
                scraper.scrape_and_save(url)
                time.sleep(3) # Μικρή παύση για να μην μπλοκαριστούμε από το server
                
            print("\nBatch scraping completed!")
    else:
        while True:
            url = input("\nPaste Euroleague URL (or 'exit' to quit): ").strip()
            if url.lower() in ['exit', 'quit']:
                break
            if url.startswith("http"):
                scraper.scrape_and_save(url)
            else:
                print("Please enter a valid URL.")