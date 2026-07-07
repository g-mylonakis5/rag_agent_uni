import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from newspaper import Article

def scrape_eurohoops_summary(url, output_name):
    """
    Opens the link, removes extra noise and scrapes the summary text 
    """
    print(f" Beginning text collection from: {url}")
    
    chrome_options = Options()
    chrome_options.add_argument("--headless") 
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    
    try:
        driver.get(url)
        time.sleep(5) 

        # Using NewsPaper3k for extracting text 
        article = Article(url)
        article.set_html(driver.page_source)
        article.parse()
        
        raw_text = article.text 

        # Noise Removal
        lines = raw_text.split('\n')
        filtered_lines = []
        
        for line in lines:
            clean_line = line.strip()
            # 1. Removing blank lines 
            if not clean_line:
                continue
            # 2. Removing emails 
            if "info@eurohoops.net" in clean_line:
                continue
            # 3. Removing writers
            if clean_line.startswith("By "):
                continue
            # 4. Removing timestamps 
            if "2025-" in clean_line or "2026-" in clean_line:
                continue
            
            filtered_lines.append(clean_line)
        
        # Joining clean text into one 
        clean_text = "\n".join(filtered_lines)

        if clean_text:
            os.makedirs("data/summaries", exist_ok=True)
            file_path = f"data/summaries/{output_name}.txt"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(clean_text)
            
            print(f"Summary saved in: {file_path}")
        else:
            print(f"No text found in {url}")

    except Exception as e:
        print(f" Error scraping {output_name}: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    
    summaries_to_collect = {
        "baskonia_olympiacos": "https://www.eurohoops.net/en/euroleague/1874634/olympiacos-escapes-from-baskonia-with-almost-perfect-31-32-free-throws/",
        "real_olympiacos": "https://www.eurohoops.net/en/euroleague/1875475/real-madrid-completes-a-tough-comeback-against-olympiacos/",
        "olympiacos_dubai": "https://www.eurohoops.net/en/euroleague/1878766/vezenkov-and-milutinov-power-olympiacos-past-dubai/",
        "olympiacos_efes": "https://www.eurohoops.net/en/euroleague/1880277/olympiacos-versus-efes-round-4-euroleague/",
        "maccabi_olympiacos": "https://www.eurohoops.net/en/euroleague/1881672/olympiacos-completes-an-epic-comeback-over-maccabi/",
        "bayern_olympiacos": "https://www.eurohoops.net/en/euroleague/1885089/olympiacos-easily-defeated-bayern-in-munich/",
        "olympiacos_hapoel": "https://www.eurohoops.net/en/euroleague/1888264/olympiacos-prevailed-after-a-dogfight-over-hapoel/",
        "olympiacos_partizan": "https://www.eurohoops.net/en/euroleague/1891293/olympiacos-defeated-partizan-after-a-big-comeback/",
        "olympiacos_zalgiris": "https://www.eurohoops.net/en/euroleague/1893613/dorsey-scores-30-as-olympiacos-beats-zalgiris-and-loses-keenan-evans-again/",
        "milano_olympiacos": "https://www.eurohoops.net/en/euroleague/1894676/olimpia-milano-claims-narrow-victory-over-olympiacos/",
        "olympiacos_paris": "https://www.eurohoops.net/en/euroleague/1897470/robinson-exploded-for-35-points-but-olympiacos-stopped-paris/",
        "zvezda_olympiacos": "https://www.eurohoops.net/en/euroleague/1899724/zvezda-stopped-olympiacos-in-the-red-and-white-duel/",
        "barcelona_olympiacos": "https://www.eurohoops.net/en/euroleague/1906303/barca-crushes-olympiacos-for-an-statement-win/",
        "olympiacos_valencia": "https://www.eurohoops.net/en/euroleague/1907779/olympiacos-versus-valencia-euroleague-round-16-recap/",
        "olympiacos_asvel": "https://www.eurohoops.net/en/euroleague/1909464/olympiacos-had-it-easy-against-asvel/",
        "panathinaikos_olympiacos": "https://www.eurohoops.net/en/euroleague/1914195/olympiacos-edges-panathinaikos-in-thrilling-greek-derby/",
        "fenerbahce_olympiacos": "https://www.eurohoops.net/en/euroleague/1915716/fenerbahce-versus-olympiacos-euroleague-round-20-recap/",
        "partizan_olympiacos": "https://www.eurohoops.net/en/euroleague/1919304/olympiacos-blows-out-partizan-in-belgrade/",
        "efes_olympiacos": "https://www.eurohoops.net/en/euroleague/1922832/olympiacos-hands-anadolu-efes-eighth-straight-defeat/",
        "olympiacos_barcelona": "https://www.eurohoops.net/en/euroleague/1925706/vezenkov-and-jones-lead-olympiacos-past-barcelona/",
        "olympiacos_virtus": "https://www.eurohoops.net/en/euroleague/1929171/olympiacos-triumphs-at-home-against-virtus-bologna/",
        "olympiacos_zvezda": "https://www.eurohoops.net/en/euroleague/1931322/olympiacos-broke-zvezda-after-drama-in-the-red-white-clash/",
        "zalgiris_olympiacos": "https://www.eurohoops.net/en/euroleague/1936638/zalgiris-wins-double-overtime-drama-against-olympiacos/",
        "olympiacos_panathinaikos": "https://www.eurohoops.net/en/euroleague/1940214/olympiacos-beat-panathinaikos-in-the-last-greek-derby-of-the-regular-season/",
        "paris_olympiacos": "https://www.eurohoops.net/en/euroleague/1941917/olympiacos-beats-paris-by-scoring-104-points-on-the-road/",
        "monaco_olympiacos": "https://www.eurohoops.net/en/euroleague/1942922/strazel-hits-game-winner-as-monaco-beats-olympiacos/",
        "olympiacos_fenerbahce": "https://www.eurohoops.net/en/euroleague/1944949/fournier-with-a-euroleague-career-high-led-olympiacos-over-shorthanded-fenerbahce/",
        "olympiacos_baskonia": "https://www.eurohoops.net/en/euroleague/1945909/vezenkov-and-milutinov-lead-olympiacos-past-baskonia/",
        "valencia_olympiacos": "https://www.eurohoops.net/en/euroleague/1947621/valencia-versus-olympiacos-recap-euroleague/",
        "asvel_olympiacos": "https://www.eurohoops.net/en/euroleague/1951964/olympiacos-takes-down-asvel-to-seize-top-spot/",
        "olympiacos_real": "https://www.eurohoops.net/en/euroleague/1953489/dorsey-scorched-real-madrid-with-37-points-giving-the-top-to-olympiacos/",
        "hapoel_olympiacos": "https://www.eurohoops.net/en/euroleague/1954342/olympiacos-beats-hapoel-in-sofia-secures-home-court-advantage/",
        "olympiacos_armani": "https://www.eurohoops.net/en/euroleague/1956712/olympiacos-crushed-milano-and-secured-the-top-spot-in-euroleague/"
    }

    print(f" Starting collection of {len(summaries_to_collect)} summaries.")

    for file_name, url in summaries_to_collect.items():
        
        if os.path.exists(f"data/summaries/{file_name}.txt"):
            os.remove(f"data/summaries/{file_name}.txt")
            
        scrape_eurohoops_summary(url, file_name)
        print("Waiting 10 seconds for IP protection...")
        time.sleep(10)

    print("Scraping completed.")