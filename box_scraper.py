import pandas as pd
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Import the Agent database manager for automated ChromaDB rebuilds
try:
    from main import setup_rag_index
    HAS_AGENT_LINK = True
except ImportError:
    HAS_AGENT_LINK = False

def format_name(full_name):
    """Converts full player names into standard initial format (e.g., 'S. Vezenkov')."""
    parts = full_name.split()
    if len(parts) >= 2:
        return f"{parts[0][0]}. {parts[-1]}"
    return full_name

def scrape_basketball_reference(url, output_name):
    # SMART CHECK: Skip execution immediately if the target CSV already exists on disk
    target_file = f"data/box_scores/{output_name}.csv"
    if os.path.exists(target_file):
        print(f" [SKIP]: File {output_name}.csv already exists. Skipping extraction.")
        return False # Returns False since no new data was fetched

    print(f" Opening browser for: {url}")
    
    try:
        name_parts = output_name.split("_")
        home_team_name = name_parts[0].capitalize() 
        away_team_name = name_parts[1].capitalize()
    except:
        home_team_name = "Home"
        away_team_name = "Away"

    chrome_options = Options()
    chrome_options.add_argument("--headless") 
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    
    try:
        driver.get(url)
        time.sleep(5) 

        html_content = driver.page_source
        tables = pd.read_html(html_content)
        
        all_players = []
        team_count = 0 

        for df in tables:
            # Validate if the parsed table contains standard box score statistical columns
            if 'MP' in df.columns and 'PTS' in df.columns:
                
                team_label = away_team_name if team_count == 0 else home_team_name
                
                df = df.dropna(subset=['Player'])
                df = df[df['Player'] != 'Team Totals']
                
                for _, row in df.iterrows():
                    try:
                        # Cast raw string metrics to integers and calculate Performance Index Rating (PIR)
                        pts = int(row['PTS'])
                        trb = int(row['TRB'])
                        ast = int(row['AST'])
                        stl = int(row['STL'])
                        blk = int(row['BLK'])
                        tov = int(row['TOV'])
                        pf  = int(row['PF'])
                        fga = int(row['FGA'])
                        fg  = int(row['FG'])
                        fta = int(row['FTA'])
                        ft  = int(row['FT'])

                        missed_fg = fga - fg
                        missed_ft = fta - ft
                        pir = (pts + trb + ast + stl + blk) - (missed_fg + missed_ft + tov + pf)

                        all_players.append({
                            "Match": f"{home_team_name} vs {away_team_name}", 
                            "Team": team_label, 
                            "Player": format_name(row['Player']),
                            "MIN": row['MP'],
                            "PTS": pts,
                            "2FG": f"{fg-int(row['3P'])}/{fga-int(row['3PA'])}",
                            "3FG": f"{row['3P']}/{row['3PA']}",
                            "FT": f"{ft}/{fta}",
                            "REB": trb,
                            "AST": ast,
                            "STL": stl,
                            "TO": tov,
                            "PIR": pir
                        })
                    except:
                        continue
                
                team_count += 1 

        if all_players:
            final_df = pd.DataFrame(all_players)
            os.makedirs("data/box_scores", exist_ok=True)
            final_df.to_csv(target_file, index=False)
            print(f"File {output_name}.csv created successfully.")
            return True # Returns True indicating new spreadsheet data was successfully stored
        else:
            print("No data found.")
            return False

    except Exception as e:
        print(f"Selenium error: {e}")
        return False
    finally:
        driver.quit()

if __name__ == "__main__":
    # Core target dataset collection mappings
    games_to_scrape = {
        "baskonia_olympiacos": "https://www.basketball-reference.com/international/boxscores/2025-09-30-vitoria.html",
        "real_olympiacos": "https://www.basketball-reference.com/international/boxscores/2025-10-02-real-madrid.html",
        "olympiacos_dubai": "https://www.basketball-reference.com/international/boxscores/2025-10-10-olympiakos.html",
        "olympiacos_efes": "https://www.basketball-reference.com/international/boxscores/2025-10-14-olympiakos.html",
        "maccabi_olympiacos": "https://www.basketball-reference.com/international/boxscores/2025-10-16-maccabi-tel-aviv.html",
        "bayern_olympiacos": "https://www.basketball-reference.com/international/boxscores/2025-10-24-bayern-muenchen.html",
        "olympiacos_monaco": "https://www.basketball-reference.com/international/boxscores/2025-10-29-olympiakos.html",
        "olympiacos_hapoel": "https://www.basketball-reference.com/international/boxscores/2025-10-31-olympiakos.html",
        "olympiacos_partizan": "https://www.basketball-reference.com/international/boxscores/2025-11-07-olympiakos.html",
        "olympiacos_zalgiris": "https://www.basketball-reference.com/international/boxscores/2025-11-12-olympiakos.html",
        "armani_olympiacos": "https://www.basketball-reference.com/international/boxscores/2025-11-14-milano.html",
        "olympiacos_paris": "https://www.basketball-reference.com/international/boxscores/2025-11-21-olympiakos.html",
        "zvezda_olympiacos": "https://www.basketball-reference.com/international/boxscores/2025-11-26-red-star.html",
        "barcelona_olympiacos": "https://www.basketball-reference.com/international/boxscores/2025-12-12-barcelona.html",
        "olympiacos_valencia": "https://www.basketball-reference.com/international/boxscores/2025-12-16-olympiakos.html",
        "olympiacos_asvel": "https://www.basketball-reference.com/international/boxscores/2025-12-19-olympiakos.html",
        "virtus_olympiacos": "https://www.basketball-reference.com/international/boxscores/2025-12-26-virtus-bologna.html",
        "panathinaikos_olympiacos": "https://www.basketball-reference.com/international/boxscores/2026-01-02-panathinaikos.html",
        "fenerbahce_olympiacos": "https://www.basketball-reference.com/international/boxscores/2026-01-06-ulker-fenerbahce.html",
        "olympiacos_bayern": "https://www.basketball-reference.com/international/boxscores/2026-01-09-olympiakos.html",
        "partizan_olympiacos": "https://www.basketball-reference.com/international/boxscores/2026-01-16-partizan.html",
        "olympiacos_maccabi": "https://www.basketball-reference.com/international/boxscores/2026-01-20-olympiakos.html",
        "efes_olympiacos": "https://www.basketball-reference.com/international/boxscores/2026-01-22-anadolu-efes.html",
        "olympiacos_barcelona": "https://www.basketball-reference.com/international/boxscores/2026-01-29-olympiakos.html",
        "dubai_olympiacos": "https://www.basketball-reference.com/international/boxscores/2026-02-03-dubai.html",
        "olympiacos_virtus": "https://www.basketball-reference.com/international/boxscores/2026-02-06-olympiakos.html",
        "olympiacos_zvezda": "https://www.basketball-reference.com/international/boxscores/2026-02-12-olympiakos.html",
        "zalgiris_olympiacos": "https://www.basketball-reference.com/international/boxscores/2026-02-25-zalgiris.html",
        "olympiacos_panathinaikos": "https://www.basketball-reference.com/international/boxscores/2026-03-06-olympiakos.html",
        "monaco_olympiacos": "https://www.basketball-reference.com/international/boxscores/2026-03-13-monaco.html",
        "olympiacos_fenerbahce": "https://www.basketball-reference.com/international/boxscores/2026-03-17-olympiakos.html",
        "olympiacos_baskonia": "https://www.basketball-reference.com/international/boxscores/2026-03-19-olympiakos.html",
        "valencia_olympiacos": "https://www.basketball-reference.com/international/boxscores/2026-03-24-valencia.html",
        "paris_olympiacos": "https://www.basketball-reference.com/international/boxscores/2026-03-26-paris-basket.html",
        "asvel_olympiacos": "https://www.basketball-reference.com/international/boxscores/2026-04-03-villeurbanne.html",
        "olympiacos_real": "https://www.basketball-reference.com/international/boxscores/2026-04-07-olympiakos.html",
        "hapoel_olympiacos": "https://www.basketball-reference.com/international/boxscores/2026-04-09-hapoel-tel-aviv.html",
        "olympiacos_armani": "https://www.basketball-reference.com/international/boxscores/2026-04-16-olympiakos.html"
    }

    print(f"Starting scraping check for {len(games_to_scrape)} matches.")
    
    new_data_added = False

    for file_name, url in games_to_scrape.items():
        was_scraped = scrape_basketball_reference(url, file_name)
        # Flip the flag to True if at least one new file is downloaded
        if was_scraped:
            new_data_added = True
            print("Waiting for IP protection...")
            time.sleep(12) 

    print(" Box score check complete.")

    # AUTO-REBUILD: Trigger an index refresh only if new matches were actually fetched
    if new_data_added and HAS_AGENT_LINK:
        print("\n New match data detected! Rebuilding Agent Vector Database...")
        setup_rag_index(rebuild=True)
        print(" Agent Database updated successfully with new box scores!")
    elif HAS_AGENT_LINK:
        print("\n Database up to date. No rebuild required.")