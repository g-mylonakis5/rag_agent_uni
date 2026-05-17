import pandas as pd
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

def format_name(full_name):
    """Name conversions """
    parts = full_name.split()
    if len(parts) >= 2:
        return f"{parts[0][0]}. {parts[-1]}"
    return full_name

def scrape_basketball_reference(url, output_name):
    print(f" Opening browser for: {url}")
    
    #
    # Example = real_olympiacos -> Real = 'Home', Olympiacos = 'Away'
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
            # Check if table contains stats 
            if 'MP' in df.columns and 'PTS' in df.columns:
                
                
                team_label = away_team_name if team_count == 0 else home_team_name
                
                df = df.dropna(subset=['Player'])
                df = df[df['Player'] != 'Team Totals']
                
                for _, row in df.iterrows():
                    try:
                        # Conversion to numbers and calculating Performance Index rating 
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
                            "Match": f"{home_team_name} vs {away_team_name}", # Νέο πεδίο για extra context[cite: 2]
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
            final_df.to_csv(f"data/box_scores/{output_name}.csv", index=False)
            print(f"File {output_name}.csv created.")
        else:
            print("No data found.")

    except Exception as e:
        print(f"Selenium error: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
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
        "milano_olympiacos": "https://www.basketball-reference.com/international/boxscores/2025-11-14-milano.html",
        "olympiacos_paris": "https://www.basketball-reference.com/international/boxscores/2025-11-21-olympiakos.html",
        "zvezda_olympiacos": "https://www.basketball-reference.com/international/boxscores/2025-11-26-red-star.html",
        "barcelona_olympiacos": "https://www.basketball-reference.com/international/boxscores/2025-12-12-barcelona.html",
        "panathinaikos_olympiacos": "https://www.basketball-reference.com/international/boxscores/2026-01-02-panathinaikos.html",
    }

    print(f"Starting scraping for  {len(games_to_scrape)} matches.")

    for file_name, url in games_to_scrape.items():
        
        scrape_basketball_reference(url, file_name)
        print("Waiting for IP protection")
        time.sleep(12) 

    print("✅ Box score scraping complete.")