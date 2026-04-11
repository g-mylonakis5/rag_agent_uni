import csv
import re
import os
from glob import glob

def convert_all_boxes_to_csv(directory):
    files = glob(os.path.join(directory, "*_box.txt"))
    headers = ['Name', 'Team', 'PTS', 'REB', 'AST', 'STL', 'TO', 'PIR', '2FG', '3FG', 'FT', 'MIN']
    
    # Νέο, πιο ευέλικτο pattern που επιτρέπει κείμενο/παρενθέσεις ενδιάμεσα
    pattern = (
        r"PLAYER:\s*(.*?)\s*\(TEAM:\s*(.*?)\)\s*-\s*STATS:\s*"
        r"PTS:\s*(\d+).*?REB:\s*(\d+).*?AST:\s*(\d+).*?"
        r"STL:\s*(\d+).*?TO:\s*(\d+).*?PIR:\s*(-?\d+).*?"
        r"2FG:\s*(\d+/\d+).*?3FG:\s*(\d+/\d+).*?FT:\s*(\d+/\d+).*?MIN:\s*([\d:]+)"
    )

    for txt_path in files:
        csv_path = txt_path.replace('.txt', '.csv')
        rows = []
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Καθαρίζουμε τη γραμμή από περιττά κενά
                line = line.strip()
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    rows.append(list(match.groups()))
                elif "PLAYER:" in line:
                    print(f"⚠️ Skipped potential player line (Format error): {line[:50]}...")

        if rows:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)
            print(f"✅ Μετατράπηκε με επιτυχία: {os.path.basename(txt_path)} ({len(rows)} παίκτες)")

convert_all_boxes_to_csv('basketball_articles')