import re
import pandas as pd
from textblob import TextBlob

# ==============================================================================
# 1. READ RAW TEXT FILE
# ==============================================================================
BOOK_ID = "jekyll_and_hyde" 
file_path = f"data/{BOOK_ID}.txt"

with open(file_path, "r", encoding="utf-8-sig") as f:
    raw_text = f.read()

data = []

# ==============================================================================
# 2. MATCH TITLE-ONLY CHAPTER HEADINGS
# ==============================================================================
# This pattern looks for:
# ^[ \t]*       -> Start of a line with optional horizontal whitespace
# ([A-Z][A-Z\s’'.,]{5,}) -> 5+ Uppercase letters/punctuation (the title)
# [ \t]*$       -> End of that line
# It uses re.MULTILINE to check every line in the book.
title_pattern = r'^[ \t]*([A-Z][A-Z\s’\'.,]{5,})[ \t]*$'
matches = list(re.finditer(title_pattern, raw_text, re.MULTILINE))

print(f"Total title matches found: {len(matches)}")

# Jekyll & Hyde has 10 chapters. If we find 20+, the first 10 are the Table of Contents.
if len(matches) >= 20:
    print("Table of Contents detected. Slicing narrative sections...")
    matches = matches[10:] # Skip the first 10 matches (the Index)

print(f"--> Successfully isolated {len(matches)} narrative chapters.")

# ==============================================================================
# 3. COMPUTE NLP METRICS
# ==============================================================================
for i in range(len(matches)):
    start_pos = matches[i].start()
    
    if i + 1 < len(matches):
        end_pos = matches[i + 1].start()
    else:
        end_pos = raw_text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
        if end_pos == -1: end_pos = len(raw_text)
            
    chapter_content = raw_text[start_pos:end_pos].strip()
    
    blob = TextBlob(chapter_content)
    clean_words = [w.lower() for w in blob.words if w.isalnum()]
    word_count = len(clean_words)
    
    if word_count < 100: continue # Filter out any lingering metadata crumbs

    data.append({
        "chapter": i + 1,
        "title": matches[i].group(1).strip(),
        "word_count": word_count,
        "sentiment_polarity": blob.sentiment.polarity,
        "lexical_diversity": len(set(clean_words)) / word_count if word_count > 0 else 0,
        "punctuation_count": len(re.findall(r'[!?,.\-—;:]', chapter_content))
    })

# ==============================================================================
# 4. EXPORT TO CSV
# ==============================================================================
df = pd.DataFrame(data)
output_path = f"data/{BOOK_ID}_petals.csv"
df.to_csv(output_path, index=False)

print("\n=== DATA RETRIEVAL COMPLETE ===")
print(df[['chapter', 'word_count', 'sentiment_polarity']])