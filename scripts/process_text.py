import re
import pandas as pd
from textblob import TextBlob

# ==============================================================================
# 1. READ RAW TEXT FILE LOCALLY
# ==============================================================================
BOOK_ID = "alice_in_wonderland"  # Options: 'alice', 'the_secret_garden', 'peter_pan', or 'jekyll_and_hyde'
file_path = f"data/{BOOK_ID}.txt"

print(f"Opening and parsing raw text file: {file_path}...")
with open(file_path, "r", encoding="utf-8-sig") as f:
    raw_text = f.read()

# FIXED: Initialize data array at the top level to prevent NameError
data = []

# ==============================================================================
# 2. DYNAMIC CHAPTER ISOLATION (No Hardcoding Required)
# ==============================================================================
regex_pattern = r'(?:^|\s)Chapter\s+[IVXLCDM]+\.?'
matches = list(re.finditer(regex_pattern, raw_text, re.IGNORECASE))

print(f"Total raw matches found: {len(matches)}")

if len(matches) > 1:
    # Calculate the 'spread' of the matches to find the index
    half = len(matches) // 2
    first_half_dist = matches[half-1].start() - matches[0].start()
    second_half_dist = matches[-1].start() - matches[half].start()

    # If the second half of matches covers significantly more text, it's an index
    if second_half_dist > (first_half_dist * 5):
        print("Dynamic detection: Table of Contents found. Discarding index...")
        matches = matches[half:]
    else:
        print("Dynamic detection: No Table of Contents detected.")

print(f"--> Successfully isolated {len(matches)} narrative chapters.")

# ==============================================================================
# 3. COMPUTE NLP METRICS ON ISOLATED SEGMENTS
# ==============================================================================
for i in range(len(matches)):
    start_pos = matches[i].start()
    
    if i + 1 < len(matches):
        end_pos = matches[i + 1].start()
    else:
        # Search for the Gutenberg footer to end the final chapter
        end_pos = raw_text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
        if end_pos == -1:
            end_pos = len(raw_text)
            
    chapter_content = raw_text[start_pos:end_pos].strip()
    
    # Process text metrics
    blob = TextBlob(chapter_content)
    clean_words = [w.lower() for w in blob.words if w.isalnum()]
    word_count = len(clean_words)
    
    # Skip processing if it's an empty fragment (safety check)
    if word_count == 0:
        continue

    punctuation_count = len(re.findall(r'[!?,.\-—;:]', chapter_content))
    header_line = chapter_content.split('\n')[0].strip()
    clean_title = re.sub(r'\s+', ' ', header_line)
    
    data.append({
        "chapter": i + 1,
        "title": clean_title,
        "word_count": word_count,
        "sentiment_polarity": blob.sentiment.polarity,
        "lexical_diversity": len(set(clean_words)) / word_count if word_count > 0 else 0,
        "punctuation_count": punctuation_count
    })

# ==============================================================================
# 4. EXPORT COMPILED METRICS MATRIX TO CSV
# ==============================================================================
if data:
    df = pd.DataFrame(data)
    output_path = f"data/{BOOK_ID}_petals.csv"
    df.to_csv(output_path, index=False)

    print("\n=== DATA RETRIEVAL GENERATION COMPLETE ===")
    print(f"Saved compiled dataset cleanly to: {output_path}")
    print(df[['chapter', 'word_count', 'sentiment_polarity']].head(10))
    print("===========================================")
else:
    print("\nERROR: No data was collected. Check your text landmarks.")