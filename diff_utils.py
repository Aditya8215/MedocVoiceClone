import re
from difflib import SequenceMatcher

def generate_diff_html(original_text, transcribed_text):
    original_words_with_separators = re.split(r'(\s+)', original_text)
    norm_original = re.sub(r'[^\w\s]', '', original_text).lower().split()
    norm_transcribed = re.sub(r'[^\w\s]', '', transcribed_text).lower().split()

    matcher = SequenceMatcher(None, norm_original, norm_transcribed)
    html_parts = []
    original_word_idx = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal' or tag == 'delete' or tag == 'replace':
            norm_word_count = i2 - i1
            words_processed = 0
            while words_processed < norm_word_count and original_word_idx < len(original_words_with_separators):
                word = original_words_with_separators[original_word_idx]
                is_actual_word = word.strip()

                if is_actual_word:
                    if tag in ('delete', 'replace'):
                        html_parts.append(f'<span style="background-color: #993333; color: white; padding: 1px; border-radius: 3px;">{word}</span>')
                    else:
                        html_parts.append(word)
                    words_processed += 1
                else:
                    html_parts.append(word)
                original_word_idx += 1
    
    if original_word_idx < len(original_words_with_separators):
        html_parts.extend(original_words_with_separators[original_word_idx:])

    return "".join(html_parts).replace('\n', '<br>')