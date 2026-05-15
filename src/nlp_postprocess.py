from importlib.resources import files
from symspellpy import SymSpell, Verbosity

def setup_symspell():
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    
    # The modernized, warning-free way to load the dictionary
    dictionary_path = str(files("symspellpy").joinpath("frequency_dictionary_en_82_765.txt"))
    
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    return sym_spell

def resolve_keystrokes(predicted_keys):
    """Parses a list of key predictions and actively applies 'backspace' deletions."""
    buffer = []
    for key in predicted_keys:
        if key == 'backspace':
            if buffer:
                buffer.pop()  # Delete the last character
        elif key == 'space':
            buffer.append(' ')
        else:
            buffer.append(key)
    return "".join(buffer)

def correct_text(sym_spell, raw_text):
    """Splits raw text into words and corrects them using SymSpell."""
    corrected_words = []
    for word in raw_text.split():
        # Lookup suggestions for maximum edit distance 2
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions:
            # Append the best suggestion
            corrected_words.append(suggestions[0].term)
        else:
            corrected_words.append(word)
    return " ".join(corrected_words)

if __name__ == "__main__":
    print("\n--- AcoustiGuard NLP Post-Processing Demo ---")
    
    # Simulated noisy output from your EfficientNet/ViT attacker model
    # It meant to type: "hello world" but made a typo and hit backspace, and misspelled world.
    noisy_predictions = [
        'h', 'e', 'l', 'p', 'backspace', 'l', 'o', 'space', 
        'w', 'o', 'r', 'k', 'd'
    ]
    
    print(f"1. Raw Neural Network Predictions:\n   {noisy_predictions}\n")
    
    # Step 1: Resolve mechanical backspaces
    raw_string = resolve_keystrokes(noisy_predictions)
    print(f"2. String Buffer (After applying backspaces):\n   '{raw_string}'\n")
    
    # Step 2: Apply SymSpell NLP Correction
    sym_spell = setup_symspell()
    final_text = correct_text(sym_spell, raw_string)
    
    print(f"3. Final AcoustiGuard Transcribed Text:\n   '{final_text}'\n")
    print("-------------------------------------------\n")