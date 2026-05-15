"""
Natural Language Processing interface implementing SymSpell 
for post-classification transcription correction.
"""
from importlib.resources import files
from symspellpy import SymSpell

class KeystrokeCorrector:
    def __init__(self):
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = str(files("symspellpy").joinpath("frequency_dictionary_en_82_765.txt"))
        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        
    def correct_string(self, raw_text):
        """Attempts to correct typographic deviations utilizing edit distance parameters."""
        suggestions = self.sym_spell.lookup_compound(raw_text, max_edit_distance=2)
        return suggestions[0].term if suggestions else raw_text