import json
from typing import List
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Tokenizer:
    def __init__(self, model_path: str):
        logging.info(f"Loading tokenizer from {model_path}")
        # Load the model file
        with open(model_path, "r", encoding="utf-8") as f:
            model = json.load(f)
        
        # Extract tokens and scores
        self.tokens = model["tokens"]
        self.scores = model["scores"]
        
        # Create lookup dictionaries
        self._token_to_id = {}
        self._id_to_token = {}
        
        # Handle the special Ġ (U+0120) character used for spaces in Llama tokenizer
        self._space_token = 'Ġ'  # This is the byte 0xC4 0xA0 in UTF-8 (U+0120)
        self._has_space_token = False
        
        for i, token in enumerate(self.tokens):
            if token:  # Skip empty tokens
                self._token_to_id[token] = i
                self._id_to_token[i] = token
                
                # Check if this is the special space token
                if token == self._space_token:
                    self._has_space_token = True
                    self._space_token_id = i
        
        # Set special token IDs
        self.bos_id = model["bos_id"]
        self.eos_id = model["eos_id"]
        
        # Extract special tokens
        self._special_tokens = set(model.get("special_tokens", []))
        
        logging.info(f"Tokenizer initialized with {len(self._token_to_id)} tokens")
        logging.info(f"BOS ID: {self.bos_id}, EOS ID: {self.eos_id}")
        logging.info(f"Special tokens: {self._special_tokens}")
        logging.info(f"Has space token: {self._has_space_token}")
    
    def str_lookup(self, token: str) -> int:
        """Look up the ID of a token."""
        # Check for the special case of a space character
        if token == ' ' and self._has_space_token:
            return self._space_token_id
            
        # Check if the token exists directly
        return self._token_to_id.get(token, -1)
    
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """Encode text using tokenization."""
        # Replace spaces with the special space token
        if self._has_space_token:
            text = text.replace(' ', self._space_token)
            
        tokens = []
        i = 0
        
        # Process the text character by character
        while i < len(text):
            # Find the longest matching token
            best_len = 0
            best_token_id = -1
            
            # Try to match tokens of different lengths
            for end in range(i + 1, min(i + 20, len(text) + 1)):  # Limit max token length for efficiency
                substr = text[i:end]
                token_id = self._token_to_id.get(substr, -1)
                
                if token_id >= 0 and end - i > best_len:
                    best_len = end - i
                    best_token_id = token_id
            
            # If we found a match, add it
            if best_token_id >= 0:
                tokens.append(best_token_id)
                i += best_len
            else:
                # Handle character not in the vocabulary
                logging.warning(f"Character not in vocabulary: {repr(text[i])}")
                # Skip this character
                i += 1
        
        # Add special tokens
        if add_bos:
            tokens.insert(0, self.bos_id)
        if add_eos:
            tokens.append(self.eos_id)
        
        return tokens
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to a string."""
        result = ""
        for idx in ids:
            if idx in self._id_to_token:
                token = self._id_to_token[idx]
                # Replace the special space token with an actual space
                if token == self._space_token:
                    result += ' '
                else:
                    # Replace any special space tokens at the beginning of tokens
                    if token.startswith(self._space_token):
                        result += ' ' + token[len(self._space_token):]
                    else:
                        result += token
        return result