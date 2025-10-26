# agent to act as imagination for visual(eye rods and cones activation) and sound(tongue movement)

import os
import json
import string
import base64

# Base62 alphabet
BASE62_ALPHABET = string.ascii_letters + string.digits

def base64_to_base62_sound(b64_str):
    """Convert Base64 string to Base62 string."""
    num = int.from_bytes(base64.b64decode(b64_str), byteorder='big')
    base62_str = ""
    while num > 0:
        num, rem = divmod(num, 62)
        base62_str = BASE62_ALPHABET[rem] + base62_str
    return base62_str

def base62_to_base64_sound(base62_str):
    """Convert Base62 string back to Base64 string."""
    num = 0
    for char in base62_str:
        num = num * 62 + BASE62_ALPHABET.index(char)
    # Convert integer back to bytes, then Base64
    byte_length = (num.bit_length() + 7) // 8
    return base64.b64encode(num.to_bytes(byte_length, byteorder='big')).decode()

def compare_values(val1, val2):
    if isinstance(val1, tuple) and isinstance(val2, tuple):
        if len(val1) != len(val2):
            return False
        return all(abs(a - b) < 0.001 for a, b in zip(val1, val2))
    return val1 == val2

def dict_match(dict1, dict2, tolerance=0.05):

    # Direct comparison for non-dictionary objects
    if not isinstance(dict1, dict) and not isinstance(dict2, dict):
        # For numeric values, apply tolerance
        if isinstance(dict1, (int, float)) and isinstance(dict2, (int, float)):
            # For zero or very small values, use absolute difference
            if abs(dict1) < 1e-10 or abs(dict2) < 1e-10:
                print(abs(dict1 - dict2))
                return abs(dict1 - dict2) < tolerance
            # For larger values, use relative difference
            else:
                relative_diff = abs(dict1 - dict2) / max(abs(dict1), abs(dict2))
                return relative_diff <= tolerance
        # For encoded strings, implement string similarity comparison
        elif isinstance(dict1, str) and isinstance(dict2, str):
            # If strings are identical, quick return
            if dict1 == dict2:
                return True

            # For encoded strings, implement more flexible matching
            # If strings have similar length (within 5%)
            len_diff_ratio = abs(len(dict1) - len(dict2)) / max(len(dict1), len(dict2))
            if len_diff_ratio > 0.05:  # If length differs by more than 5%, not a match
                return False

            # Compare character by character with some tolerance
            # Count number of matching characters
            shorter_len = min(len(dict1), len(dict2))
            matching_chars = sum(1 for i in range(shorter_len) if dict1[i] == dict2[i])
            match_ratio = matching_chars / shorter_len
            # Consider it a match if at least 95% of characters match
            return match_ratio >= 0.50
        else:
            # For other types, require exact match
            return dict1 == dict2

   # Ensure both are dictionaries for further checks
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        return False

    # Check if keys match
    if set(dict1.keys()) != set(dict2.keys()):
        return False

    # Check values using compare_values
    for key in dict1:
        if key not in dict2 or not compare_values(dict1[key], dict2[key]):
            return False
    return True

class ImagineAgent:
    def __init__(self, max_size=3, storage_dir=r"D:\artist\brainX\CRX\Properties\Imagination"):
        self.max_size = max_size
        self.memory = []  # in-memory cache
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self._load_from_storage()

    def _load_from_storage(self):
        """Load stored JSON files into memory on start."""
        files = sorted(
            [f for f in os.listdir(self.storage_dir) if f.endswith(".json")],
            key=lambda x: os.path.getmtime(os.path.join(self.storage_dir, x))
        )
        for file in files[-self.max_size:]:  # Only keep the latest files
            path = os.path.join(self.storage_dir, file)
            with open(path, "r") as f:
                try:
                    self.memory.append(json.load(f))
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode {file}")

    def _save_to_storage(self, output):
        """Save the given output as a JSON file."""
        filename = f"node_{len(os.listdir(self.storage_dir)) + 1}.json"
        path = os.path.join(self.storage_dir, filename)
        with open(path, "w") as f:
            json.dump(output, f, indent=2)

    def store(self, output):
        """Store unchanged output in memory and as a JSON file."""
        self.memory.append(output)
        self._save_to_storage(output)
        if len(self.memory) > self.max_size:
            self.memory.pop(0)  # remove oldest
            # Also remove the oldest file from disk
            files = sorted(
                [f for f in os.listdir(self.storage_dir) if f.endswith(".json")],
                key=lambda x: os.path.getmtime(os.path.join(self.storage_dir, x))
            )
            if files:
                os.remove(os.path.join(self.storage_dir, files[0]))

    def confirm(self, output, branch=False):
        """
        Check if output is in memory or urgent node (imagine/node.json).
        If found, remove it (from memory + disk).
        Returns:
            If branch=True:
                - ((next_spot_id_key_1, next_spot_id_key_2), True)  # if 2 outputs available
                - (next_spot_id_key, True)  # if only 1 output available
            If branch=False:
                - (next_spot_id_key, True)  # single match
            If not found:
                - (None, False)
        """
        # --- Step 0: Urgent node check in imagine/ ---
        urgent_node_path = r"D:\artist\brainX\CRX\Properties\Imagination\node.json"
        if os.path.exists(urgent_node_path):
            try:
                with open(urgent_node_path, "r") as f:
                    urgent_node = json.load(f)

                # Remove urgent node immediately (one-time use)
                os.remove(urgent_node_path)
                print("🚨 Urgent node retrieved from imagine folder")

                # Return urgent node regardless of normal output match
                return (urgent_node, True)

            except Exception as e:
                print(f"⚠️ Error reading urgent node: {e}")
                
        match_found = False
        # --- Step 1: Normal confirm from memory + storage_dir ---
        for i, stored in enumerate(self.memory):
            if dict_match(stored, output):  # Found match
                removed_output = self.memory.pop(i)  # remove from memory

                # Also remove from disk
                files = [f for f in os.listdir(self.storage_dir) if f.endswith(".json")]
                for file in files:
                    path = os.path.join(self.storage_dir, file)
                    try:
                        with open(path, "r") as f:
                            if dict_match(json.load(f), output):
                                match_found = True  # mark for deletion

                        if match_found:  # only delete after file is closed
                            os.remove(path)
                            break
                    except json.JSONDecodeError:
                        continue

                # Handle branch logic
                if branch:
                    if i < len(self.memory):  # Next element available
                        next_output = self.memory.pop(i)  # get next consecutive
                        return ((removed_output, next_output), True)
                    else:  # Only one output
                        return (removed_output, True)
                else:
                    return (removed_output, True)

        return (None, False)  # Not found

