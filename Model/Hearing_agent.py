# audio/text to phonemes

import numpy as np
import eng_to_ipa as ipa

import os
import json
import zlib
import base64
import string
import time

# Map phonemes to IDs
PHONEME2ID = {}
ID2PHONEME = {}
next_id = 0

def safe_remove(file_path, retries=5, delay=0.2):
    """Try removing a file with retries (for Windows file lock issues)."""
    for attempt in range(retries):
        try:
            os.remove(file_path)
            print(f"[safe_remove] Removed {file_path}")
            return True
        except PermissionError as e:
            # File is locked by another process
            print(f"[safe_remove] Attempt {attempt+1}/{retries} failed: {e}")
            time.sleep(delay)
        except Exception as e:
            print(f"[safe_remove] Unexpected error while removing {file_path}: {e}")
            return False
    print(f"[safe_remove] Giving up on {file_path}")
    return False

def load_hearing_input(hearing_dir=r"D:/artist/brainX/CRX/Properties/Ear_input"): 
    """
    Load all stored hearing input JSON files in order.
    Returns a list of strings: ["walk", "run", ...].
    Removes successfully processed files.
    """
    hearing_texts = [] 

    if not os.path.exists(hearing_dir): 
        print(f"[load_hearing_input] Directory not found: {hearing_dir}")
        return hearing_texts

    for file_name in sorted(os.listdir(hearing_dir)):
        if file_name.endswith(".json") and file_name.startswith("hearing_"):
            file_path = os.path.join(hearing_dir, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "hearing_text" in data:
                    hearing_texts.append(data["hearing_text"])
                    safe_remove(file_path)  # delete with retries
                else:
                    print(f"[load_hearing_input] Unexpected format in {file_path}: {data}")
            except Exception as e:
                print(f"[load_hearing_input] Error reading {file_path}: {e}")

    return hearing_texts


def continuous_hearing_monitor(hearing_dir=r"D:/artist/brainX/CRX/Properties/Ear_input"):
    """
    Continuously monitor hearing input every 5 seconds until user interrupts.
    Processes each new input immediately when detected.
    Returns the latest hearing texts when interrupted.
    """
    print("[continuous_hearing_monitor] Starting monitoring... Press Ctrl+C to stop.")
    
    latest_hearing_texts = []
    try:
        while True:
            # Load hearing input using the original logic
            current_hearing_texts = load_hearing_input(hearing_dir)
            
            # Check if there are new inputs
            if current_hearing_texts:
                latest_hearing_texts = current_hearing_texts
                print(f"[continuous_hearing_monitor] Updated hearing texts: {latest_hearing_texts}")
                
                # Process the new input immediately
                if latest_hearing_texts:
                    print(f"[continuous_hearing_monitor] Processing new input...")
                    process_hearing_input(latest_hearing_texts)
                
            else:
                print(f"[continuous_hearing_monitor] Checking... (found {len(latest_hearing_texts)} items)")
            
            # Wait for 5 seconds
            time.sleep(5.0)
                
    except KeyboardInterrupt:
        print("\n[continuous_hearing_monitor] Interrupted by user (Ctrl+C)")
    
    print(f"[continuous_hearing_monitor] Monitoring stopped. Final result: {latest_hearing_texts}")
    return latest_hearing_texts

def process_hearing_input(hearing_texts):
    """
    Process hearing input immediately - convert to phonemes and save
    """
    if not hearing_texts:
        return
    
    print(f"[process_hearing_input] Processing: {hearing_texts}")
    
    # Convert to phonemes
    phonemes = ipa.convert(hearing_texts)
    print("Phonemes:", phonemes)
    
    # Ensure we always work with a list of phonemes
    if isinstance(phonemes, str):
        phoneme_list = phonemes.split()
    elif isinstance(phonemes, list):
        phoneme_list = [p for part in phonemes for p in part.split()]
    else:
        phoneme_list = phonemes
    
    # Convert phonemes to encoded IDs
    ids = []
    for p in phoneme_list:
        byte_values = list(p.encode('utf-8'))
        encoded_id = list_to_reversible_id_sound(byte_values)
        ids.append(encoded_id)
    
    print("Phonemes:", phoneme_list)
    print("Encoded IDs:", ids)
    
    # Save the processed phoneme IDs
    try:
        save_phoneme_ids(ids)
        print(f"[process_hearing_input] ✅ Successfully saved {len(ids)} phoneme IDs")
    except Exception as e:
        print(f"[process_hearing_input] ❌ Error saving phoneme IDs: {e}")

def get_phonemes_from_audio(continuous=False):
    # Load audio - use continuous monitoring if requested
    if continuous:
        hearing_data = continuous_hearing_monitor()
        # When using continuous monitoring, processing happens inside the monitor
        # So we just return the final data without further processing
        return hearing_data
    else:
        hearing_data = load_hearing_input()
    
    if hearing_data:
        text = hearing_data
    else:
        text = "Say my name"

    phonemes = ipa.convert(text)
    print("Phonemes:", phonemes)
    
    # Ensure we always return a list of phonemes
    if isinstance(phonemes, str):
        return phonemes.split()
    elif isinstance(phonemes, list):
        return [p for part in phonemes for p in part.split()]
    else:
        return phonemes
    
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

def list_to_reversible_id_sound(data_list):
    """Encode list of integers to alphanumeric ID."""
    array = np.array(data_list, dtype=np.uint8)
    compressed = zlib.compress(array.tobytes())
    b64 = base64.b64encode(compressed).decode()
    return base64_to_base62_sound(b64)

def id_to_list_sound(encoded_id, original_length):
    """Decode alphanumeric ID back to list of integers."""
    b64 = base62_to_base64_sound(encoded_id)
    compressed = base64.b64decode(b64)
    data_bytes = zlib.decompress(compressed)
    return np.frombuffer(data_bytes, dtype=np.uint8).tolist()[:original_length]

def save_phoneme_ids(
    ids,
    save_dir=r"D:\artist\brainX\CRX\Properties\System1_inputs_ear",
    counter_file_path=r"D:\artist\brainX\CRX\Properties\System1_inputs_ear\last_processed_counter.txt",
    imagine_dir=r"D:\artist\brainX\CRX\Properties\Imagination"
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(imagine_dir, exist_ok=True)

    # Count existing phoneme files
    existing_phonemes = [f for f in os.listdir(save_dir) if f.startswith("phoneme_") and f.endswith(".json")]
    num_phonemes = len(existing_phonemes)

    # Check if urgent node already exists
    urgent_node_path = os.path.join(imagine_dir, "node.json")
    urgent_exists = os.path.exists(urgent_node_path)

    # Ensure counter file exists
    if not os.path.exists(counter_file_path):
        with open(counter_file_path, "w") as f:
            f.write("0")
        last_processed = 0
    else:
        with open(counter_file_path, "r") as f:
            try:
                last_processed = int(f.read().strip())
            except ValueError:
                last_processed = 0

    # Start from next number after last processed
    start_index = last_processed + 1

    for idx, encoded_id in enumerate(ids, start=start_index):
        if num_phonemes < 20:
            # Store in ids_phoneme
            file_name = f"phoneme_{idx:03d}.json"
            save_path = os.path.join(save_dir, file_name)

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(encoded_id, f, ensure_ascii=False, indent=2)

            print(f"✅ Saved {file_name} to: {save_dir}")
            num_phonemes += 1

            # Update counter file
            with open(counter_file_path, "w") as f:
                f.write(str(idx))

        elif not urgent_exists:
            # Store urgent phoneme as single node
            with open(urgent_node_path, "w", encoding="utf-8") as f:
                json.dump(encoded_id, f, ensure_ascii=False, indent=2)

            print(f"🚨 Saved urgent node to: {urgent_node_path}")
            urgent_exists = True

            # Update counter file
            with open(counter_file_path, "w") as f:
                f.write(str(idx))

        else:
            # Both directories are full → ignore
            print("⚠️ Directories full (20 phonemes + urgent node exists). Ignoring phoneme.")
            continue

def main(continuous=False):
    if continuous:
        # In continuous mode, processing happens inside the monitor
        final_hearing_texts = get_phonemes_from_audio(continuous=True)
        print(f"[main] Final hearing texts when monitoring stopped: {final_hearing_texts}")
    else:
        # Extract phonemes
        phonemes = get_phonemes_from_audio(continuous=False)
        print("Phonemes:", phonemes)

        ids = []
        for p in phonemes:
            byte_values = list(p.encode('utf-8'))
            encoded_id = list_to_reversible_id_sound(byte_values)
            ids.append(encoded_id)

        print("Phonemes:", phonemes)
        print("Encoded IDs:", ids)

        # Saving
        save_phoneme_ids(ids)

if __name__ == "__main__":
    main(continuous=True)
