
# system 1 is the brain
# intelligence + consciousness(re-circulating nodes gives consciousness)

import json
import time
import os
import zlib
import base64
import numpy as np
import matplotlib.pyplot as plt
import struct
import cv2
from difflib import SequenceMatcher

from Focusing_agent import focusing_agent, list_to_reversible_id
from Imagine import ImagineAgent
from Hearing_agent import base62_to_base64_sound
from shared_config import SharedConfig
from Imagination import Imagination

def dict_match(dict1, dict2, tolerance=0.05):

    def compare_values(val1, val2):
        if isinstance(val1, tuple) and isinstance(val2, tuple):
            if len(val1) != len(val2):
                return False
            return all(abs(a - b) < 0.001 for a, b in zip(val1, val2))
        return val1 == val2
    
    # Convert tuple-wrapped strings -> plain string
    if isinstance(dict1, tuple) and len(dict1) == 1:
        dict1 = dict1[0]
    if isinstance(dict2, tuple) and len(dict2) == 1:
        dict2 = dict2[0]
    # Strip unwanted outer quotes/brackets if present
    if isinstance(dict1, str):
        dict1 = dict1.strip("[]'\"()")
    if isinstance(dict2, str):
        dict2 = dict2.strip("[]'\"()")

    dict1 = str(dict1)
    dict1 = dict1.strip("()[],'\" ")

    dict2 = str(dict2)
    dict2 = dict2.strip("()[],'\" ")
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
            #print(f"len_diff_ratio - {len_diff_ratio}")
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
            print(f"dict1 - {dict1}")
            print(f"dict2 - {dict2}")

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

def ring_tree(data=None, training=False, organizing=False, initial_grouping=False, testing=False, continuation=False, destabilizing_mechanism=False, previous_result=None, eye=False, ear=False, image=None, importance_map=None):
    #importance_map = load_importance_map()
    image1 = []

    # Load existing state when the function is first called
    if not hasattr(ring_tree, 'surface'):
        # Try to load existing state from drive
        load_ring_tree_from_drive()
        ring_tree.last_branch_match = None # for remmebering previous consecutive matched branch

        # If loading fails or no previous state, initialize
        if not hasattr(ring_tree, 'surface'):
            ring_tree.surface = {}
            ring_tree.surface_rankings = {"R'": [], "R''": [], "R'''": []}
            ring_tree.access_counts = {} #for trees use access counts to create windows,
            ring_tree.access_counts_branch = {}  #for connections use access counts branch to remove least used connections
            ring_tree.windows = {}
            ring_tree.branches = {}
            ring_tree.branch_connections = {}
            ring_tree.cross_branch_connections = {}
            ring_tree.tree_counter = 0
            ring_tree.branch_counter = 0
            ring_tree.cross_branch_counter = 0
            ring_tree.current_tree = None
            ring_tree.last_branch_match = None # for remmebering previous consecutive matched branch

        ring_tree.min_trees_for_ranking = 10
        ring_tree.activation_thresholds = {"detach": 60}

    def compare_values(val1, val2):
        if isinstance(val1, tuple) and isinstance(val2, tuple):
            if len(val1) != len(val2):
                return False
            return all(abs(a - b) < 0.001 for a, b in zip(val1, val2))
        return val1 == val2

    def dict_match(dict1, dict2, tolerance=0.05):

        # Convert tuple-wrapped strings -> plain string
        if isinstance(dict1, tuple) and len(dict1) == 1:
            dict1 = dict1[0]
        if isinstance(dict2, tuple) and len(dict2) == 1:
            dict2 = dict2[0]
        # Strip unwanted outer quotes/brackets if present
        if isinstance(dict1, str):
            dict1 = dict1.strip("[]'\"()")
        if isinstance(dict2, str):
            dict2 = dict2.strip("[]'\"()")

        dict1 = str(dict1)
        dict1 = dict1.strip("()[],'\" ")

        dict2 = str(dict2)
        dict2 = dict2.strip("()[],'\" ")
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
                #print(f"len_diff_ratio - {len_diff_ratio}")
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
                print(f"dict1 - {dict1}")
                print(f"dict2 - {dict2}")

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

    def update_rankings():
        """Update rankings of trees based on access patterns"""
        if len(ring_tree.surface) < ring_tree.min_trees_for_ranking:
            return "Not enough trees for ranking system"

        # Calculate total access per tree
        tree_access = {}
        for tree_name, counts in ring_tree.access_counts.items():
            if tree_name in ring_tree.surface:  # Only existing trees
                tree_access[tree_name] = sum(counts)

        # Sort trees by access (descending)
        sorted_trees = sorted(tree_access.items(), key=lambda x: x[1], reverse=True)

        # Reset as lists
        ring_tree.surface_rankings = {"R'": [], "R''": [], "R'''": []}

        total_trees = len(sorted_trees)
        r_prime_count = max(1, total_trees // 3)
        r_double_prime_count = max(1, total_trees // 3)

        # Assign R' (top third or at least 1)
        for i in range(min(r_prime_count, len(sorted_trees))):
            ring_tree.surface_rankings["R'"].append(sorted_trees[i][0])

        # Assign R'' (middle third or at least 1)
        for i in range(r_prime_count, min(r_prime_count + r_double_prime_count, len(sorted_trees))):
            ring_tree.surface_rankings["R''"].append(sorted_trees[i][0])

        # Assign R''' (remaining bottom third)
        for i in range(r_prime_count + r_double_prime_count, len(sorted_trees)):
            ring_tree.surface_rankings["R'''"].append(sorted_trees[i][0])

        r1 = len(ring_tree.surface_rankings["R'"])
        r2 = len(ring_tree.surface_rankings["R''"])
        r3 = len(ring_tree.surface_rankings["R'''"])
        return f"Updated rankings: R': {r1} trees, R'': {r2} trees, R''': {r3} trees"

    def find_matching_prefix(input_data, min_prefix=2):
        """
        Find the longest matching prefix between input_data and any existing tree,
        its cross-branch connections, or branches.
        Priority:
          1. Consecutive matches in main tree
          2. Consecutive matches in cross-branch transition nodes (receiver_end↔donor_start or donor_end↔receiver_start)
          3. Consecutive matches in branches
          4. Fallback: non-consecutive matches in main tree
        """
        best_match = (None, -1, 0, False)
        # (name, start_idx, match_length, is_consecutive)

        if not ring_tree.surface:
            return best_match

        # --- Helper to check matches ---
        def check_sequence(seq_name, seq_data, rank):
            """
            Returns (name, start_idx, match_length, is_consecutive)
            """

            # Prefer exact equality for strings to avoid false positives like
            # 'WHYwxscuLOgX3OZW' vs 'WHYwxscuMKlsZGJM'. Fall back to dict_match otherwise.
            def matches(a, b):
                if isinstance(a, str) and isinstance(b, str):
                    return a == b
                return dict_match(a, b)

            n_in = len(input_data)
            n_seq = len(seq_data)

            # --- Consecutive check (sliding window) ---
            if n_in <= n_seq:
                for i in range(0, n_seq - n_in + 1):
                    window = seq_data[i:i + n_in]
                    if all(matches(a, b) for a, b in zip(input_data, window)):
                        return (seq_name, i, n_in, True)

            # --- Non-consecutive check (order-preserving subsequence) ---
            # Greedily walk seq_data once, tracking the first matched index.
            first_idx = -1
            last_taken = -1
            match_len = 0

            for a in input_data:
                j = last_taken + 1
                while j < n_seq and not matches(a, seq_data[j]):
                    j += 1
                if j < n_seq:  # found next in-order match
                    if first_idx == -1:
                        first_idx = j
                    last_taken = j
                    match_len += 1
                # if not found, just skip this input item and continue

            return (seq_name, first_idx, match_len, False)

        # --- Search main trees ---
        for rank in ["R'", "R''", "R'''"]:
            for tree_name in ring_tree.surface_rankings.get(rank, []):
                tree_data = ring_tree.surface.get(tree_name, [])

                # Step 1: Check main tree
                tree_match = check_sequence(tree_name, tree_data, rank)

                if tree_match[3] and tree_match[2] >= min_prefix:
                    # ✅ Consecutive match in main tree
                    return tree_match

                # Step 2: If no consecutive, check cross connections of this tree
                for conn_name, conn_data in ring_tree.cross_branch_connections.items():
                    donor = conn_data["trees"]["donor"]
                    receiver = conn_data["trees"]["receiver"]

                    donor_tree = donor["name"]
                    receiver_tree = receiver["name"]

                    donor_nodes = ring_tree.surface.get(donor_tree, [])[donor["start_index"]: donor["end_index"] + 1]
                    receiver_nodes = ring_tree.surface.get(receiver_tree, [])[receiver["start_index"]: receiver["end_index"] + 1]

                    # Condition: check only if current tree participates in this cross connection
                    if tree_name in [donor_tree, receiver_tree]:

                            # --- Receiver end → Donor start ---
                          if receiver_nodes and donor_nodes:
                              if (dict_match(receiver_nodes[-1], input_data[0]) and
                                  dict_match(donor_nodes[0], input_data[min(1, len(input_data)-1)])):
                                  print(f"Consecutive match at receiver_end→donor_start in {conn_name}")
                                  return (
                                      donor_tree,                   # tree name → donor
                                      donor["start_index"],         # start index of donor
                                      len(input_data),              # match length
                                      True                          # is_consecutive
                                  )

                          # --- Donor end → Receiver start ---
                          if donor_nodes and receiver_nodes:
                              if (dict_match(donor_nodes[-1], input_data[0]) and
                                  dict_match(receiver_nodes[0], input_data[min(1, len(input_data)-1)])):
                                  print(f"Consecutive match at donor_end→receiver_start in {conn_name}")
                                  return (
                                      receiver_tree,                # tree name → receiver
                                      receiver["start_index"],      # start index of receiver
                                      len(input_data),              # match length
                                      True                          # is_consecutive
                                  )

                # Step 3: If still no consecutive, check branches
                for branch_name, branch_info in ring_tree.branches.items():
                    if branch_info.get("reference_tree") == tree_name:
                        branch_data = branch_info.get("unique_data", [])
                        branch_match = check_sequence(branch_name, branch_data, rank)

                        if branch_match[3] and branch_match[2] >= min_prefix:
                            # ✅ Found consecutive match in branch
                            return branch_match

                # Step 4: If no branch consecutive, keep main tree non-consecutive
                if tree_match[2] > best_match[2]:
                    best_match = tree_match

        return best_match

    def find_return_point(branch_data, main_tree):
        """Find where the branch can return to the main tree"""
        tree_data = ring_tree.surface[main_tree]

        # Check the first node in branch data
        first_node = branch_data[0]
       # print(f"first_node - {first_node}")
        for idx, node in enumerate(tree_data):
           # print(f"Checking node {idx} - {node}")
            if dict_match(first_node, node):
                # Found a matching node - return to tree from this point
                return idx, idx + 1

        return None, None

    def create_branch(tree_name, initial_branch_data, common_prefix_length):

        # --- Step 1: Validate initial data strictly against current main tree ---
        matched_initial_data = []
        any_match_found = False  # <-- track if at least one matched

        for item in initial_branch_data:
            found_match = False
            for tree_data_node in ring_tree.surface.get(tree_name, []):
                if dict_match(item, tree_data_node):
                    matched_initial_data.append(item)
                    found_match = True
                    any_match_found = True
                    break  # stop checking once matched
            if not found_match:
                print(f"Initial data item {item} not found in main node {tree_name}.")

        # ✅ Only exit if **none** of the items matched
        if not any_match_found:
            return f"Branch creation aborted: no nodes found in {tree_name}"

        # Initialize accumulated data with only confirmed matches
        accumulated_data = matched_initial_data[:]
        continue_gathering = True

        # Rule of thumb: sound inputs have short sequences
        is_sound_input = len(accumulated_data[-1]) < 50  # adjust threshold as needed

        while continue_gathering:
            print(f"accumulated_data[-1] - {accumulated_data[-1]}")
            if is_sound_input:
                # Get additional sound data
                additional_data = get_next_sound_input(mode="organizing")
                print(f"Sound additional_data: {additional_data}")
                boolean_flag = additional_data is not None
            else:
                # Get additional image data
                result2 = focusing_agent(
                    image, image1, importance_map,
                    dual_process=True,
                    request_input=accumulated_data[-1],
                    branch=True
                )

                if isinstance(result2, tuple) and len(result2) == 2:
                    additional_data, boolean_flag = result2
                    print(f"Additional data: {additional_data}, Boolean Flag: {boolean_flag}")
                else:
                    boolean_flag = result2
                    print(f"Boolean Flag: {boolean_flag}")
                    additional_data = None

            if boolean_flag is None or additional_data is None:
                print("No more additional data. Ending data gathering.")
                break

            # --- Step 2: Match additional data strictly within the same main tree ---
            matched_items = []
            if isinstance(additional_data, (tuple, list)):
                for data_item in additional_data:
                    if data_item is None:
                        continue
                    for tree_data_node in ring_tree.surface.get(tree_name, []):
                        if dict_match(data_item, tree_data_node):
                            matched_items.append(data_item)
                            break
            else:  # single item
                for tree_data_node in ring_tree.surface.get(tree_name, []):
                    if dict_match(additional_data, tree_data_node):
                        matched_items.append(additional_data)
                        break

            # If no matches found in current tree → stop gathering
            if not matched_items:
                print(f"Additional data not found in main node {tree_name}. Stopping.")
                continue_gathering = False
            else:
                accumulated_data.extend(matched_items)

        if continue_gathering == False:
            return False

        branch_name = f"branch_{ring_tree.branch_counter}"
        ring_tree.branch_counter += 1
        ring_tree.access_counts_branch[branch_name] = 0

        # --- Step 3: Finalize branch creation with only matched data ---
        return_from_idx, return_to_idx = find_return_point(accumulated_data, tree_name)
        print(f"return_from_idx - {return_from_idx}, return_to_idx - {return_to_idx}")

        ring_tree.branches[branch_name] = {
            "reference_tree": tree_name,
            "reference_length": len(accumulated_data),
            "unique_data": accumulated_data,
            "returns_to_tree": return_to_idx is not None,
            "return_from_idx": return_from_idx,
            "return_to_idx": return_to_idx
        }

        ring_tree.branch_connections[branch_name] = {
            "parent_tree": tree_name,
            "branch_point_index": return_from_idx,
            "unique_data": accumulated_data,
            "returns_to_tree": return_to_idx is not None,
            "return_from_branch_idx": len(ring_tree.branches[branch_name]["unique_data"]) - 1 if return_to_idx is not None else None
        }

        message = f"Created branch {branch_name} from {tree_name}\n"
        message += f"Branch-specific matched data: {accumulated_data}\n"
        if return_to_idx is not None:
            message += f"Branch returns to tree at index {return_to_idx}\n"

        return message

    def extend_branch(branch_name, additional_branch_data, common_prefix_length, image=None, image1=None, importance_map=None):
        """
        Extend an existing branch by gathering new data and appending it to its unique_data.

        Logic:
        - Normal extension: extend with nodes that match the branch's reference surface.
        - Special case: if the forward node in branch does NOT match current input,
          but input is found in main tree surface, create a new branch (branch from branch), but cannot extend as forward node is present so creating a new.
        - If input not found in either branch or main tree → stop.
        """

        if branch_name not in ring_tree.branches:
            print(f"Branch '{branch_name}' not found.")
            return None

        branch_info = ring_tree.branches[branch_name]
        tree_name = branch_info["reference_tree"]

        # Copy current branch data into accumulator
        accumulated_data = branch_info["unique_data"] + additional_branch_data.copy()
        continue_gathering = True
        print(f"accumulated_data - {accumulated_data}")
        # Detect type of input (sound or image)
        is_sound_input = len(accumulated_data[-1]) < 50  # adjust threshold as needed

        new_accumulated_data = []

        while continue_gathering:
            print(f"accumulated_data[-1] - {accumulated_data[-1]}")
            print(f"Extending branch {branch_name}...")
            matched_nodes = []
            matched_nodes.append(accumulated_data[-1])
            # --- Step 1: Fetch candidate additional data ---
            if is_sound_input:
                additional_data = get_next_sound_input(mode="organizing")
                print(f"Sound additional_data: {additional_data}")
                boolean_flag = additional_data is not None
            else:
                result2 = focusing_agent(
                    image, image1, importance_map,
                    dual_process=True,
                    request_input=accumulated_data[-1],
                    branch=True
                )
                print(f"Image result2 - {result2}")

                if isinstance(result2, tuple) and len(result2) == 2:
                    additional_data, boolean_flag = result2
                    print(f"Additional data: {additional_data}, Boolean Flag: {boolean_flag}")
                else:
                    boolean_flag = result2
                    additional_data = None
                    print(f"Boolean Flag: {boolean_flag}")

            if boolean_flag is None or additional_data is None:
                print("No more additional data. Ending branch extension.")
                break

            # --- Step 2: Check if candidate exists in current branch's reference tree surface ---
            branch_info = ring_tree.branches[branch_name]
            reference_tree = branch_info.get("reference_tree")


            if reference_tree and reference_tree in ring_tree.surface:
                for node in ring_tree.surface[reference_tree]:
                    if isinstance(additional_data, (list, tuple)):
                        for ad in additional_data:
                            if ad is not None and dict_match(ad, node):
                                matched_nodes.append(ad)
                    else:
                        if dict_match(additional_data, node):
                            matched_nodes.append(additional_data)

            # ---(not needed) Step 3: Handle forward node mismatch (branch from branch creation) ---
            forward_idx = len(accumulated_data) - common_prefix_length
            if forward_idx < len(branch_info["unique_data"]):
                forward_node = branch_info["unique_data"][forward_idx]
                if not dict_match(additional_data, forward_node):
                    # check if input exists in main tree surface
                    if tree_name in ring_tree.surface:
                        found_in_main = False
                        for node in ring_tree.surface[tree_name]:
                            if dict_match(additional_data, node):
                                found_in_main = True
                                break

                        if found_in_main:
                            # create a new branch from current branch
                            new_branch_name = f"{branch_name}_sub_{len(ring_tree.branches)}"

                            # unique data = only the new input that matched main tree but not branch front
                            new_unique_data = [additional_data]

                            # finalize with return mapping
                            return_from_idx, return_to_idx = find_return_point(new_unique_data, branch_name)

                            ring_tree.branches[new_branch_name] = {
                                "reference_tree": branch_name,  # parent branch
                                "reference_length": len(new_unique_data),
                                "unique_data": new_unique_data,
                                "returns_to_tree": return_to_idx is not None,
                                "return_from_idx": return_from_idx,
                                "return_to_idx": return_to_idx
                            }

                            print(f"🌱 Created new sub-branch {new_branch_name} from {branch_name}, "
                                  f"with input {additional_data}, "
                                  f"returns_to_tree={return_to_idx is not None}")
                            return f"New sub-branch {new_branch_name} created from {branch_name}"
                        else:
                            print("❌ Input not found in main tree either. Stopping extension.")
                            break

            # --- Step 4: Extend only with matched nodes if available ---
            if not matched_nodes:
                print(f"❌ No match in reference tree '{reference_tree}' surface. "
                      f"Stopping extension for {branch_name}.")
                continue_gathering = False
                break


            for m in matched_nodes:
                new_accumulated_data.append(m)
            print(f"✅ Extended with matched nodes: {matched_nodes}")
            print(f"Updated accumulated_data - {new_accumulated_data}")

        if continue_gathering == False:
            return False
        # --- Step 5: Finalize branch updates ---
        # Only keep unique nodes (avoid repetition)
        branch_info["unique_data"].extend(new_accumulated_data)

        branch_info["reference_length"] = len(branch_info["unique_data"])

        # Preserve original metadata
        ring_tree.branches[branch_name] = branch_info

        # Update branch_connections without dropping info
        if branch_name in ring_tree.branch_connections:
            ring_tree.branch_connections[branch_name]["unique_data"] = branch_info["unique_data"]
            ring_tree.branch_connections[branch_name]["reference_length"] = branch_info["reference_length"]

        ring_tree.access_counts_branch[branch_name] += 1

        message = f"Extended branch {branch_name}\n"
        message += f"Updated branch data: {branch_info['unique_data']}\n"
        message += f"Reference length: {branch_info['reference_length']}\n"


        return message

    def update_tree_with_new_nodes(tree_name, data):
        """Update the specified tree with new nodes from data,
        append progressively until a match is found in another main tree.
        """

        # --- Step 1: Append the initial input data at the end ---
        tree_data = ring_tree.surface[tree_name]
        nodes_to_add = data if isinstance(data, list) else [data]
        insertion_position = len(tree_data)

        for i, node in enumerate(nodes_to_add):
            tree_data.insert(insertion_position + i, node)
            ring_tree.access_counts[tree_name].insert(insertion_position + i, 0)

        # Adjust windows if needed
        if ring_tree.windows[tree_name] >= insertion_position:
            ring_tree.windows[tree_name] += len(nodes_to_add)

        # Adjust branch connection indices
        for conn_name, conn_data in ring_tree.branch_connections.items():
            if conn_data["parent_tree"] == tree_name and conn_data["branch_point_index"] >= insertion_position:
                conn_data["branch_point_index"] += len(nodes_to_add)

            # Update similarity nodes indices if needed
            for i, sim_node in enumerate(conn_data.get("similarity_nodes", [])):
                idx, sim_tree, sim_idx = sim_node
                if sim_tree == tree_name and sim_idx >= insertion_position:
                    conn_data["similarity_nodes"][i] = (idx, sim_tree, sim_idx + len(nodes_to_add))

        # --- Step 2: Progressive gathering loop ---
        accumulated_data = nodes_to_add[:]  # Start with the newly appended nodes
        continue_gathering = True

        # Rule of thumb: sound inputs have short sequences
        is_sound_input = len(accumulated_data[-1]) < 50 if accumulated_data else False

        while continue_gathering:
            print(f"Gathering after: {accumulated_data[-1]}")

            if is_sound_input:
                additional_data = get_next_sound_input(mode="organizing")
                print(f"Sound additional_data: {additional_data}")
                boolean_flag = additional_data is not None
            else:
                result2 = focusing_agent(
                    image, image1, importance_map,
                    dual_process=True,
                    request_input=accumulated_data[-1],
                    branch=True
                )
                print(f"Image result2 - {result2}")

                if isinstance(result2, tuple) and len(result2) == 2:
                    additional_data, boolean_flag = result2
                    print(f"Additional data: {additional_data}, Boolean Flag: {boolean_flag}")
                else:
                    boolean_flag = result2
                    additional_data = None
                    print(f"Boolean Flag: {boolean_flag}")

            if not boolean_flag or additional_data is None:
                print("No more additional data. Ending data gathering.")
                break

            if not boolean_flag or additional_data is None:
                print("No more additional data. Ending data gathering.")
                break

            # Check if this new_data exists in any main tree (surface)
            found_in_main = False
            for other_tree, other_data in ring_tree.surface.items():
                for item in other_data:
                    if dict_match(item, additional_data):
                        found_in_main = True
                        break
                if found_in_main:
                    break

            if found_in_main:
                print(f"Found match in main tree. Stopping gathering to update with new node on {tree_name}.")
                return True, "main node updated"
            else:
                # Append this new data into the same tree
                pos = len(ring_tree.surface[tree_name])
                ring_tree.surface[tree_name].append(additional_data)
                ring_tree.access_counts[tree_name].append(0)
                accumulated_data.append(additional_data)
                print(f"Appended additional data: {additional_data}")

        # If we exhausted gathering without finding match
        return False, "stopped gathering"

    def update_access_counts(tree_name, node_index):
        """Update access counts for a node and check if window should be moved"""
        if tree_name in ring_tree.access_counts:
            # Ensure access_counts array is large enough
            if len(ring_tree.access_counts[tree_name]) <= node_index:
                # Extend the array to accommodate the node index
                extension = [0] * (node_index - len(ring_tree.access_counts[tree_name]) + 1)
                ring_tree.access_counts[tree_name].extend(extension)

            # Now we can safely increment the access count
            ring_tree.access_counts[tree_name][node_index] += 1
            #print(f" access count - {ring_tree.access_counts[tree_name][node_index]}")
            # Update window to most frequently accessed node
            max_access = max(ring_tree.access_counts[tree_name])
            max_index = ring_tree.access_counts[tree_name].index(max_access)
            #print(f"max index - {max_index}")
            old_window = ring_tree.windows.get(tree_name, 0)
            ring_tree.windows[tree_name] = max_index

            # Check if access count exceeds threshold for detaching
            if ring_tree.access_counts[tree_name][max_index] >= ring_tree.activation_thresholds["detach"]:
                return detach_window_to_new_tree(tree_name, max_index)

            return f"Updated window for {tree_name} from {old_window} to {max_index}, access count now: {ring_tree.access_counts[tree_name][node_index]}"
        return "No access count updated - tree not found"

    def detach_window_to_new_tree(tree_name, window_index):
        """
        Detach a window of continuous activated nodes and all following nodes
        from the original tree and create a new single tree with them.
        """
        if tree_name not in ring_tree.surface:
            return f"Tree {tree_name} not found"

        # ✅ safeguard: if window_index already at 0 and it is already in surface, skip detaching
        if window_index == 0 and tree_name in ring_tree.surface:
            return (f"Window index {window_index} in {tree_name} is already at the "
                    f"surface (highest frequency). No detachment performed.")

        tree_data = ring_tree.surface[tree_name]
        access_counts = ring_tree.access_counts[tree_name]
        tree_size = len(tree_data)

        # Find a continuous sequence of activated nodes (3-5 nodes)
        # Start with the most activated node
        window_start = window_index
        window_end = window_index + 1

        # Look for activated nodes before the window_index (up to 2 positions)
        for i in range(1, 3):
            prev_idx = (window_index - i) % tree_size
            # Only include if it's consecutive (physically adjacent) and has been activated
            if prev_idx == (window_index - i) and access_counts[prev_idx] > 0:
                window_start = prev_idx
            else:
                break  # Stop if we hit a non-activated node or non-consecutive position

        # Look for activated nodes after the window_index (up to 2 positions)
        for i in range(1, 3):
            next_idx = (window_index + i) % tree_size
            # Only include if it's consecutive and has been activated
            if next_idx == (window_index + i) and access_counts[next_idx] > 0:
                window_end = next_idx + 1
            else:
                break  # Stop if we hit a non-activated node or non-consecutive position

        # Check if we have at least 3 continuous nodes in our window
        if window_end - window_start < 3:
            return f"Not enough continuous activated nodes around {window_index} in {tree_name} to form a window"

        # Create a new tree from the window area plus all following nodes
        new_tree_name = f"tree_{ring_tree.tree_counter}"
        ring_tree.tree_counter += 1

        # Copy nodes from the window start to the end of the tree
        # This includes both the activated window and all following nodes
        new_tree_data = tree_data[window_start:]

        # Store as a new tree
        ring_tree.surface[new_tree_name] = new_tree_data

        # Initialize access counts for new tree
        ring_tree.access_counts[new_tree_name] = [0] * len(new_tree_data)
        ring_tree.windows[new_tree_name] = 0

        # Update the original tree to only contain nodes before the window
        ring_tree.surface[tree_name] = tree_data[:window_start]
        ring_tree.access_counts[tree_name] = access_counts[:window_start]

        # Add new tree to highest rank if ranking system is active
        if len(ring_tree.surface) >= ring_tree.min_trees_for_ranking:
            if new_tree_name not in ring_tree.surface_rankings["R'"]:
                ring_tree.surface_rankings["R'"].append(new_tree_name)
               # ring_tree.surface_rankings["R'"][new_tree_name].append(new_tree_data)

        update_rankings()

        result_message = (f"Detached window of {window_end - window_start} continuous nodes "
                        f"plus {len(new_tree_data) - (window_end - window_start)} following nodes "
                        f"to new tree {new_tree_name}. "
                        f"Original tree {tree_name} now has {len(ring_tree.surface[tree_name])} nodes")

        return result_message

    def add_connection_if_absent(donor_name, donor_start_idx, donor_end_idx, donor_type,
                                 receiver_name, receiver_start_idx, receiver_end_idx,  receiver_type):
        """
        Create a bidirectional cross connection if it doesn't already exist.
        Supports both surface trees and branches by storing type metadata.
        """

        connection_name = f"cross_connection_{ring_tree.cross_branch_counter}"
        ring_tree.cross_branch_counter += 1

        conn_dict = {
            "trees": {
                "donor": {
                    "name": donor_name,
                    "type": donor_type,   # "tree" or "branch"
                    "start_index": donor_start_idx,
                    "end_index": donor_end_idx
                },
                "receiver": {
                    "name": receiver_name,
                    "type": receiver_type,
                    "start_index": receiver_start_idx,
                    "end_index": receiver_end_idx
                }
            },
            "connection_type": "cross",
            "directions": {
                "forward": {
                    "from": {"name": donor_name, "type": donor_type, "index": donor_end_idx},
                    "to":   {"name": receiver_name, "type": receiver_type, "index": receiver_start_idx}
                },
                "reverse": {
                    "from": {"name": receiver_name, "type": receiver_type, "index": receiver_start_idx},
                    "to":   {"name": donor_name, "type": donor_type, "index": donor_end_idx}
                }
            },
            "similarity_nodes": [
                (receiver_start_idx, donor_name, donor_start_idx),
                (donor_end_idx, receiver_name, receiver_end_idx)
            ]
        }

        ring_tree.cross_branch_connections[connection_name] = conn_dict
        ring_tree.access_counts_branch[connection_name] = 0  # start usage counter

        print(f"[DEBUG] Added cross connection {connection_name}")
        print(f"   Donor:    {donor_type}:{donor_name}[{donor_start_idx} → {donor_end_idx}]")
        print(f"   Receiver: {receiver_type}:{receiver_name}[{receiver_start_idx} → {receiver_end_idx}]")
        print(f"   Forward : {donor_type}:{donor_name}[{donor_end_idx}] → {receiver_type}:{receiver_name}[{receiver_start_idx}]")
        print(f"   Reverse : {receiver_type}:{receiver_name}[{receiver_start_idx}] → {donor_type}:{donor_name}[{donor_end_idx}]")

    def sequence_in_connection(seq, connection_nodes):
        """Check if seq appears as a consecutive subsequence in connection_nodes."""
        for i in range(len(connection_nodes) - len(seq) + 1):
            if connection_nodes[i:i+len(seq)] == seq:
                return True
        return False

    def find_matches_with_sources(ids, mode):

        """
        Refactored version:
        - Processes input in batches of 5.
        - Checks for matches in existing cross connections first.
        - If found, validates extension with get_focus_ids() or creates secondary cross connection if partial match.
        - If not found, searches ring_tree.surface to create new secondary connection.
        - After creation, extends connection by checking donor tree with new inputs one by one.
        - no need for the extra data to create secondary connection. as that extra data will in the next time be the main data and will
          help to create the new second connection from the newly created second connection(now becomes the main connection).
        - sometimes, the node match may not be consecutive, but over time, as branchs accumulate important nodes in correct order,
          will help to create cross connection. even if not connected now, after some time, the branch will create order which help cross connection created later.
        """
        ids_0 = str(ids[0])
        ids_0 = ids_0.strip("()[],'\" ")
        input_is_sound = isinstance(ids_0, str) and len(ids_0) < 50

        def get_focus_ids():
            if input_is_sound:
                val = get_next_sound_input(mode)
                return [val] if not isinstance(val, list) else val
            else:
                focus, _, _ = focusing_agent(image, image1, importance_map, training=True)
                return [list_to_reversible_id(f) for f in focus]


        def get_nodes(name, idx, batch_len):
            """
            Fetch nodes from ring_tree.surface (trees) or ring_tree.branches.
            Use the name to disambiguate.
            """
            if name.startswith("branch"):   # force branch lookup
                binfo = ring_tree.branches.get(name)
                if not binfo:
                    raise KeyError(f"[get_nodes] Branch '{name}' not found in ring_tree.branches")
                branch_nodes = binfo["unique_data"]
                return branch_nodes[idx:idx + batch_len]
            
            elif name.startswith("tree"):   # force tree lookup
                if name not in ring_tree.surface:
                    raise KeyError(f"[get_nodes] Tree '{name}' not found in ring_tree.surface")
                return ring_tree.surface[name][idx:idx + batch_len]
            
            else:
                raise ValueError(f"[get_nodes] Unknown node name '{name}' (must start with 'tree' or 'branch')")

        batch = ids  # always take 5 inputs
        print(f"[DEBUG] Processing batch: {batch}")

        # --- Step 1: Check existing cross connections ---
        for conn_name, conn_info in ring_tree.cross_branch_connections.items():
            # Use .get() to safely access 'trees', defaulting to {} if missing
            trees = conn_info.get("trees", {})

            if not trees:
                continue  # skip if trees section is missing

            donor_info = trees.get("donor", {})
            receiver_info = trees.get("receiver", {})

            # If either side is missing, skip this connection
            if not donor_info or not receiver_info:
                continue

            donor_name = donor_info.get("name")
            donor_type = donor_info.get("type")
            donor_start = donor_info.get("start_index")
            donor_end = donor_info.get("end_index")

            receiver_name = receiver_info.get("name")
            receiver_type = receiver_info.get("type")
            receiver_start = receiver_info.get("start_index")
            receiver_end = receiver_info.get("end_index")

            # If any critical info is missing, skip safely
            if None in [donor_name, donor_type, donor_start, donor_end,
                        receiver_name, receiver_type, receiver_start, receiver_end]:
                continue

            # Now use donor_type/receiver_type when comparing nodes
            donor_nodes = get_nodes(donor_name, donor_start, len(batch))
            receiver_nodes = get_nodes(receiver_name, receiver_start, len(batch))

            # Convert to just comparable dicts (or however dict_match works)
            connection_nodes = donor_nodes + receiver_nodes

            # --- Case A: Check if batch fully matches in same consecutive order ---
            if sequence_in_connection(batch, connection_nodes):
               # print(f"[DEBUG] Full batch match in existing connection {conn_name}")

                # Try extending connection with extra inputs
                while True:
                    extra_ids = get_focus_ids()
                   # print(f"extra ids = {extra_ids}")

                    if not extra_ids:
                        return True, "connection extended (no more inputs)"

                    # Compare next receiver node with new inputs
                    next_idx = receiver_end + 1
                    if next_idx < len(ring_tree.surface[receiver_name]):
                        next_node = ring_tree.surface[receiver_name][next_idx]
                        if any(dict_match(next_node, eid) for eid in extra_ids):
                            receiver_end += 1
                            conn_info["trees"]["receiver"]["end_index"] = receiver_end
                            print(f"[DEBUG] Extended receiver end index → {receiver_end}")
                        else:
                            return True, "connection extended (stopped at mismatch)"
                    else:
                        return True, "connection extended (receiver tree exhausted)"


        # --- Step 2: If not found in existing connections, search surface trees and branches ---

        def find_longest_consecutive_match(nodes, batch):
            """
            Find the longest consecutive match between tree nodes and batch.
            Returns (best_start_idx, best_len).
            """
            best_start, best_len = None, 0
            for i in range(len(nodes)):
                match_len = 0
                for j in range(len(batch)):
                    if i + j < len(nodes) and dict_match(nodes[i + j], batch[j]):
                        match_len += 1
                    else:
                        break
                if match_len > best_len:
                    best_len = match_len
                    best_start = i
            return best_start, best_len


        # ---- Donor detection ----
        donor, receiver = None, None
        remaining_batch = batch[:]  # copy so we can trim

        for tname, nodes in ring_tree.surface.items():
            start_idx, match_len = find_longest_consecutive_match(nodes, remaining_batch)
            if match_len > 0:
                donor = (tname, start_idx, start_idx + match_len - 1)
                print(f"[DEBUG] Donor found → {donor}, matched {match_len} nodes")
                donor_type = "tree"
                # cut matched part from batch → use only the rest for receiver
                remaining_batch = remaining_batch[match_len:]
                break

        # ---- Receiver detection ----
        if donor and remaining_batch:
            for tname, nodes in ring_tree.surface.items():
                if tname == donor[0]:
                    continue  # skip donor tree
                start_idx, match_len = find_longest_consecutive_match(nodes, remaining_batch)
                if match_len > 0:
                    receiver = (tname, start_idx, start_idx + match_len - 1)
                    print(f"[DEBUG] Receiver found → {receiver}, matched {match_len} nodes")
                    receiver_type = "tree"
                    break

        if not donor or not receiver:
            print("matches - False reason - no donor/receiver found")

        # --- Branch search (donor/receiver) ---
        for bname, binfo in ring_tree.branches.items():
            branch_nodes = binfo["unique_data"]

            # branches
            start_idx, match_len = find_longest_consecutive_match(branch_nodes, batch)
            if match_len > 0 and not donor:
                donor = (bname, start_idx, start_idx + match_len - 1)  # END
                donor_type = "branch"
            elif match_len > 0 and not receiver:
                receiver = (bname, start_idx, start_idx + match_len - 1)  # END
                receiver_type = "branch"


            if donor and receiver:
                break
        # ---- Extension phase (using focus input) ----
        def try_extend_match(donor, receiver, donor_type, receiver_type):
            """
            Extend donor or receiver consecutively using get_focus_ids()
            until mismatch occurs.
            """
            if not donor or not receiver:
                return donor, receiver  # nothing to extend

            donor_tree, d_start, d_end = donor
            receiver_tree, r_start, r_end = receiver

            donor_nodes = (
                ring_tree.surface.get(donor_tree, [])
                if donor_type == "tree"
                else ring_tree.branches[donor_tree]["unique_data"]
            )
            receiver_nodes = (
                ring_tree.surface.get(receiver_tree, [])
                if receiver_type == "tree"
                else ring_tree.branches[receiver_tree]["unique_data"]
            )

            # Case A: donor was found first → try extending receiver
            if d_end >= 0 and r_start >= 0 and d_start <= d_end:
                while True:
                    new_vals = get_focus_ids()
                    if not new_vals:
                        break
                    for val in new_vals:
                        # check if val matches receiver[r_end+1]
                        if r_end + 1 < len(receiver_nodes) and dict_match(val, receiver_nodes[r_end + 1]):
                            print(f"[EXTEND] Receiver extended with {val}")
                            r_end += 1
                        else:
                            # mismatch → stop extension
                            return (donor_tree, d_start, d_end), (receiver_tree, r_start, r_end)

            # Case B: receiver was found first → try extending donor
            elif r_end >= 0 and d_start >= 0 and r_start <= r_end:
                while True:
                    new_vals = get_focus_ids()
                    if not new_vals:
                        break
                    for val in new_vals:
                        if d_end + 1 < len(donor_nodes) and dict_match(val, donor_nodes[d_end + 1]):
                            print(f"[EXTEND] Donor extended with {val}")
                            d_end += 1
                        else:
                            return (donor_tree, d_start, d_end), (receiver_tree, r_start, r_end)

            return (donor_tree, d_start, d_end), (receiver_tree, r_start, r_end)


        # Apply extension logic
        if donor and receiver:
            donor, receiver = try_extend_match(donor, receiver, donor_type, receiver_type)
            print(f"[FINAL] Donor: {donor}, Receiver: {receiver}")

        # --- If both donor and receiver found ---
        if donor and receiver:
            donor_tree, donor_start_idx, donor_len = donor
            receiver_tree, receiver_start_idx, receiver_len = receiver

            donor_nodes = get_nodes(donor_tree, donor_start_idx, len(batch))
            receiver_nodes = get_nodes(receiver_tree, receiver_start_idx, len(batch))

            # Fix: Include donor_nodes length in the range calculation
            donor_match = all(
                dict_match(batch[j], donor_nodes[j]) for j in range(min(len(batch), donor_len, len(donor_nodes)))
            ) if donor_nodes else False

            # Fix: Include receiver_nodes length in the range calculation
            receiver_match = all(
                dict_match(batch[j], receiver_nodes[j]) for j in range(min(len(batch), receiver_len, len(receiver_nodes)))
            ) if receiver_nodes else False

            # Order check → donor/receiver must align consecutively
            donor_range = list(range(donor_start_idx, donor_start_idx + donor_len))
            if (donor_match or receiver_match) and not (donor_range == sorted(donor_range)):
                return False, "more order is needed, branching is required to create consecutive connection"

            # Prevent same-tree donor/receiver
            if donor_tree == receiver_tree:
                return False, "same tree matches found"
            
        if donor and receiver:
            donor_tree, donor_start_idx, donor_end_idx = donor
            receiver_tree, receiver_start_idx, receiver_end_idx = receiver

            # sanity
            assert donor_end_idx >= donor_start_idx
            assert receiver_end_idx >= receiver_start_idx

            print(f"[DEBUG] Donor → {donor_tree}[{donor_start_idx}:{donor_end_idx}]")
            print(f"[DEBUG] Receiver → {receiver_tree}[{receiver_start_idx}:{receiver_end_idx}]")

            # if you need lengths:
            donor_len    = donor_end_idx    - donor_start_idx    + 1
            receiver_len = receiver_end_idx - receiver_start_idx + 1

          #  if donor and receiver:
                # Try to extend with extra inputs
            #    while True:
                #    extra_ids = get_focus_ids()

                    # ✅ Stop if no IDs or all are None
                #    if not extra_ids or all(eid is None for eid in extra_ids):
                 #       break

                    # ✅ Filter out None values
                 #   extra_ids = [eid for eid in extra_ids if eid is not None]

                #    if not extra_ids:
                 #       break

                  #  extended_nodes = ring_tree.surface[donor_tree][donor_end_idx + 1:
                    #                                              donor_end_idx + 1 + len(extra_ids)]

                   # if all(dict_match(node, eid) for node, eid in zip(extended_nodes, extra_ids)):
                        #donor_end_idx += len(extra_ids)
                        #receiver_end_idx += len(extra_ids)
                   #     print(f"[DEBUG] Extended donor end_idx to {donor_end_idx}")
                   # else:
                   #     break

            # 2️⃣ Now store the final extended connection once
            add_connection_if_absent(
                donor_tree, donor_start_idx, donor_end_idx, donor_type,
                receiver_tree, receiver_start_idx, receiver_end_idx, receiver_type
            )
            return True, "new connection stored (stopped extension)"

        return False, "no donor/receiver found"

    def searching_via_common_information(data, mode="organizing"):
        """Simplified: delegates sequence/connection logic to find_matches_with_sources."""

        if not data:
            return False, "no data"

        def get_focus_ids():
            data_0 = str(data[0])
            data_0 = data_0.strip("()[],'\" ")

            input_is_sound = isinstance(data_0, str) and len(data_0) < 50

            if input_is_sound:
                val = get_next_sound_input(mode)
                return [val] if not isinstance(val, list) else val
            else:
                focus, _ , _ = focusing_agent(image, image1, importance_map, training=True)
                return [list_to_reversible_id(f) for f in focus]

        # 🔹 Step 1: Find donor tree
        donor_tree = None
        for rank in ["R'", "R''", "R'''"]:
            for tree_name in ring_tree.surface_rankings.get(rank, []):
                # Just check if any element matches
                if any(dict_match(node, d) for d in data for node in ring_tree.surface[tree_name]):
                    donor_tree = tree_name
                    break
            if donor_tree:
                break

        if not donor_tree:
            return False, "no donor"

        # 🔹 Step 2: Loop batches of 5 and pass directly to find_matches_with_sources
        while True:
            batch = []
            batch.extend([i for i in data if i is not None])
            for _ in range(5):
                ids = get_focus_ids()

                # ✅ stop early if ids is None or all None
                if not ids or all(i is None for i in ids):
                    return False, "no new inputs"

                # ✅ collect only valid ids
                batch.extend([i for i in ids if i is not None])

            # ✅ stop if batch is empty OR not enough inputs
            if not batch or len(batch) < 2:
                return False, "no new inputs"

            print(f"batch - {batch}")
            # Skip identical sequences
           # if sequence_exists_in_tree_or_branch(batch):
            #    return False, "same order in tree or branch"

            matches, reason = find_matches_with_sources(batch, mode)
            print(f"matches - {matches} reason - {reason}")
            return matches, reason

    def cleanup_low_usage_connections(min_usage=5):
        """
        Remove branch and cross-branch connections from ring_tree if
        their usage in ring_tree.access_counts_branch is below min_usage.
        """
        to_remove = []

        # Find connections below threshold
        for conn, count in list(ring_tree.access_counts_branch.items()):
            if count < min_usage:
                to_remove.append(conn)

        # Remove from access_counts_branch and respective dictionaries
        for conn in to_remove:
            # Remove from branches
            if conn in ring_tree.branches:
                print(f"Removing branch connection: {conn} (usage={ring_tree.access_counts_branch[conn]})")
                del ring_tree.branches[conn]

            # Remove from cross-branch connections
            if conn in ring_tree.cross_branch_connections:
                print(f"Removing cross-branch connection: {conn} (usage={ring_tree.access_counts_branch[conn]})")
                del ring_tree.cross_branch_connections[conn]

            # Always remove from access_counts_branch
            del ring_tree.access_counts_branch[conn]

    #connection creation for imagination
    def create_cross_connection_for_imagination(imagination_node, activated_node):
        """Create cross-connection when imagination node activates a testing node"""
        
        # Find locations of both nodes in ring_tree structure
        imagination_location = find_node_location(imagination_node)
        activated_location = find_node_location(activated_node)
        
        if imagination_location and activated_location:
            print(f"Creating cross-connection between imagination node and activated node")
            print(f"Imagination node location: {imagination_location}")
            print(f"Activated node location: {activated_location}")
            
            add_connection_if_absent(
                donor_name=imagination_location['tree_name'],
                donor_start_idx=imagination_location['index'],
                donor_end_idx=imagination_location['end_index'],
                donor_type=imagination_location['type'],
                receiver_name=activated_location['tree_name'],
                receiver_start_idx=activated_location['index'],
                receiver_end_idx=activated_location['end_index'],
                receiver_type=activated_location['type']
            )
        else:
            if not imagination_location:
                print("Warning: Could not find imagination node location in ring_tree")
            if not activated_location:
                print("Warning: Could not find activated node location in ring_tree")

    def find_node_location(node_data):
        """Find where a node exists in ring_tree - implement based on your ring_tree structure"""
        
        def find_longest_consecutive_match(nodes, batch):
            """
            Find the longest consecutive match between tree nodes and batch.
            Returns (best_start_idx, best_len).
            """
            best_start, best_len = None, 0
            for i in range(len(nodes)):
                match_len = 0
                for j in range(len(batch)):
                    if i + j < len(nodes) and dict_match(nodes[i + j], batch[j]):
                        match_len += 1
                    else:
                        break
                if match_len > best_len:
                    best_len = match_len
                    best_start = i
            return best_start, best_len

        # Convert single node to batch format for matching
        if not isinstance(node_data, list):
            batch = [node_data]
        else:
            batch = node_data

        # ---- Search in surface trees ----
        for tname, nodes in ring_tree.surface.items():
            start_idx, match_len = find_longest_consecutive_match(nodes, batch)
            if match_len > 0:
                return {
                    'tree_name': tname,
                    'index': start_idx,
                    'end_index': start_idx + match_len - 1,
                    'type': 'tree',
                    'match_length': match_len
                }

        # ---- Search in branches ----
        for bname, binfo in ring_tree.branches.items():
            if "unique_data" in binfo:
                branch_nodes = binfo["unique_data"]
                start_idx, match_len = find_longest_consecutive_match(branch_nodes, batch)
                if match_len > 0:
                    return {
                        'tree_name': bname,
                        'index': start_idx,
                        'end_index': start_idx + match_len - 1,
                        'type': 'branch',
                        'match_length': match_len
                    }

        # Node not found
        return None
        
    # Training mode: create and populate trees
    if training and data:
        if ring_tree.current_tree is None or len(ring_tree.surface.get(ring_tree.current_tree, [])) >= 48:
            ring_tree.current_tree = f"tree_{ring_tree.tree_counter}"
            ring_tree.tree_counter += 1
            ring_tree.surface[ring_tree.current_tree] = []
            ring_tree.access_counts[ring_tree.current_tree] = []
            ring_tree.windows[ring_tree.current_tree] = 0

        if "R'" not in ring_tree.surface_rankings or not isinstance(ring_tree.surface_rankings["R'"], list):
            ring_tree.surface_rankings["R'"] = []

        if ring_tree.current_tree not in ring_tree.surface_rankings["R'"]:
            ring_tree.surface_rankings["R'"].append(ring_tree.current_tree)

        # Duplicate check across ALL trees
        #if any(data in items for items in ring_tree.surface.values()):
        #    message = f"Duplicate detected, not storing: {data}"
         #   return message

        ring_tree.surface[ring_tree.current_tree].append(data)
        ring_tree.access_counts[ring_tree.current_tree].append(0)

        # Check if ranking system should be activated
        if len(ring_tree.surface) == ring_tree.min_trees_for_ranking:
            update_rankings()
            message = f"Stored in {ring_tree.current_tree}: {data} (Item {len(ring_tree.surface[ring_tree.current_tree])} of 5). Ranking system activated!"
        else:
            message = f"Stored in {ring_tree.current_tree}: {data} (Item {len(ring_tree.surface[ring_tree.current_tree])} of 5)"

            # If we already have enough trees, update rankings
            if len(ring_tree.surface) > ring_tree.min_trees_for_ranking:
                update_rankings()

        return message

    # Initial grouping - forms connection to concentrate similar nodes by creating cross connection
    if initial_grouping and data:
        print("initial_grouping of making cross connection of main nodes to assist in full branch formation")

        def is_sound_input(item):
            return isinstance(item, str) and len(item) < 50

        # Decide processing mode
        if eye and not ear:
            # Image-only input (existing flow with focusing_agent)
            print("Mode: Visual only")
            result = focusing_agent(image, image1, importance_map, dual_process=True, request_input=data)
            if isinstance(result, tuple) and len(result) == 2:
                data_2, boolean_flag = result
            else:
                data_2, boolean_flag = None, result
            if not boolean_flag:
                print("Boolean flag is None. Ending function.")
                return
            combined_data = [data, data_2] if data_2 else [data]

        elif ear and not eye:
            # Sound-only input (skip focusing_agent)
            print("Mode: Sound only")
            sound_inputs = []
            sound_inputs.append(data)

            second_sound = get_next_sound_input(mode="initial_grouping")
            if second_sound:
                sound_inputs.append(second_sound)
            else:
                #print("error in getting the second sound input")
                pass
            combined_data = sound_inputs


        elif eye and ear:
            # Mixed mode: process image and sound separately
            image_inputs = []
            sound_inputs = []
            print("Mode: Mixed input (sound + image)")
            if is_sound_input(data):
                sound_inputs.append(data)
                combined_data = data
            else:
                image_inputs.append(data)
                combined_data = data

            # Process image inputs via focusing_agent
            if len(image_inputs) > 0:
                img_result = focusing_agent(image, image1, importance_map, dual_process=True, request_input=image_inputs)
                print(f"img result - {img_result}")
                if isinstance(img_result, tuple) and len(img_result) == 2:
                    img_data_2, boolean_flag = img_result
                    image_inputs.append(img_data_2)
                else:
                    img_data_2, boolean_flag = None, img_result
                    
                if not boolean_flag:
                    print("Boolean flag is None for image part.")
                    pass

                combined_data = image_inputs
            # Ensure both sound and image parts have at least two elements
            if len(sound_inputs) > 2:
                print("Awaiting second sound input...")
                second_sound = get_next_sound_input(mode="organizing")
                if second_sound:
                    sound_inputs.append(second_sound)
                else:
                    #print("error in getting the second sound input")
                    pass

                combined_data = sound_inputs
        else:
            print("No valid sensory mode detected (eye/ear flags missing).")
            return

        print(f"Combined data for processing: {combined_data}")
        #intial gathering of nodes to set all main nodes to act as one node which helps in full branch formation
        if combined_data:
                  common_information, reason = searching_via_common_information(combined_data, mode="initial_grouping")

                  return common_information, reason
        return "no combined data to perform inital gathering"

    # Organizing mode: handle sequence according to the logic
    if organizing and data:
        print("organizing")

        def unwrap_to_str(value):
            # Keep unwrapping until not a tuple/list
            while isinstance(value, (list, tuple)):
                if len(value) == 0:
                    return ""   # empty list/tuple case
                value = value[0]
            return str(value)

        def is_sound_input(item):
            """Classify based on input length (short = sound/phoneme, long = image)."""
            return isinstance(item, str) and len(item) < 50

        # Decide processing mode
        if eye and not ear:
            # Image-only input (existing flow with focusing_agent)
            print("Mode: Visual only")
            result = focusing_agent(image, image1, importance_map, dual_process=True, request_input=data)
            if isinstance(result, tuple) and len(result) == 2:
                data_2, boolean_flag = result
            else:
                data_2, boolean_flag = None, result
            if not boolean_flag:
                print("Boolean flag is None. Ending function.")
                return
            combined_data = [data, data_2] if data_2 else [data]

        elif ear and not eye:
            # Sound-only input (skip focusing_agent)
            print("Mode: Sound only")
            sound_inputs = []
            sound_inputs.append(data)

            second_sound = get_next_sound_input(mode="organizing")
            if second_sound:
                sound_inputs.append(second_sound)
            else:
                #print("error in getting the second sound input")
                pass
            combined_data = sound_inputs

        elif eye and ear:
            image_inputs = []
            sound_inputs = []
            # Mixed mode: process image and sound separately
            print("Mode: Mixed input (sound + image)")
            if is_sound_input(data):
                sound_inputs.append(data)
                combined_data = data
            else:
                image_inputs.append(data)
                combined_data = data

            # Process image inputs via focusing_agent
            if len(image_inputs) > 0:
                img_result = focusing_agent(image, image1, importance_map, dual_process=True, request_input=image_inputs)
                print(f"img result - {img_result}")
                if isinstance(img_result, tuple) and len(img_result) == 2:
                    img_data_2, boolean_flag = img_result
                    image_inputs.append(img_data_2)
                else:
                    img_data_2, boolean_flag = None, img_result
                    
                if not boolean_flag:
                    print("Boolean flag is None for image part.")
                    pass
                #image_inputs.append(img_data_2)
               # image_inputs = [image_inputs[0], img_data_2] if img_data_2 else [image_inputs[0]]

                combined_data = image_inputs
            # Ensure both sound and image parts have at least two elements
            if len(sound_inputs) > 2:
                print("Awaiting second sound input...")
                second_sound = get_next_sound_input(mode="organizing")
                if second_sound:
                    sound_inputs.append(second_sound)
                else:
                    #print("error in getting the second sound input")
                    pass

                combined_data = sound_inputs

        else:
            print("No valid sensory mode detected (eye/ear flags missing).")
            return

        print(f"Combined data for processing: {combined_data}")

        # Check if combined_data already exists in ring tree
        def check_existing_order():
            for tree_name in ring_tree.surface:
                tree_data = ring_tree.surface[tree_name]
                for i in range(len(tree_data) - len(combined_data) + 1):
                    if all(dict_match(combined_data[j], tree_data[i + j]) for j in range(len(combined_data))):
                        print(f"Combined data already exists in tree: {tree_name}")
                        return True
                for branch_name, branch_info in ring_tree.branches.items():
                    if branch_info["reference_tree"] == tree_name:
                        branch_data = tree_data[:branch_info["reference_length"]] + branch_info["unique_data"]
                        for i in range(len(branch_data) - len(combined_data) + 1):
                            if all(dict_match(combined_data[j], branch_data[i + j]) for j in range(len(combined_data))):
                                print(f"Combined data already exists in branch: {branch_name}")
                                ring_tree.last_branch_match = combined_data[:]  # copy for safety
                                print(f"last_branch_match 1 - {ring_tree.last_branch_match}")
                                return True
            return False

        if check_existing_order():
            return "Data is already correctly ordered in the ring tree."
        #print(f"last_branch_match - {ring_tree.last_branch_match}")
        # If we have a branch continuation candidate
        if ring_tree.last_branch_match:
            print(f"last_branch_match - {ring_tree.last_branch_match}")
            # Update last_branch_match as per your rule
            corrected_match = [
                ring_tree.last_branch_match[-1],   # previous last
                combined_data[-1]                  # new last
            ]
            print(f"last_branch_match (after update) - {ring_tree.last_branch_match}")

            match_result = find_matching_prefix(ring_tree.last_branch_match, min_prefix=2)
            #print(f"match_result - {match_result}")
            if match_result and not match_result[3]:  # ensure match_result exists
                # Continue existing match/branch/tree creation logic
                match_result = find_matching_prefix(combined_data, min_prefix=2)
        else:
            match_result = find_matching_prefix(combined_data, min_prefix=2)

        if match_result is None:
            print("No matching prefix found.")
            return
        else:
            tree_name, start_idx, match_length, consecutive = match_result
            message = None

            if tree_name:
                # If the match was found in a branch(only if consecutive)
                if tree_name.startswith("branch_"):
                    message = extend_branch(tree_name, corrected_match, match_length)

                # If the match was found in a main tree + consecutive
                if not message and consecutive is True:
                    print(f"searching via common information")
                    # Step 1: Try searching via common information first
                    common_information, reason = searching_via_common_information(combined_data, mode="organizing")
                    message = common_information, reason

                if not message:
                    print("checking for branch creation possibility")
                    # Step 2: Check for branch creation possibility
                    should_create_branch = False
                    for node in combined_data[:2]:
                        for tree_data2 in ring_tree.surface.get(tree_name, []):
                            if dict_match(node, tree_data2):
                                should_create_branch = True
                                break
                        if should_create_branch:
                            break

                    if should_create_branch:
                        message = create_branch(tree_name, combined_data, match_length)
                        # Fallback: if branch creation failed, update the tree with new nodes
                    if not message:
                        print("update the same tree with new node")
                        message = update_tree_with_new_nodes(tree_name, data)

                return message

            else:
                print("creating new tree")
                # --- Step 1: Create a new tree ---
                new_tree_name = f"tree_{ring_tree.tree_counter}"
                ring_tree.tree_counter += 1

                if new_tree_name not in ring_tree.surface:
                    ring_tree.surface[new_tree_name] = []

                # Handle different types of data safely
                if isinstance(data, (list, dict)):
                    ring_tree.surface[new_tree_name] = data.copy()
                elif isinstance(data, str):
                    ring_tree.surface[new_tree_name] = [data]  # store as list of one string
                else:
                    ring_tree.surface[new_tree_name].append(data)  # fallback for other types

                ring_tree.access_counts[new_tree_name] = [0] * len(data)
                ring_tree.windows[new_tree_name] = 0

                # Ensure ranking dictionary exists
                if "R'''" not in ring_tree.surface_rankings:
                    ring_tree.surface_rankings["R'"] = []

                if new_tree_name not in ring_tree.surface_rankings["R'"]:
                    ring_tree.surface_rankings["R'"].append(new_tree_name)

                # Add to lowest rank if ranking is active
                if len(ring_tree.surface) >= ring_tree.min_trees_for_ranking:
                    #ring_tree.surface_rankings["R'''"][new_tree_name].append(data)
                    update_rankings()

                print(f"Created new tree {new_tree_name} with {len(data)} nodes.")
                print("now started gathering with new node")

                # --- Step 2: Progressive gathering loop ---
                if isinstance(data, list):
                    accumulated_data = data.copy()
                else:
                    accumulated_data = [data]  # ensure list always

                continue_gathering = True

                # Rule of thumb: sound inputs have short sequences
                is_sound_input = len(accumulated_data[-1]) < 50 if accumulated_data else False

                while continue_gathering:
                    print(f"Gathering after: {accumulated_data[-1] if len(accumulated_data)>0 else accumulated_data}")

                    if is_sound_input:
                        additional_data = get_next_sound_input(mode="organizing")
                        print(f"Sound additional_data: {additional_data}")
                        boolean_flag = additional_data is not None
                    else:
                        result2 = focusing_agent(
                            image, image1, importance_map,
                            dual_process=True,
                            request_input=accumulated_data[-1],
                            branch=True
                        )
                        result2 = unwrap_to_str(result2)
                        print(f"Image result2 - {result2}")

                        if isinstance(result2, tuple) and len(result2) == 2:
                            additional_data, boolean_flag = result2
                            print(f"Additional data: {additional_data}, Boolean Flag: {boolean_flag}")
                        else:
                            boolean_flag = result2
                            additional_data = None
                            print(f"Boolean Flag: {boolean_flag}")

                    if not boolean_flag or additional_data is None:
                        print("No more additional data. Ending data gathering.")
                        break

                    # --- Step 3: Check if additional_data exists in ANY existing main tree ---
                    found_in_main = False
                    for other_tree, other_data in ring_tree.surface.items():
                        for item in other_data:
                            if dict_match(item, additional_data):
                                found_in_main = True
                                break
                        if found_in_main:
                            break

                    if found_in_main:
                        print(f"Found match in existing main tree. Stopping gathering. Updating new tree {new_tree_name}.")
                        return True, "new tree updated"

                    # --- Step 4: Otherwise, append additional data into the same new tree ---
                    ring_tree.surface[new_tree_name].append(additional_data)
                    ring_tree.access_counts[new_tree_name].append(0)
                    accumulated_data.append(additional_data)

                    print(f"Appended additional data: {additional_data}")

                message = f"updated the new tree"
                return message

    # Testing mode: find and retrieve data with confirmation mechanism
    if testing and data:
        result = {"found": None, "message": "Data not found", "connected": [], "tree": None, "awaiting_confirmation": True}
        # images from the sound also act as confirmation, create imagine to hold last 3 visual outputs
        imagine_agent = ImagineAgent(max_size=3)
        imagination = Imagination()
        visual_output = []
        sound_output = []
        
        def get_next_search_index(start_index, connected_nodes, total_length):
            # Determine the next search index, ensuring it is after the connected nodes
            last_connected_index = max([start_index] + connected_nodes)
            return (last_connected_index + 1) % total_length

        # Initialize traversal state
        state = {
            "current_tree": None,
            "current_branch" : None,
            "current_index": 0,
            "visited_nodes": set(),
            "path": [],
            "matched_rings": [],
            "window_updates": []  # Track window updates during traversal
        }
        
        # First search in high-ranked trees if ranking is active
        potential_matches = []
        if len(ring_tree.surface) > 0:
            for rank in ["R'", "R''", "R'''"]:
                for tree_name in ring_tree.surface_rankings.get(rank, []):
                    tree_data = ring_tree.surface.get(tree_name, [])
                    for index, item in enumerate(tree_data):
                        if dict_match(item, data):
                            # Update access count for the matched node
                            window_update = update_access_counts(tree_name, index)
                            # Get initial connected nodes (just 2)
                            connected = []
                            for offset in range(1, 3):
                                if index + offset < len(tree_data):   # forward only, no wrap
                                    connected.append(tree_data[index + offset])

                            # Continue searching from after connected nodes
                            next_search_index = get_next_search_index(
                                index,
                                [i for i in range(index+1, min(index+3, len(tree_data)))],
                                len(tree_data)
                            )

                            potential_matches.append({
                                "found": item,
                                "message": f"Found in {tree_name} ({rank} rank) at position {index}",
                                "tree": tree_name,
                                "connected": connected,
                                "type": "tree",
                                "index": index,
                                "next_search_index": next_search_index,
                                "window_update": window_update
                            })

        # If we found potential matches, begin traversal with the first one
        if potential_matches:
            # Start with the first match
            match = potential_matches[0]
            state["current_tree"] = match["tree"]
            state["current_index"] = match["index"]
            state["matched_rings"].append(match["found"])

            state["type"] = match["type"]
            node = match["found"]
            if len(node) > 70:
                # Visual node
                visual_output.append(node)
            else:
                # Sound node
                sound_output.append(node)

            if "window_update" in match and match["window_update"]:
                state["window_updates"].append(match["window_update"])
            print(f"potential match - {potential_matches}")

            while True:
                # Get current tree data
                if match["type"] == "branch":
                        current_tree_data = ring_tree.branches.get(state["current_tree"], [])
                else:
                        current_tree_data = ring_tree.surface.get(state["current_tree"], [])

                if not current_tree_data:
                    result = {
                        "status": "error",
                        "message": f"Tree {state['current_tree']} is empty",
                        "matched_rings": state["matched_rings"],
                        "path": state["path"],
                        "window_updates": state["window_updates"]
                    }
                    break

                # Before the loop, make sure current_node is a list
                current_node = []
                # Check if index is valid before accessing
                if 0 <= state["current_index"] < len(current_tree_data):
                    current_node.append(current_tree_data[state["current_index"]])
                   # print(f"current_node - {current_node}")
                else:
                    print(f"Warning: current_index {state['current_index']} out of range for tree of size {len(current_tree_data)}")
                    #return "current index out of range"
                    break

                current_node_0 = str(current_node[0])
                current_node_0 = current_node_0.strip("()[],'\" ")
                input_is_sound = isinstance(current_node_0, str) and len(current_node_0) < 70

                def get_focus_ids():
                    if input_is_sound:
                        val = get_next_sound_input(mode="testing")
                        return [val] if not isinstance(val, list) else val
                    else:
                        return None

                # Store the matched node in ImagineAgent (unchanged)
                node_value = current_node[-1] if current_node else None

                #imagine_agent.store(node_value)
                data_2 = None
                # First confirm with ImagineAgent(using imagination)
                result = imagine_agent.confirm(node_value, branch=False)

                if isinstance(result, tuple) and len(result) == 2:
                    data_2, boolean_flag = result
                else:
                    boolean_flag = result

                # --- Step 2: If imagine_agent fails OR node_value == data_2 ---(using environment inputs)
                if (not boolean_flag) or (node_value == data_2):
                    if input_is_sound:
                        focus_ids = get_focus_ids()

                        if focus_ids is not None and len(focus_ids) > 0:  # --- If we got something from get_focus_ids ---
                            #print(f"Using focus_ids: {focus_ids}")
                            data_2 = focus_ids
                            boolean_flag = True
                    else:
                        if image is not None:
                            # --- Step 3: fallback to focusing_agent ---
                            data_2 = focusing_agent(
                                image, image1, importance_map,
                                dual_process=True,
                                request_input=node_value
                            )
                            if isinstance(data_2, tuple) and len(data_2) == 2:
                                data_2, boolean_flag = data_2
                            else:
                                boolean_flag = data_2
                        else:
                            boolean_flag = False
                            
                if not boolean_flag:
                        result = {
                            "status": "error",
                            "message": f"node confirmation failed",
                            "matched_rings": state["matched_rings"],
                            "path": state["path"],
                            "window_updates": state["window_updates"]
                        }
                        break
                
                # Normalize to list of items
                if not isinstance(data_2, list):
                    data_2 = [data_2]

                current_node.extend(data_2) 
                #print(f"Confirmed nodes: {current_node}")

                for node in current_node:
                    if node is None:
                        break

                    if len(node) > 70:
                        # Visual node
                        visual_output.append(node)
                    else:
                        # Sound node
                        sound_output.append(node)

                node_key = (state["current_tree"], state["current_index"])

                # Avoid infinite loops by tracking visited nodes
                if node_key in state["visited_nodes"]:
                    result = {
                        "status": "complete",
                        "message": "Traversal complete - returned to previously visited node",
                        "matched_rings": state["matched_rings"],
                        "path": state["path"],
                        "window_updates": state["window_updates"]
                    }
                    break

                state["visited_nodes"].add(node_key)
                state["path"].append({"tree": state["current_tree"], "index": state["current_index"], "data": current_node})

                # Update access count for this node
                window_update = update_access_counts(state["current_tree"], state["current_index"])
                if window_update:
                    state["window_updates"].append(window_update)

                #to give data to the retrieval program.
                # Check if there's a branch connection at this point
                branch_connection = None
                # --- Branch connections only consecutive---
                #print("branch connection checking")
                for conn_name, conn_data in ring_tree.branch_connections.items():
                    if conn_data["parent_tree"] == match["tree"]:
                        branch_tree_data = conn_data["unique_data"]

                        # Track consecutive matches
                        consecutive_matches = []
                        start_index = None

                        for index, item in enumerate(branch_tree_data):
                            for node_idx, node in enumerate(current_node):

                                if dict_match(item, node):
                                    consecutive_matches = [item]
                                    start_index = index

                                    # Compare both lists step by step
                                    next_idx = index + 1
                                    next_node_idx = node_idx + 1

                                    while (next_idx < len(branch_tree_data) and
                                        next_node_idx < len(current_node) and
                                        dict_match(branch_tree_data[next_idx], current_node[next_node_idx])):
                                        consecutive_matches.append(branch_tree_data[next_idx])
                                        next_idx += 1
                                        next_node_idx += 1

                                    if len(consecutive_matches) > 1:
                                        #print(f"Consecutive match found in branch: {consecutive_matches}")
                                        branch_connection = {
                                            "name": conn_name,
                                            "data": conn_data,
                                            "matches": consecutive_matches,
                                            "start_index": start_index
                                        }
                                    break

                # --- Cross-branch connections only if match found at transition nodes (only if not already found) ---
                cross_branch_connection = None
                if branch_connection is None:
                   # print("cross branch connection checking")
                    for conn_name, conn_data in ring_tree.cross_branch_connections.items():
                        donor = conn_data["trees"]["donor"]
                        receiver = conn_data["trees"]["receiver"]

                        donor_tree = donor["name"]
                        donor_nodes = ring_tree.surface.get(donor_tree, [])[donor["start_index"]: donor["end_index"] + 1]

                        receiver_tree = receiver["name"]
                        receiver_nodes = ring_tree.surface.get(receiver_tree, [])[receiver["start_index"]: receiver["end_index"] + 1]

                        # --- Condition 1: receiver end ↔ donor start ---
                        if receiver_nodes and donor_nodes:
                            for i, node in enumerate(current_node):
                            # print(f"node 1 - {node}")
                            # print(f"receiver_nodes[-1] - {receiver_nodes[-1]}")
                            # print(f"donor_nodes[0] - {donor_nodes[0]}")

                                # ensure we have a "next" node to compare against
                                if i + 1 < len(current_node):
                                    next_node = current_node[i + 1]
                                #  print(f"next_node - {next_node}")

                                    if dict_match(receiver_nodes[-1], node) and dict_match(donor_nodes[0], next_node):
                                       # print("Match found at receiver end → donor start cross-branch")
                                        cross_branch_connection = {"name": conn_name, **conn_data}
                                        break

                        # --- Condition 2: donor end ↔ receiver start ---
                        if donor_nodes and receiver_nodes:
                            for i, node in enumerate(current_node):
                                #print(f"node 2 - {node}")
                                #print(f"donor_nodes[-1] - {donor_nodes[-1]}")
                                #print(f"receiver_nodes[0] - {receiver_nodes[0]}")

                                # ensure we have a "next" node to compare against
                                if i + 1 < len(current_node):
                                    next_node = current_node[i + 1]
                                # print(f"next_node - {next_node}")

                                    if dict_match(donor_nodes[-1], node) and dict_match(receiver_nodes[0], next_node):
                                       # print("Match found at donor end → receiver start cross-branch")
                                        cross_branch_connection = {"name": conn_name, **conn_data}
                                        break

                if branch_connection:
                    print(f"branch connection - {branch_connection}")
                    branch_tree = branch_connection["name"]
                    branch_start_index = branch_connection["data"]["branch_point_index"]
                    parent_tree = branch_connection["data"]["parent_tree"]
                    ring_tree.access_counts_branch[branch_tree] +=1

                    # Get branch nodes
                    branch_tree_ = ring_tree.branches.get(branch_tree, [])
                    branch_tree_data = branch_tree_["unique_data"]
                    if not branch_tree_data:
                        result = {
                            "status": "error",
                            "message": f"Branch tree {branch_tree_data} is empty",
                            "matched_rings": state["matched_rings"],
                            "path": state["path"],
                            "window_updates": state["window_updates"]
                        }
                        break

                    state["current_tree"] = branch_tree
                    state["current_index"] = branch_start_index

                    while True:
                        if state["current_index"] + 1 < len(branch_tree_data):
                            next_index = state["current_index"] + 1
                            branch_node = branch_tree_data[next_index]
                        else:
                            print("Reached end of branch_tree_data, stopping traversal.")
                            branch_node = None 
                            break

                        imagine_agent.store(branch_node)

                        # First confirm with ImagineAgent
                        result = imagine_agent.confirm(branch_node, branch=False)
                        if isinstance(result, tuple) and len(result) == 2:
                            data_2, boolean_flag = result
                        else:
                            boolean_flag = result

                        if not boolean_flag:
                            # Try focusing agent if ImagineAgent fails
                            data_2 = focusing_agent(
                                image, image1, importance_map,
                                dual_process=True,
                                requested_input=branch_node
                            )
                            if isinstance(data_2, tuple) and len(data_2) == 2:
                                data_2, boolean_flag = data_2
                            else:
                                boolean_flag = result

                            if not boolean_flag:
                                result = {
                                    "status": "error",
                                    "message": f"Branch node confirmation failed with focusing agent",
                                    "matched_rings": state["matched_rings"],
                                    "path": state["path"],
                                    "window_updates": state["window_updates"]
                                }
                                break

                        branch_node = data_2
                        # Mark branch node as visited
                        state["visited_nodes"].add((branch_tree, next_index))
                        state["path"].append({"tree": branch_tree, "index": next_index, "data": branch_node})
                        state["matched_rings"].append(branch_node)
                       # print(f"Confirmed branch node: {branch_node}")

                        if len(branch_node) > 70:  # Visual node
                            visual_output.append(branch_node)
                        else:  # Sound node
                            sound_output.append(branch_node)

                        # Update access counts
                        window_update = update_access_counts(branch_tree, next_index)
                        if window_update:
                            state["window_updates"].append(window_update)

                        # Move traversal forward
                        state["current_index"] = next_index

                        # --- End condition: return to main tree ---
                        if (branch_connection["data"]["returns_to_tree"] and
                            next_index == branch_connection["data"]["return_from_branch_idx"]):
                            print("Reached end of branch, returning to main tree")
                            state["current_tree"] = branch_connection["data"]["parent_tree"]
                            state["current_index"] = branch_connection["data"]["branch_point_index"]
                            branch_connection = False  # reset flag
                            break

                elif cross_branch_connection:
                    print(f"cross branch connection - {cross_branch_connection}")
                    # Process cross-branch connection, cross connection tree is just like the extension of the main node tree

                    """
                    Traverse a cross-branch connection tree as if it's an extension of the main tree.
                    Continues until the reverse link node back to the main tree is reached.
                    """
                    ring_tree.access_counts_branch[cross_branch_connection["name"]] +=1

                    donor = cross_branch_connection["trees"]["donor"]
                    receiver = cross_branch_connection["trees"]["receiver"]

                    donor_nodes = ring_tree.surface.get(donor["name"], [])[donor["start_index"]: donor["end_index"] + 1]
                    receiver_nodes = ring_tree.surface.get(receiver["name"], [])[receiver["start_index"]: receiver["end_index"] + 1]

                    #print(f"Entering cross-branch traversal from {donor['name']}[{donor['start_index']}→{donor['end_index']}] "
                      #  f"to {receiver['name']}[{receiver['start_index']}→{receiver['end_index']}]")
                    
                    # --- Step 1: Traverse receiver nodes ---
                    for i, node in enumerate(receiver_nodes):
                        # 🔹 Update state before processing
                        state["current_tree"] = receiver["name"]
                        state["current_index"] = receiver["start_index"] + i

                        imagine_agent.store(node)

                        # Confirm node
                        result = imagine_agent.confirm(node, branch=False)
                        if isinstance(result, tuple) and len(result) == 2:
                            data_2, boolean_flag = result
                        else:
                            boolean_flag = result

                        if not boolean_flag:
                            result = focusing_agent(
                                image, image1, importance_map,
                                dual_process=True,
                                request_input=node
                            )
                            if isinstance(result, tuple) and len(result) == 2:
                                data_2, boolean_flag = result
                            else:
                                boolean_flag = result

                        if not boolean_flag:
                            result = {
                                "status": "error",
                                "message": "Focusing agent failed in receiver traversal",
                                "error_location": {"tree": state["current_tree"], "index": state["current_index"]},
                                "matched_rings": state["matched_rings"],
                                "path": state["path"],
                                "window_updates": state["window_updates"]
                            }
                            break

                        state["matched_rings"].append(node)
                        #print(f"Confirmed receiver node: {node}")

                        # Handle output
                        if len(node) > 70:  # visual
                            visual_output.append(node)
                        else:  # sound
                            sound_output.append(node)

                        # 🔹 Update window counts dynamically
                        window_update = update_access_counts(state["current_tree"], state["current_index"])
                        if window_update:
                            state["window_updates"].append(window_update)

                    # --- Step 2: Traverse donor nodes ---
                    for i, node in enumerate(donor_nodes):
                        # 🔹 Update state before processing
                        state["current_tree"] = donor["name"]
                        state["current_index"] = donor["start_index"] + i

                        imagine_agent.store(node)

                        result = imagine_agent.confirm(node, branch=False)
                        if isinstance(result, tuple) and len(result) == 2:
                            data_2, boolean_flag = result
                        else:
                            boolean_flag = result

                        if not boolean_flag:
                            result = focusing_agent(
                                image, image1, importance_map,
                                dual_process=True,
                                request_input=node
                            )
                            if isinstance(result, tuple) and len(result) == 2:
                                data_2, boolean_flag = result
                            else:
                                boolean_flag = result

                        if not boolean_flag:
                            result = {
                                "status": "error",
                                "message": "Focusing agent failed in donor traversal",
                                "error_location": {"tree": state["current_tree"], "index": state["current_index"]},
                                "matched_rings": state["matched_rings"],
                                "path": state["path"],
                                "window_updates": state["window_updates"]
                            }
                            break
                        state["matched_rings"].append(node)
                    # print(f"Confirmed donor node: {node}")

                        # Handle output
                        if len(node) > 70:  # visual
                            visual_output.append(node)
                        else:  # sound
                            sound_output.append(node)

                        # 🔹 Update window counts dynamically
                        window_update = update_access_counts(state["current_tree"], state["current_index"])
                        if window_update:
                            state["window_updates"].append(window_update)

                    print("Completed cross-branch traversal, returning to main tree.")

                else:
                    print("no branch/cross branch connection")
                    branch = False
                    while True:
                        # No branch, traverse to next node in current tree. no confirmation from focusing agent
                        if state["current_index"] + 1 < len(current_tree_data):
                            next_index = state["current_index"] + 1
                            next_node = current_tree_data[next_index]
                        else:
                            print("Reached end of current_tree_data, stopping traversal.")
                            next_node = None  # or handle gracefully (e.g., break, return, etc.)
                            break

                        # Verify with imagine agent and then to focusing agent
                        imagine_agent.store(next_node)

                        result = imagine_agent.confirm(next_node, branch)

                        if isinstance(result, tuple) and len(result) == 2:
                            data_2, boolean_flag = result
                            if boolean_flag:
                                pass
                                #print("Confirmed via ImagineAgent (no focusing needed)")
                        else:
                            boolean_flag = result
                        # print(f"Boolean Flag: {boolean_flag}")

                        # If not confirmed, fall back to focusing agent
                        if not boolean_flag:
                            result = focusing_agent(
                                image, image1, importance_map,
                                dual_process=True,
                                request_input=next_node
                            )

                            if isinstance(result, tuple) and len(result) == 2:
                                data_2, boolean_flag = result
                            else:
                                boolean_flag = result

                        if boolean_flag is None:
                            result = {
                                "status": "error",
                                "message": f"Focusing agent did not confirm next node in sequence",
                                "error_location": {"tree": state["current_tree"], "index": next_index},
                                "matched_rings": state["matched_rings"],
                                "path": state["path"],
                                "window_updates": state["window_updates"]
                            }
                            break

                        # Add confirmed node and update state
                        state["current_index"] = next_index
                        state["matched_rings"].append(next_node)
                        #print(f"Confirmed node branch: {next_node}")  # Add this line to see the output

                        if len(next_node) > 70:
                            # Visual node
                            visual_output.append(next_node)
                        else:
                            # Sound node
                            sound_output.append(next_node) 

                        # Update access count for this node
                        window_update = update_access_counts(state["current_tree"], next_index)
                        if window_update:
                            state["window_updates"].append(window_update)

                if len(state["matched_rings"]) > 1:
                    branch = False
                    # Determine how many nodes to output next based on consecutive confirmations
                    confirmation_count = len(state["matched_rings"])
                   # print(f"Confirmation count: {confirmation_count}")
                   # print(f"Matched rings: {state['matched_rings']}")

                    final_tree_name = state["current_tree"]
                    final_tree_index = state["current_index"]

                    if final_tree_name.startswith("branch_"):
                            final_tree_data = ring_tree.branches.get(state["current_tree"], [])
                    else:
                            final_tree_data = ring_tree.surface.get(state["current_tree"], [])

                    final_tree_data[final_tree_index:]
                   # print(f"final_tree_data - {final_tree_data}")

                    if confirmation_count >= 6:
                        nodes_to_output = 4  # Output 4 nodes after 4+ confirmations
                    elif confirmation_count <= 4:
                        nodes_to_output = 2  # Output 2 nodes after 2-3 confirmations
                    else:
                        nodes_to_output = 1  # Default to 1 node (shouldn't reach here due to condition)

                    # Cap at 6 nodes maximum
                    nodes_to_output = min(nodes_to_output, 6)

                    # --- Step 2: Confirm branch/cross-branch at this node to give extra output---
                    # Check if there's a branch connection at this point
                    branch_connection = None
                    # --- Branch connections only consecutive---
                   # print("branch connection checking")
                    for conn_name, conn_data in ring_tree.branch_connections.items():
                        if conn_data["parent_tree"] == match["tree"]:
                            branch_tree_data = conn_data["unique_data"]

                            # Track consecutive matches
                            consecutive_matches = []
                            start_index = None

                            for index, item in enumerate(branch_tree_data):
                                for node_idx, node in enumerate(final_tree_data):

                                    if dict_match(item, node):
                                        consecutive_matches = [item]
                                        start_index = index

                                        # Compare both lists step by step
                                        next_idx = index + 1
                                        next_node_idx = node_idx + 1

                                        while (next_idx < len(branch_tree_data) and
                                                next_node_idx < len(final_tree_data) and
                                                dict_match(branch_tree_data[next_idx], final_tree_data[next_node_idx])):
                                            consecutive_matches.append(branch_tree_data[next_idx])
                                            next_idx += 1
                                            next_node_idx += 1

                                        if len(consecutive_matches) > 1:
                                            print(f"Consecutive match found in branch: {consecutive_matches}")
                                            branch_connection = {
                                                "name": conn_name,
                                                "data": conn_data,
                                                "matches": consecutive_matches,
                                                "start_index": start_index
                                            }
                                        break

                    # --- Cross-branch connections only if match found at transition nodes (only if not already found) ---
                    cross_branch_connection = None
                    if branch_connection is None:
                        #print("cross branch connection checking")
                        for conn_name, conn_data in ring_tree.cross_branch_connections.items():
                            donor = conn_data["trees"]["donor"]
                            receiver = conn_data["trees"]["receiver"]

                            donor_tree = donor["name"]
                            donor_nodes = ring_tree.surface.get(donor_tree, [])[donor["start_index"]: donor["end_index"] + 1]

                            receiver_tree = receiver["name"]
                            receiver_nodes = ring_tree.surface.get(receiver_tree, [])[receiver["start_index"]: receiver["end_index"] + 1]

                            # --- Condition 1: receiver end ↔ donor start ---
                            if receiver_nodes and donor_nodes:
                                for i, node in enumerate(final_tree_data):
                                # print(f"node 1 - {node}")
                                # print(f"receiver_nodes[-1] - {receiver_nodes[-1]}")
                                # print(f"donor_nodes[0] - {donor_nodes[0]}")

                                    # ensure we have a "next" node to compare against
                                    if i + 1 < len(final_tree_data):
                                        next_node = final_tree_data[i + 1]
                                        #print(f"next_node - {next_node}")

                                        if dict_match(receiver_nodes[-1], node) and dict_match(donor_nodes[0], next_node):
                                            print("Match found at receiver end → donor start cross-branch")
                                            cross_branch_connection = {"name": conn_name, **conn_data}
                                            break

                            # --- Condition 2: donor end ↔ receiver start ---
                            if donor_nodes and receiver_nodes:
                                for i, node in enumerate(final_tree_data):
                                # print(f"node 2 - {node}")
                                # print(f"donor_nodes[-1] - {donor_nodes[-1]}")
                                # print(f"receiver_nodes[0] - {receiver_nodes[0]}")

                                    # ensure we have a "next" node to compare against
                                    if i + 1 < len(final_tree_data):
                                        next_node = final_tree_data[i + 1]
                                    # print(f"next_node - {next_node}")

                                        if dict_match(donor_nodes[-1], node) and dict_match(receiver_nodes[0], next_node):
                                            print("Match found at donor end → receiver start cross-branch")
                                            cross_branch_connection = {"name": conn_name, **conn_data}
                                            break
                            
                    # Decide which tree to pull next nodes from
                    if branch_connection:
                        branch_tree = branch_connection["data"]["unique_data"]
                        branch_start_index = branch_connection["data"]["start_index"]
                        next_tree_data = branch_tree
                    elif cross_branch_connection:
                        donor = cross_branch_connection["trees"]["donor"]
                        receiver = cross_branch_connection["trees"]["receiver"]

                        donor_nodes = ring_tree.surface.get(donor["name"], [])[donor["start_index"]: donor["end_index"] + 1]
                        receiver_nodes = ring_tree.surface.get(receiver["name"], [])[receiver["start_index"]: receiver["end_index"] + 1]

                        # check whether i is inside donor or receiver range
                        if donor["start_index"] <= final_tree_index <= donor["end_index"]:
                            # i belongs to donor
                            #branch_tree = receiver["name"]  # donor should connect to receiver side
                            next_tree_data = receiver_nodes

                        elif receiver["start_index"] <= final_tree_index <= receiver["end_index"]:
                            # i belongs to receiver
                            # branch_tree = donor["name"]  # receiver should connect to donor side
                            next_tree_data = donor_nodes

                        else:
                            # i not inside donor/receiver ranges — ignore
                            continue

                    else:
                        print("staying in main node")
                        # Stay in main node flow, but process nodes ahead of the current index
                    # print(f"current index - {state['current_index']} and len of tree data - {len(final_tree_data)}")
                        if state["current_index"] < len(final_tree_data) - 1:
                            # Slice from the next index onward
                            next_tree_data = final_tree_data
                        else:
                            # No more nodes left
                            next_tree_data = []

                    if len(next_tree_data) != 0:
                        should_output = False
                        # --- Step 3: Output next N nodes from chosen path ---
                        print(f"nodes_to_output - {nodes_to_output}")
                        for offset in range(1, nodes_to_output + 1):
                            if branch_connection:
                                next_index = branch_start_index + offset
                            if cross_branch_connection:
                                # Determine if we are in donor or receiver
                                if donor["start_index"] <= final_tree_index <= donor["end_index"]:
                                    # Currently in donor, so offset from donor start
                                    next_index = donor["start_index"] + offset
                                elif receiver["start_index"] <= final_tree_index <= receiver["end_index"]:
                                    # Currently in receiver, so offset from receiver start
                                    next_index = receiver["start_index"] + offset
                            else:
                                # Fallback to main tree index if not in either (shouldn't happen)
                                next_index = final_tree_index + offset
                            #print(f"next tree data - {next_tree_data}")
                            #print(f"next index - {next_index} and len of final data - {len(next_tree_data)}")
                            if next_index >= len(next_tree_data):
                                print("Reached end of available nodes, stopping output.")
                                # Proper result with reason
                                result = {
                                    "status": "stopped",
                                    "reason": "Reached end of available nodes while outputting next path nodes",
                                    "matched_rings": [],
                                    "path": [],
                                    "window_updates": []
                                }
                                break

                            next_node = next_tree_data[next_index]
                            #print(f"Node {offset} after confirmed node")

                            imagine_agent.store(next_node)

                            if len(next_node) > 70:
                                visual_output.append(next_node)
                            else:
                                # Sound node
                                sound_output.append(next_node)

                        # --- Step 4: Update traversal index safely ---
                        next_node_index = final_tree_index + nodes_to_output + 1
                        if next_node_index >= len(next_tree_data):
                            # Handle boundary if in branch/cross-branch
                            if branch_connection and branch_connection["data"]["returns_to_tree"]:
                                state["current_tree"] = branch_connection["data"]["parent_tree"]
                                state["current_index"] = branch_connection["data"]["return_from_branch_idx"]
                            elif cross_branch_connection:
                                state["current_tree"] = cross_branch_connection["data"]["parent_tree"]
                                state["current_index"] = cross_branch_connection["data"]["branch_point_index"]
                            else:
                                # Stay at last node (don’t wrap around)
                                state["current_index"] = len(next_tree_data) - 1
                        else:
                            state["current_index"] = next_node_index

                        # --- Step 5: Update access count ---
                        window_update = update_access_counts(state["current_tree"], i)
                        if window_update:
                            state["window_updates"].append(window_update)

                        print(f"len of visual output - {len(visual_output)} and len of sound output - {len(sound_output)}")                               
                        if sound_output:
                            print("Processing sound output through imagination")
                            final_output, should_output = imagination.process_output_node(node_data=sound_output)
                            if should_output:
                                decoded_move = []
                                decoded_words = []
                                for encoded_id in final_output:
                                    result = id_to_word_sound(encoded_id, original_length=10)
                                    store_json(result["id"])
                                    decoded_words.append(result["word"])
                                    print(result)
                                for encoded_id in final_output:
                                    result = id_to_word_move(encoded_id, original_length=10)
                                    decoded_move.append(result["word"])
                                    if result["word"] != "[Unknown]":
                                        save_move_words(result["word"])

                                if decoded_words is not None and len(decoded_words) > 0:
                                    save_sound_words(decoded_words)

                                reprocess_nodes = imagination.get_nodes_for_testing()
                                def flatten_list(lst):
                                    """Flatten a list of lists into a single list of strings"""
                                    flat = []
                                    for item in lst:
                                        if isinstance(item, list):
                                            flat.extend(flatten_list(item))
                                        else:
                                            flat.append(str(item))
                                    return flat
                                
                                reprocess_nodes = flatten_list(reprocess_nodes)
                                final_output = flatten_list(final_output)
                                matching_ratio = SequenceMatcher(None, final_output, reprocess_nodes).ratio()
                                print(f"matching_ratio - {matching_ratio}")
                                if matching_ratio < 0.45:
                                  #  image = load_latest_image()
                                    for inputs in reprocess_nodes:
                                        print("Processing imagination input in testing phase")
                                        result = ring_tree(inputs, testing=True, ear=True, image=image, importance_map=importance_map)
                                        print(result)
                                        print("forming inter-connection")
                                        # If this imagination input activates nodes in ring_tree,
                                        # create cross-connection between imagination node and activated node 
                                        create_cross_connection_for_imagination(reprocess_nodes, final_output)
                                else:
                                    pass
                        else:
                            print("no audio output was generated")

                        if visual_output:
                            print("processing visual outputs to imagination")
                            # Process through imagination first
                            final_output, should_output = Imagination.process_output_node(node_data=visual_output)
                            
                            if should_output:
                                visual_output_ = []
                                for outputs in final_output:
                                    store_json(outputs)
                                    output, shape = id_to_list_(outputs)
                                    visual_output_.append(output)
                                visualize_processed_pixels(visual_output_)

                                reprocess_nodes = imagination.get_nodes_for_testing()
                                def flatten_list(lst):
                                    """Flatten a list of lists into a single list of strings"""
                                    flat = []
                                    for item in lst:
                                        if isinstance(item, list):
                                            flat.extend(flatten_list(item))
                                        else:
                                            flat.append(str(item))
                                    return flat
                                
                                reprocess_nodes = flatten_list(reprocess_nodes)
                                final_output = flatten_list(final_output)
                                matching_ratio = SequenceMatcher(None, final_output, reprocess_nodes).ratio()
                                print(f"matching_ratio - {matching_ratio}")
                                if matching_ratio < 0.45:
                                   # image = load_latest_image()
                                    for inputs in reprocess_nodes:
                                        print("Processing imagination input in testing phase")
                                        result = ring_tree(inputs, testing=True, eye=True, image=image, importance_map=importance_map)
                                        print(result)
                                        print("forming inter-connection")
                                        # If this imagination input activates nodes in ring_tree,
                                        # create cross-connection between imagination node and activated node
                                        #once we the execution comes here, the final output will be updated with the outputs of previous inputs from reprocess nodes
                                        create_cross_connection_for_imagination(reprocess_nodes, final_output)
                                else:
                                    pass
                        else:
                            print("no visual output generated")

                    else:
                        print("not going for extra node output")
                        print(f"len of visual output - {len(visual_output)} and len of sound output - {len(sound_output)}")                               
                        if sound_output:
                            print("Processing sound output through imagination")
                            final_output, should_output = imagination.process_output_node(node_data=sound_output)
                            if should_output:
                                decoded_move = []
                                decoded_words = []
                                for encoded_id in final_output:
                                    result = id_to_word_sound(encoded_id, original_length=10)
                                    store_json(result["id"])
                                    decoded_words.append(result["word"])
                                    print(result)
                                for encoded_id in final_output:
                                    result = id_to_word_move(encoded_id, original_length=10)
                                    decoded_move.append(result["word"])
                                    if result["word"] != "[Unknown]":
                                        save_move_words(result["word"])

                                if decoded_words is not None and len(decoded_words) > 0:
                                    save_sound_words(decoded_words)

                                reprocess_nodes = imagination.get_nodes_for_testing()
                                def flatten_list(lst):
                                    """Flatten a list of lists into a single list of strings"""
                                    flat = []
                                    for item in lst:
                                        if isinstance(item, list):
                                            flat.extend(flatten_list(item))
                                        else:
                                            flat.append(str(item))
                                    return flat
                                
                                reprocess_nodes = flatten_list(reprocess_nodes)
                                final_output = flatten_list(final_output)
                                matching_ratio = SequenceMatcher(None, final_output, reprocess_nodes).ratio()
                                print(f"matching_ratio - {matching_ratio}")
                                if matching_ratio < 0.45:
                                  #  image = load_latest_image()
                                    for inputs in reprocess_nodes:
                                        print("Processing imagination input in testing phase")
                                        result = ring_tree(inputs, testing=True, ear=True, image=image, importance_map=importance_map)
                                        print(result)
                                        print("forming inter-connection")
                                        # If this imagination input activates nodes in ring_tree,
                                        # create cross-connection between imagination node and activated node 
                                        create_cross_connection_for_imagination(reprocess_nodes, final_output)
                                else:
                                    pass
                        else:
                            print("no audio output was generated")

                        if visual_output:
                            print("processing visual outputs to imagination")
                            # Process through imagination first
                            final_output, should_output = imagination.process_output_node(node_data=visual_output)
                            
                            if should_output:
                                visual_output_ = []
                                for outputs in final_output:
                                    store_json(outputs)
                                    output, shape = id_to_list_(outputs)
                                    visual_output_.append(output) 
                                visualize_processed_pixels(visual_output_)

                                reprocess_nodes = imagination.get_nodes_for_testing()
                                def flatten_list(lst):
                                    """Flatten a list of lists into a single list of strings"""
                                    flat = []
                                    for item in lst:
                                        if isinstance(item, list):
                                            flat.extend(flatten_list(item))
                                        else:
                                            flat.append(str(item))
                                    return flat
                                
                                reprocess_nodes = flatten_list(reprocess_nodes)
                                final_output = flatten_list(final_output)
                                matching_ratio = SequenceMatcher(None, final_output, reprocess_nodes).ratio()
                                print(f"matching_ratio - {matching_ratio}")
                                if matching_ratio < 0.45:
                                    #image = load_latest_image()
                                    for inputs in reprocess_nodes:
                                        print("Processing imagination input in testing phase")
                                        result = ring_tree(inputs, testing=True, eye=True, image=image, importance_map=importance_map)
                                        print(result)
                                        print("forming inter-connection")
                                        # If this imagination input activates nodes in ring_tree,
                                        # create cross-connection between imagination node and activated node
                                        #once we the execution comes here, the final output will be updated with the outputs of previous inputs from reprocess nodes
                                        create_cross_connection_for_imagination(reprocess_nodes, final_output)
                                else:
                                    pass
                        else:
                            print("no visual output generated")
                            
        print(f"len of visual output - {len(visual_output)} and len of sound output - {len(sound_output)}")                               
        if len(visual_output) > 0 or len(sound_output) > 0:
            print("Exception triggered")
            if sound_output:
                print("Processing sound output through imagination")
                final_output, should_output = imagination.process_output_node(node_data=sound_output)
                if should_output:
                    decoded_move = []
                    decoded_words = []
                    for encoded_id in final_output:
                        result = id_to_word_sound(encoded_id, original_length=10)
                        store_json(result["id"])
                        decoded_words.append(result["word"])
                        print(result)
                    for encoded_id in final_output:
                        result = id_to_word_move(encoded_id, original_length=10)
                        decoded_move.append(result["word"])
                        if result["word"] != "[Unknown]":
                            save_move_words(result["word"])

                    if decoded_words is not None and len(decoded_words) > 0:
                        save_sound_words(decoded_words)

                    reprocess_nodes = imagination.get_nodes_for_testing()
                    def flatten_list(lst):
                        """Flatten a list of lists into a single list of strings"""
                        flat = []
                        for item in lst:
                            if isinstance(item, list):
                                flat.extend(flatten_list(item))
                            else:
                                flat.append(str(item))
                        return flat
                    
                    reprocess_nodes = flatten_list(reprocess_nodes)
                    final_output = flatten_list(final_output)
                    matching_ratio = SequenceMatcher(None, final_output, reprocess_nodes).ratio()
                    print(f"matching_ratio - {matching_ratio}")
                    if matching_ratio < 0.45:
                       # image = load_latest_image()
                        for inputs in reprocess_nodes:
                            print("Processing imagination input in testing phase")
                            result = ring_tree(inputs, testing=True, ear=True, image=image, importance_map=importance_map)
                            print(result)
                            print("forming inter-connection")
                            # If this imagination input activates nodes in ring_tree,
                            # create cross-connection between imagination node and activated node 
                            create_cross_connection_for_imagination(reprocess_nodes, final_output)
                    else:
                        pass
            else:
                print("no audio output was generated")

            if visual_output:
                print("processing visual outputs to imagination")
                # Process through imagination first
                final_output, should_output = imagination.process_output_node(node_data=visual_output)
                
                if should_output:
                    visual_output_ = []
                    for outputs in final_output:
                        store_json(outputs)
                        output, shape = id_to_list_(outputs)
                        visual_output_.append(output) 
                    visualize_processed_pixels(visual_output_)

                    reprocess_nodes = imagination.get_nodes_for_testing()
                    def flatten_list(lst):
                        """Flatten a list of lists into a single list of strings"""
                        flat = []
                        for item in lst:
                            if isinstance(item, list):
                                flat.extend(flatten_list(item))
                            else:
                                flat.append(str(item))
                        return flat
                    
                    reprocess_nodes = flatten_list(reprocess_nodes)
                    final_output = flatten_list(final_output)
                    matching_ratio = SequenceMatcher(None, final_output, reprocess_nodes).ratio()
                    print(f"matching_ratio - {matching_ratio}")
                    if matching_ratio < 0.45:
                        #image = load_latest_image()
                        for inputs in reprocess_nodes:
                            print("Processing imagination input in testing phase")
                            result = ring_tree(inputs, testing=True, eye=True, image=image, importance_map=importance_map)
                            print(result)
                            print("forming inter-connection")
                            # If this imagination input activates nodes in ring_tree,
                            # create cross-connection between imagination node and activated node
                            #once we the execution comes here, the final output will be updated with the outputs of previous inputs from reprocess nodes
                            create_cross_connection_for_imagination(reprocess_nodes, final_output)
                    else:
                        pass                             
                    return "no new node to process"
            else:
                print("no visual output was generated")
                
        else:
            result = {"status": "error", "message": "Data not found", "matched_rings": [], "path": [], "window_updates": []}

        return result

    if destabilizing_mechanism:
          cleanup_low_usage_connections()

    # Print the current state if no specific operation is requested
    if not any([training, organizing, testing]):
        return {
            "surface": ring_tree.surface,
            "surface_rankings": ring_tree.surface_rankings,
            "access_counts": ring_tree.access_counts,
            "access_counts_branch": ring_tree.access_counts_branch,
            "windows": ring_tree.windows,
            "branches": ring_tree.branches,
            "branch_connections": ring_tree.branch_connections
        }
    
def store_json(data: dict):
    """
    Store dictionary data as a JSON file with an auto-incrementing counter.
    Tracks the counter using 'file_counter.txt' inside the given directory.
    
    Args:
        data (dict): The data to store as JSON.
        directory (str): Folder path where the files will be saved.
        base_name (str): Base name for the stored JSON files.
    
    Returns:
        str: Full path of the stored JSON file.
    """
    directory = r"D:\artist\brainX\CRX\Properties\s1_outputs"
    os.makedirs(directory, exist_ok=True)
    counter_file = os.path.join(directory, "file_counter.txt")

    # Read or initialize counter
    if os.path.exists(counter_file):
        try:
            with open(counter_file, "r") as f:
                last_idx = int(f.read().strip())
        except ValueError:
            last_idx = 0
    else:
        last_idx = 0

    # Increment counter
    new_idx = last_idx + 1
    json_filename = f"s1_{new_idx:03d}.json"
    json_path = os.path.join(directory, json_filename)

    # Save JSON data
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Update counter file
    with open(counter_file, "w") as f:
        f.write(str(new_idx))

    print(f"[STORE] Saved JSON: {json_path}")

def load_spot_ids_from_drive_with_waiting(mode="training", wait_interval=1, max_wait_time=2):
    """
    Continuously load and return spot IDs (from JSON), corresponding Eye_input image,
    and the importance map (numpy file) at the same counter step.

    Remembers the last processed counter using a mode-specific counter file.
    Returns False if waiting exceeds max_wait_time.

    Args:
        mode (str): One of ["training", "initial_grouping", "organizing", "testing"].
        wait_interval (int): Seconds to wait before checking again.
        max_wait_time (int): Max seconds to wait before exiting.

    Yields:
        tuple: (list[dict], str, np.ndarray) → (spot_list, image_path, importance_map)
    """
    drive_folder = r'D:\artist\brainX\CRX\Properties\System1_inputs_eye'
    eye_input_folder = r'D:\artist\brainX\CRX\Properties\latest_images'
    importance_map_folder = r'D:\artist\brainX\CRX\Properties\latest_images'

    counter_files = {
        "training": "last_processed_counter_tr.txt",
        "initial_grouping": "last_processed_counter_ig.txt",
        "organizing": "last_processed_counter_org.txt",
        "testing": "last_processed_counter_ts.txt"
    }

    if mode not in counter_files:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of {list(counter_files.keys())}.")

    counter_file_path = os.path.join(drive_folder, counter_files[mode])

    if not os.path.exists(drive_folder):
        print(f"Directory not found at {drive_folder}")
        return False

    # Ensure counter file exists
    if not os.path.exists(counter_file_path):
        with open(counter_file_path, 'w') as f:
            f.write('0')
        print(f"Initialized last processed counter for {mode} to 0.")

    # Read the last processed counter
    with open(counter_file_path, 'r') as f:
        last_processed_counter = int(f.read().strip())

    total_wait_time = 0

    while True:
        # Spot ID files
        json_files = [f for f in os.listdir(drive_folder) if f.startswith('spot_ids_') and f.endswith('.json')]
        # Image files
        image_files = [f for f in os.listdir(eye_input_folder) if f.startswith('frame_') and f.endswith('.png')]
        # Importance maps
        importance_files = [f for f in os.listdir(importance_map_folder) if f.startswith('importance_') and f.endswith('.npy')]

        if not json_files or not image_files or not importance_files:
            print(f"[{mode}] Waiting for spot IDs, images, and importance maps... (Total wait time: {total_wait_time}s)")
            if total_wait_time >= max_wait_time:
                print(f"[{mode}] Max wait time ({max_wait_time}s) exceeded. Exiting function.")
                return False
            time.sleep(wait_interval)
            total_wait_time += wait_interval
            continue

        # Sort
        json_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
        image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        importance_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

        # Maps: counter → filename
        image_map = {int(f.split('_')[1].split('.')[0]): f for f in image_files}
        importance_map = {int(f.split('_')[1].split('.')[0]): f for f in importance_files}

        new_files_found = False

        # Group spot files by counter
        groups = {}
        for fname in json_files:
            counter = int(fname.split('_')[2].split('.')[0])
            groups.setdefault(counter, []).append(fname)

        # Process counters
        for counter, files in sorted(groups.items()):
            if counter <= last_processed_counter:
                continue  # already processed

            # Ensure image + importance map exist for this counter
            if counter not in image_map or counter not in importance_map:
                continue

            new_files_found = True
            total_wait_time = 0

            spot_list = []
            for file_name in sorted(files):
                file_path = os.path.join(drive_folder, file_name)
                try:
                    with open(file_path, 'r') as f:
                        spot_ids = json.load(f)
                    spot_list.append(spot_ids)
                    print(f"[{mode}] Loaded {file_name}")

                    if mode == "testing":
                        os.remove(file_path)
                        print(f"[{mode}] File {file_name} processed and deleted.")
                except Exception as e:
                    print(f"[{mode}] Error processing {file_name}: {e}")

            if spot_list:
                # Image path
                image_path = os.path.join(eye_input_folder, image_map[counter])
                # Importance map path
                importance_path = os.path.join(importance_map_folder, importance_map[counter])
                importance = np.load(importance_path)

                # Update counter file
                last_processed_counter = counter
                with open(counter_file_path, 'w') as f:
                    f.write(str(last_processed_counter))

                print(f"[{mode}] Processed counter {counter} → {len(spot_list)} spot files.")
                yield spot_list, image_path, importance

        if not new_files_found:
            print(f"[{mode}] No new files to process. Waiting... (Total wait time: {total_wait_time}s)")
            if total_wait_time >= max_wait_time:
                print(f"[{mode}] Max wait time ({max_wait_time}s) exceeded. Exiting function.")
                return False
            time.sleep(wait_interval)
            total_wait_time += wait_interval

def save_ring_tree_to_drive():
    """Save the current ring tree structure to a JSON file in Google Drive"""
    try:
        # Correct path with consistent naming
        drive_folder = r'D:\artist\brainX\CRX\Ring_tree'
        os.makedirs(drive_folder, exist_ok=True)

        # Prepare the data to be saved
        ring_tree_data = {
            'surface': ring_tree.surface,
            'surface_rankings': ring_tree.surface_rankings,
            'access_counts': ring_tree.access_counts,
            'access_counts_branch': ring_tree.access_counts_branch,
            'windows': ring_tree.windows,
            'branches': ring_tree.branches,
            'branch_connections': ring_tree.branch_connections,
            "cross_branch_connections": ring_tree.cross_branch_connections,
            'tree_counter': ring_tree.tree_counter,
            'branch_counter': ring_tree.branch_counter,
            'cross_branch_counter': ring_tree.cross_branch_counter,
            'current_tree': ring_tree.current_tree
        }

        # Save to a JSON file
        file_path = os.path.join(drive_folder, 'ring_tree_state.json')
        with open(file_path, 'w') as f:
            json.dump(ring_tree_data, f, indent=4)

        print(f"Ring tree state saved to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving ring tree state: {e}")
        return False

def load_ring_tree_from_drive():
    """Load the ring tree structure from a JSON file in Google Drive"""
    try:
        # Correct the path
        drive_folder = r'D:\artist\brainX\CRX\Ring_tree'
        file_path = os.path.join(drive_folder, 'ring_tree_state.json')

        # Check if the folder exists
        if not os.path.exists(drive_folder):
            print(f"Directory not found: {drive_folder}")
            return False

        # Check if the file exists
        if not os.path.isfile(file_path):
            print(f"No existing ring tree state found at {file_path}")
            return False

        # Load the JSON file
        with open(file_path, 'r') as f:
            ring_tree_data = json.load(f)

        # Restore the ring tree state
        ring_tree.surface = ring_tree_data.get('surface', {})
        ring_tree.surface_rankings = ring_tree_data.get('surface_rankings', {"R'": [], "R''": [], "R'''" : []})
        ring_tree.access_counts = ring_tree_data.get('access_counts', {})
        ring_tree.access_counts_branch = ring_tree_data.get('access_counts_branch', {})
        ring_tree.windows = ring_tree_data.get('windows', {})
        ring_tree.branches = ring_tree_data.get('branches', {})
        ring_tree.branch_connections = ring_tree_data.get('branch_connections', {})
        ring_tree.cross_branch_connections = ring_tree_data.get('cross_branch_connections', {})
        ring_tree.tree_counter = ring_tree_data.get('tree_counter', 1)
        ring_tree.branch_counter = ring_tree_data.get('branch_counter', 1)
        ring_tree.cross_branch_counter = ring_tree_data.get('cross_branch_counter', 1)
        ring_tree.current_tree = ring_tree_data.get('current_tree', None)

       # print("Ring tree state loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading ring tree state: {e}")
        return False

def id_to_list_(encoded_id):
    """Properly decode with metadata"""
    try:
        compressed = base64.urlsafe_b64decode(encoded_id.encode())
        decompressed = zlib.decompress(compressed)

        # Extract metadata
        metadata_length = struct.unpack('I', decompressed[:4])[0]
        metadata_bytes = decompressed[4:4+metadata_length]
        pixel_data = decompressed[4+metadata_length:]

        # Parse metadata
        metadata = json.loads(metadata_bytes.decode('utf-8'))
        original_shape = metadata['shape']
        original_size = metadata['size']

        # Reconstruct array
        array = np.frombuffer(pixel_data, dtype=np.uint8)

        #print(f"Decoded metadata: shape={original_shape}, size={original_size}")
        #print(f"Actual decoded size: {len(array)}")

        return array, original_shape

    except Exception as e:
        print(f"Error decoding ID with metadata: {e}")
        # Fallback to old method
        try:
            compressed = base64.urlsafe_b64decode(encoded_id.encode())
            decompressed = zlib.decompress(compressed)
            array = np.frombuffer(decompressed, dtype=np.uint8)

            # Try to guess the shape based on size
            if len(array) == 1024:
                shape = (32, 32)
            elif len(array) == 3072:
                shape = (32, 32, 3)
            else:
                shape = None

            return array, shape
        except Exception as e2:
            print(f"Fallback decoding also failed: {e2}")
            return None, None

def visualize_processed_pixels(
    focused_regions,
    images_per_row=4,
    max_images_per_figure=24,
    save_path="focused_spots_test.png",
    save_dir=r"D:\artist\brainX\CRX\Properties\mental_representation",
    counter_file_path=r"D:\artist\brainX\CRX\Properties\mental_representation\last_processed_counter.txt"
):
    """
    Visualizes processed image regions in a grid format AND saves each region as an individual PNG.
    - Grid visualization logic is preserved.
    - Individual images are saved in serial order (01.png, 02.png, ...).
    - Serial continues from last saved counter stored in `counter_file_path`.
    """
    num_regions = len(focused_regions)
    print(f"num_regions - {num_regions}")

    if num_regions == 0:
        print("No processed regions to visualize.")
        return

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # --- Load last counter ---
    if os.path.exists(counter_file_path):
        try:
            with open(counter_file_path, "r") as f:
                counter = int(f.read().strip())
        except Exception:
            counter = 0
    else:
        counter = 0

    # --- Visualization setup (grid figures) ---
    all_figures = []
    all_paths = []
    num_figures = (num_regions + max_images_per_figure - 1) // max_images_per_figure

    for fig_num in range(num_figures):
        start_idx = fig_num * max_images_per_figure
        end_idx = min((fig_num + 1) * max_images_per_figure, num_regions)
        current_batch = focused_regions[start_idx:end_idx]
        batch_size = len(current_batch)
        rows = (batch_size + images_per_row - 1) // images_per_row

        fig, axes = plt.subplots(rows, images_per_row, figsize=(images_per_row * 3, rows * 3))

        # Ensure axes is always a 2D array
        if rows == 1 and images_per_row == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = np.array([axes])
        elif images_per_row == 1:
            axes = np.array([[ax] for ax in axes])

        for i, ax in enumerate(axes.flatten()):
            if i < batch_size:
                img = np.array(current_batch[i], dtype=np.uint8)

                # Handle flattened CIFAR-like (32x32x3)
                if img.ndim == 1 and img.size == 3072:
                    img = img.reshape((32, 32, 3))

                # --- Save plain image (no titles/axes) ---
                counter += 1
                img_save_path = os.path.join(save_dir, f"{counter:04d}.png")
                try:
                    plt.imsave(img_save_path, img, cmap='gray' if img.ndim == 2 else None)
                    print(f"Saved plain image: {img_save_path}")
                except Exception as e:
                    print(f"⚠️ Could not save {img_save_path}: {e}")

                # --- Grid visualization (preserve your original logic) ---
                if img.ndim == 2:
                    ax.imshow(img, cmap='gray')
                elif img.ndim == 3 and img.shape[2] == 3:
                    ax.imshow(img)
                else:
                    print(f"Skipping unrecognized image shape: {img.shape}")
                    ax.axis('off')
                    continue

                ax.set_title(f"Spot {start_idx + i + 1}")
                ax.axis('off')
            else:
                ax.axis('off')

        plt.suptitle(f"Focused Spots (Batch {fig_num + 1} of {num_figures})", fontsize=14)
        plt.tight_layout()

        # Show the figure interactively during the loop
        plt.show()

        # Save path setup
        if num_figures > 1:
            base, ext = os.path.splitext(save_path)
            batch_save_path = f"{base}_batch{fig_num + 1}{ext}"
        else:
            batch_save_path = save_path

        all_figures.append(fig)
        all_paths.append(batch_save_path)

    # Save grid figures after loop
    for fig, path in zip(all_figures, all_paths):
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved as {path}")
        plt.close(fig)

    # --- Save updated counter ---
    try:
        with open(counter_file_path, "w") as f:
            f.write(str(counter))
        print(f"Updated counter saved at {counter_file_path}: {counter}")
    except Exception as e:
        print(f"⚠️ Could not update counter file: {e}")

def get_next_sound_input(mode="training"):
    """
    Loads the next unprocessed phoneme ID from Google Drive storage.
    Increments the counter file after processing.

    Returns:
        The phoneme ID (string, list, or dict depending on file content),
        or None if no new phoneme is available.
    """
    drive_folder = r'D:\artist\brainX\CRX\Properties\System1_inputs_ear'
    counter_files = {
        "training": "last_processed_counter_tr.txt",
        "initial_grouping": "last_processed_counter_ig.txt",
        "organizing": "last_processed_counter_org.txt",
        "testing": "last_processed_counter_ts.txt"
    }

    if mode not in counter_files:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of {list(counter_files.keys())}.")

    counter_file_path = os.path.join(drive_folder, counter_files[mode])

    # Ensure the directory exists
    if not os.path.exists(drive_folder):
        print(f"Phoneme directory not found at {drive_folder}")
        return None

    # Ensure the counter file exists
    if not os.path.exists(counter_file_path):
        with open(counter_file_path, 'w') as f:
            f.write('0')
        last_processed_counter = 0
    else:
        with open(counter_file_path, 'r') as f:
            last_processed_counter = int(f.read().strip())

    # List phoneme JSON files
    json_files = [f for f in os.listdir(drive_folder) if f.startswith('phoneme_') and f.endswith('.json')]
    if not json_files:
        print("No phoneme files found.")
        return None

    # Sort by counter value
    json_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    # Find the next unprocessed file
    for file_name in json_files:
        current_counter = int(file_name.split('_')[1].split('.')[0])
        if current_counter > last_processed_counter:
            file_path = os.path.join(drive_folder, file_name)

            try:
                # Load phoneme ID
                with open(file_path, 'r', encoding='utf-8') as f:
                    phoneme_id = json.load(f)

                print(f"Loaded phoneme from {file_name}")

                # Update counter
                with open(counter_file_path, 'w') as f:
                    f.write(str(current_counter))

                return phoneme_id

            except Exception as e:
                print(f"Error loading {file_name}: {e}")
                return None

    print("No new phoneme to process.")
    return None

def load_phoneme_ids_from_drive_with_waiting(mode="training", wait_interval=1, max_wait_time=2):
    """
    Continuously load and return phoneme IDs, corresponding Eye_input image,
    and importance map (numpy array) if available at the same counter step.

    Remembers the last processed counter using a mode-specific counter file.
    Exits if waiting exceeds max_wait_time.

    Args:
        mode (str): One of ["training", "initial_grouping", "organizing", "testing"].
        wait_interval (int): Time (in seconds) to wait before checking again.
        max_wait_time (int): Maximum time (in seconds) to wait before exiting.

    Yields:
        tuple: (phoneme_ids, image_path or None, importance_map or None)
        
    Returns:
        bool: False if function exits due to timeout.
    """
    drive_folder = r'D:\artist\brainX\CRX\Properties\System1_inputs_ear'
    eye_input_folder = r'D:\artist\brainX\CRX\Properties\latest_images'
    importance_map_folder = r'D:\artist\brainX\CRX\Properties\latest_images'

    counter_files = {
        "training": "last_processed_counter_tr.txt",
        "initial_grouping": "last_processed_counter_ig.txt",
        "organizing": "last_processed_counter_org.txt",
        "testing": "last_processed_counter_ts.txt"
    }

    if mode not in counter_files:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of {list(counter_files.keys())}.")

    counter_file_path = os.path.join(drive_folder, counter_files[mode])

    if not os.path.exists(drive_folder):
        print(f"Directory not found at {drive_folder}")
        return False

    # Ensure counter file exists
    if not os.path.exists(counter_file_path):
        with open(counter_file_path, 'w') as f:
            f.write('0')
        print(f"Initialized last processed counter for {mode} to 0.")

    # Read last processed counter
    with open(counter_file_path, 'r') as f:
        last_processed_counter = int(f.read().strip())

    total_wait_time = 0

    while True:
        # List available files
        phoneme_files = [f for f in os.listdir(drive_folder) if f.startswith('phoneme_') and f.endswith('.json')]
        image_files = [f for f in os.listdir(eye_input_folder) if f.startswith('frame_') and f.endswith('.png')]
        importance_files = [f for f in os.listdir(importance_map_folder) if f.startswith('importance_') and f.endswith('.npy')]

        if not phoneme_files:
            print(f"[{mode}] Waiting for phoneme files... (Total wait time: {total_wait_time}s)")
            if total_wait_time >= max_wait_time:
                print(f"[{mode}] Max wait time exceeded. Exiting function.")
                return False
            time.sleep(wait_interval)
            total_wait_time += wait_interval
            continue

        # Sort files by counter
        phoneme_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        importance_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

        # Maps
        image_map = {int(f.split('_')[1].split('.')[0]): f for f in image_files}
        importance_map = {int(f.split('_')[1].split('.')[0]): f for f in importance_files}

        new_files_found = False

        # Process new phoneme files
        for file_name in phoneme_files:
            file_path = os.path.join(drive_folder, file_name)
            try:
                current_counter = int(file_name.split('_')[1].split('.')[0])

                if current_counter <= last_processed_counter:
                    continue  # already processed

                new_files_found = True
                total_wait_time = 0

                # Load phoneme IDs
                with open(file_path, 'r') as f:
                    phoneme_ids = json.load(f)

                # Default to None
                image_path = None
                importance = None

                # Try image
                if current_counter in image_map:
                    image_path = os.path.join(eye_input_folder, image_map[current_counter])

                # Try importance
                if current_counter in importance_map:
                    importance_path = os.path.join(importance_map_folder, importance_map[current_counter])
                    try:
                        importance = np.load(importance_path)
                    except Exception as e:
                        print(f"[{mode}] Failed to load importance map {importance_path}: {e}")

                print(f"[{mode}] Processing counter {current_counter} → phoneme:{file_name}, "
                      f"image:{os.path.basename(image_path) if image_path else 'None'}, "
                      f"importance:{'OK' if importance is not None else 'None'}")

                # Update counter
                last_processed_counter = current_counter
                with open(counter_file_path, 'w') as f:
                    f.write(str(last_processed_counter))

                # Delete only in testing mode
                if mode == "testing":
                    os.remove(file_path)
                    if image_path and os.path.exists(image_path):
                        os.remove(image_path)
                    if importance is not None and os.path.exists(importance_path):
                        os.remove(importance_path)
                    print(f"[{mode}] Files for counter {current_counter} processed and deleted.")
                else:
                    print(f"[{mode}] Files for counter {current_counter} processed.")

                yield phoneme_ids, image_path, importance

            except Exception as e:
                print(f"[{mode}] Error processing {file_name}: {e}")

        if not new_files_found:
            print(f"[{mode}] No new phoneme files. Waiting... (Total wait time: {total_wait_time}s)")
            if total_wait_time >= max_wait_time:
                print(f"[{mode}] Max wait time exceeded. Exiting function.")
                return False
            time.sleep(wait_interval)
            total_wait_time += wait_interval

#sound output to words
def id_to_word_sound(encoded_id, original_length, phoneme_dict=None):
    """
    Convert encoded sound ID -> list of integers -> phonemes -> word.
    Now uses dynamically generated phoneme dictionary.
    """
    def load_phoneme_dictionary(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading dictionary: {e}")
            return {}
        
    # Load phoneme dictionary if not provided
    if phoneme_dict is None:
        phoneme_dict = load_phoneme_dictionary(r"D:\artist\brainX\CRX\Datasets\words\phoneme_dictionary.json")
    
    # Step 1: Decode ID back to list of integers
    b64 = base62_to_base64_sound(encoded_id)
    compressed = base64.b64decode(b64)
    data_bytes = zlib.decompress(compressed)
    int_list = np.frombuffer(data_bytes, dtype=np.uint8).tolist()
    
    # Step 2: Convert integers back to phoneme string
    phoneme_str = bytes(int_list).decode('utf-8', errors='ignore')
    
    # Step 3: Map phoneme to actual word using generated dictionary
    word = phoneme_dict.get(phoneme_str, "[Unknown word]")
    
    return {
        "int_list": int_list,   # raw integers
        "phoneme": phoneme_str, # decoded phoneme string
        "word": word,
        "id": encoded_id         # mapped word
    }

#id to move
def id_to_word_move(encoded_id, original_length):
    """
    Convert encoded sound ID -> list of integers -> mapped word.
    Only full-sequence matches are allowed. If no match, returns "[Unknown]".
    """
    # Step 1: Decode ID back to list of integers
    b64 = base62_to_base64_sound(encoded_id)
    compressed = base64.b64decode(b64)
    data_bytes = zlib.decompress(compressed)
    int_list = np.frombuffer(data_bytes, dtype=np.uint8).tolist()
    print(f"int_list - {int_list}")

    # Step 2: Define full sequence → word mapping
    sequence_to_word = {
        (119, 201, 148, 107): "WALK",
        (114, 201, 153, 110): "RUN",
        (123, 45): "STOP",
        (67, 89, 0): "JUMP"
    }

    # Step 3: Match only full sequence
    tuple_key = tuple(int_list)
    word = sequence_to_word.get(tuple_key, "[Unknown]")

    return {
        "int_list": int_list,
        "word": word
    }

# save move data
def save_move_words(
    words, 
    save_dir=r"D:\artist\brainX\CRX\Properties\Motor_output",
    counter_file_path=r"D:\artist\brainX\CRX\Properties\Motor_output\last_word_counter.txt"
):
    """
    Save recognized words into sequentially numbered JSON files.
    Each file will contain only the recognized word.
    """

    # Handle both single word and list of words
    if isinstance(words, str):
        words = [words]  # Convert single word to list
    
    os.makedirs(save_dir, exist_ok=True)

    # Ensure the directory for counter file exists
    os.makedirs(os.path.dirname(counter_file_path), exist_ok=True)

    # Ensure counter file exists
    if not os.path.exists(counter_file_path):
        with open(counter_file_path, "w", encoding="utf-8") as f:
            f.write("0")
        last_processed = 0
    else:
        with open(counter_file_path, "r", encoding="utf-8") as f:
            try:
                last_processed = int(f.read().strip())
            except ValueError:
                last_processed = 0

    # Start from next number after last processed
    start_index = last_processed + 1

    for idx, word in enumerate(words, start=start_index):
        file_name = f"word_{idx:03d}.json"
        save_path = os.path.join(save_dir, file_name)

        # Only keep the word itself
        word_data = {"word": word}

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(word_data, f, ensure_ascii=False, indent=2)

        print(f"Saved {file_name} → {save_dir}")

        # Update counter file after each save
        with open(counter_file_path, "w", encoding="utf-8") as f:
            f.write(str(idx))

#save sound output
def save_sound_words(words, 
                     save_dir=r"D:\artist\brainX\CRX\Properties\Speech_output",
                     counter_file_path=r"D:\artist\brainX\CRX\Properties\Speech_output\last_word_counter.txt"):
    """
    Save recognized words into sequentially numbered JSON files.
    Each file will contain only the recognized word.
    """
    os.makedirs(save_dir, exist_ok=True)

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

    for idx, word in enumerate(words, start=start_index):
        file_name = f"word_{idx:03d}.json"
        save_path = os.path.join(save_dir, file_name)

        # Only keep the word itself
        word_data = {"word": word}

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(word_data, f, ensure_ascii=False, indent=2)

        print(f"Saved {file_name} → {save_dir}")

        # Update counter file after each save
        with open(counter_file_path, "w") as f:
            f.write(str(idx))

# --- Storage paths ---
def load_latest_image(image_path, mode="training"):
    """
    Retrieve the latest stored image if it exists.
    Returns:
        image (numpy array) or None if not found
    """
    print(f"image - {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("[Load] Failed to read the latest image.")
        return None

    print(f"[Load] Loaded latest image -> {image_path}")
    # Delete the image file after loading
    if mode == "testing":
        try:
            os.remove(image_path)
            print(f"[Load] Deleted image file -> {image_path}")
        except Exception as e:
            print(f"[Load] Could not delete {image_path}: {e}")
    else:
        pass
    return image

def load_spot_and_phoneme_ids_with_waiting(mode="training", wait_interval=1, max_wait_time=2):
    """
    Continuously load and return spot IDs (visual), phoneme IDs (audio), 
    Eye_input image, and the importance map when all are available 
    at the same counter step.

    Remembers the last processed counter using a mode-specific counter file.
    Exits if waiting exceeds max_wait_time.

    Args:
        mode (str): One of ["training", "initial_grouping", "organizing", "testing"].
        wait_interval (int): Time (in seconds) to wait before checking again.
        max_wait_time (int): Maximum time (in seconds) to wait before exiting.

    Yields:
        tuple(list, str, np.ndarray): ([spot_ids, phoneme_ids], image_path, importance_map)
        
    Returns:
        bool: False if function exits due to timeout.
    """
    # Folders
    spot_folder = r'D:\artist\brainX\CRX\Properties\System1_inputs_eye'
    phoneme_folder = r'D:\artist\brainX\CRX\Properties\System1_inputs_ear'
    eye_input_folder = r'D:\artist\brainX\CRX\Properties\latest_images'
    importance_map_folder = r'D:\artist\brainX\CRX\Properties\latest_images'

    counter_files = {
        "training": "last_processed_counter_tr.txt",
        "initial_grouping": "last_processed_counter_ig.txt",
        "organizing": "last_processed_counter_org.txt",
        "testing": "last_processed_counter_ts.txt"
    }

    if mode not in counter_files:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of {list(counter_files.keys())}.")

    counter_file_path = os.path.join(spot_folder, counter_files[mode])  # store counter in eye folder

    # Ensure folders exist
    for folder in [spot_folder, phoneme_folder, eye_input_folder, importance_map_folder]:
        if not os.path.exists(folder):
            print(f"Directory not found: {folder}")
            return False

    # Ensure counter file exists
    if not os.path.exists(counter_file_path):
        with open(counter_file_path, 'w') as f:
            f.write('0')
        print(f"Initialized last processed counter for {mode} to 0.")

    # Read last counter
    with open(counter_file_path, 'r') as f:
        last_processed_counter = int(f.read().strip())

    total_wait_time = 0

    while True:
        # Collect files
        spot_files = [f for f in os.listdir(spot_folder) if f.startswith('spot_ids_') and f.endswith('.json')]
        phoneme_files = [f for f in os.listdir(phoneme_folder) if f.startswith('phoneme_') and f.endswith('.json')]
        image_files = [f for f in os.listdir(eye_input_folder) if f.startswith('frame_') and f.endswith('.png')]
        importance_files = [f for f in os.listdir(importance_map_folder) if f.startswith('importance_') and f.endswith('.npy')]
        
        if not spot_files or not phoneme_files or not image_files or not importance_files:
            print(f"[{mode}] Waiting for spot, phoneme, image, and importance files... (Total wait: {total_wait_time}s)")
            if total_wait_time >= max_wait_time:
                print(f"[{mode}] Max wait time exceeded. Exiting.")
                return False
            time.sleep(wait_interval)
            total_wait_time += wait_interval
            continue

        # Sort them by counter
        spot_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
        phoneme_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        importance_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

        new_files_found = False

        # Map counters
        phoneme_map = {int(f.split('_')[1].split('.')[0]): f for f in phoneme_files}
        spot_map = {int(f.split('_')[2].split('.')[0]): f for f in spot_files}
        image_map = {int(f.split('_')[1].split('.')[0]): f for f in image_files}
        importance_map = {int(f.split('_')[1].split('.')[0]): f for f in importance_files}

        # Process by matching counter
        for phoneme_counter, phoneme_file in phoneme_map.items():
            if phoneme_counter <= last_processed_counter:
                continue  # already processed

            if (phoneme_counter not in spot_map or 
                phoneme_counter not in image_map or 
                phoneme_counter not in importance_map):
                continue  # wait until all three exist

            # Load spot
            spot_path = os.path.join(spot_folder, spot_map[phoneme_counter])
            with open(spot_path, 'r') as f:
                spot_ids = json.load(f)

            # Load phoneme
            phoneme_path = os.path.join(phoneme_folder, phoneme_file)
            with open(phoneme_path, 'r') as f:
                phoneme_ids = json.load(f)

            # Image path
            image_path = os.path.join(eye_input_folder, image_map[phoneme_counter])

            # Importance map
            importance_path = os.path.join(importance_map_folder, importance_map[phoneme_counter])
            importance = np.load(importance_path)

            output = [spot_ids, phoneme_ids]
            print(f"[{mode}] Processing counter {phoneme_counter} → {spot_map[phoneme_counter]}, {phoneme_file}, {image_map[phoneme_counter]}, {importance_map[phoneme_counter]}")

            # Update counter
            last_processed_counter = phoneme_counter
            with open(counter_file_path, 'w') as f:
                f.write(str(last_processed_counter))

            # Delete files in testing mode
            if mode == "testing":
                os.remove(spot_path)
                os.remove(phoneme_path)
               # os.remove(image_path)
                os.remove(importance_path)
               # print(f"[{mode}] Files processed and deleted.")
            else:
               # print(f"[{mode}] Files processed.")
                pass

            new_files_found = True
            total_wait_time = 0
            output_ = []

            for item in output:
                if isinstance(item, list):
                    for sub_item in item:
                        output_.append(sub_item)
                else:
                    output_.append(item)

            #print(f"output - {output_}")
            # Yield ([spot_ids, phoneme_ids], image_path, importance)
            yield output_, image_path, importance

        if not new_files_found:
            print(f"[{mode}] No new matching sets. Waiting... (Total wait: {total_wait_time}s)")
            if total_wait_time >= max_wait_time:
                print(f"[{mode}] Max wait time exceeded. Exiting.")
                return False
            time.sleep(wait_interval)
            total_wait_time += wait_interval

def system1_main():
    shared_config = SharedConfig()

    # one run from training to testing = one dynamic ring formation cycle for the brain
    while True:  # Continuous loop to mimic the brain
        # 1. Gather values from config
        config = shared_config.load_config()
        training = config.get("training", False)
        initial_grouping = config.get("initial_grouping", False)
        organizing = config.get("organizing", False)
        testing = config.get("testing", False)
        eye = config.get("eye", False)
        ear = config.get("ear", False)

        print("\n[system1_main] Config Loaded:")
        print(f"  training={training}, initial_grouping={initial_grouping}, "
              f"organizing={organizing}, testing={testing}, eye={eye}, ear={ear}")

        # 2. Run based on config values
        if training:
            print(">>> Training Mode")
            if eye and ear:
                print(f"eye and ear")
                for inputs,image,importance_map in load_spot_and_phoneme_ids_with_waiting(mode="training"):
                    image = load_latest_image(image, mode="training")
                    if inputs is False: break
                    for input in inputs:
                        print(ring_tree(input, training=True, eye=True, ear=True, image=image, importance_map=importance_map))
                    save_ring_tree_to_drive()
            elif eye:
                
                for inputs,image,importance_map in load_spot_ids_from_drive_with_waiting(mode="training"):
                    image = load_latest_image(image, mode="training")
                    if inputs is False: break
                    for input in inputs:
                        print(ring_tree(input, training=True, eye=True, image=image, importance_map=importance_map))
                    save_ring_tree_to_drive()
            elif ear:
                
                for inputs,image,importance_map in load_phoneme_ids_from_drive_with_waiting(mode="training"):
                    if image is not None:
                        image = load_latest_image(image, mode="training")
                    else:
                        print("[training] No image available for this counter.")
                        image = None

                    if inputs is False: break
                    print(ring_tree(inputs, training=True, ear=True, image=image, importance_map=importance_map))
                    save_ring_tree_to_drive()

        if initial_grouping:
            print(">>> Initial Grouping Mode")
            if eye and ear:
                print(f"eye and ear")
                for inputs,image, importance_map in load_spot_and_phoneme_ids_with_waiting(mode="initial_grouping"):
                    image = load_latest_image(image, mode="initial_grouping")
                    if inputs is False: break
                    for input in inputs:
                        print(ring_tree(input, initial_grouping=True, eye=True, ear=True, image=image, importance_map=importance_map))
                    save_ring_tree_to_drive()
            elif eye:  
                for inputs,image,importance_map in load_spot_ids_from_drive_with_waiting(mode="initial_grouping"):
                    image = load_latest_image(image, mode="initial_grouping")
                    if inputs is False: break
                    for input in inputs:
                        print(ring_tree(input, initial_grouping=True, eye=True, image=image, importance_map=importance_map))
                    save_ring_tree_to_drive()
            elif ear:
                for inputs,image,importance_map in load_phoneme_ids_from_drive_with_waiting(mode="initial_grouping"):
                    if image is not None:
                        image = load_latest_image(image, mode="initial_grouping")
                    else:
                        print("[initial_grouping] No image available for this counter.")
                        image = None
                    if inputs is False: break
                    print(ring_tree(inputs, initial_grouping=True, ear=True, image=image, importance_map=importance_map))
                    save_ring_tree_to_drive()

        if organizing:
            print(">>> Organizing Mode")
            if eye and ear:
                print(f"eye and ear")
                for inputs, image, importance_map in load_spot_and_phoneme_ids_with_waiting(mode="organizing"):
                    image = load_latest_image(image, mode="organizing")
                    if inputs is False: break
                    for input in inputs:
                        print(ring_tree(input, organizing=True, eye=True, ear=True, image=image, importance_map=importance_map))
                    save_ring_tree_to_drive()
            elif eye:
                for inputs,image,importance_map in load_spot_ids_from_drive_with_waiting(mode="organizing"):
                    image = load_latest_image(image, mode="organizing")
                    if inputs is False: break
                    for input in inputs:
                        print(ring_tree(input, organizing=True, eye=True, image=image, importance_map=importance_map))
                    save_ring_tree_to_drive()
            elif ear:
                for inputs,image,importance_map in load_phoneme_ids_from_drive_with_waiting(mode="organizing"):
                    if image is not None:
                        image = load_latest_image(image, mode="organizing")
                    else:
                        print("[organizing] No image available for this counter.")
                        image = None
                    if inputs is False: break
                    print(ring_tree(inputs, organizing=True, ear=True, image=image, importance_map=importance_map))
                    save_ring_tree_to_drive()

        if testing:
            print(">>> Testing Mode")
            if eye and ear:
                print(f"eye and ear")
                for inputs, image, importance_map in load_spot_and_phoneme_ids_with_waiting(mode="testing"):
                    image = load_latest_image(image, mode="testing")
                    if inputs is False: break
                    for input in inputs:
                        print(ring_tree(input, testing=True, eye=True, ear=True, image=image, importance_map=importance_map))
                    save_ring_tree_to_drive()
            elif eye:
                for inputs,image,importance_map in load_spot_ids_from_drive_with_waiting(mode="testing"):
                    image = load_latest_image(image, mode="testing")
                    if inputs is False: break
                    for input in inputs:
                        print(ring_tree(input, testing=True, eye=True, image=image, importance_map=importance_map))
                    save_ring_tree_to_drive()
            elif ear:
                for inputs,image,importance_map in load_phoneme_ids_from_drive_with_waiting(mode="testing"):
                    if image is not None:
                        image = load_latest_image(image, mode="testing")
                    else:
                        print("[testing] No image available for this counter.")
                        image = None
                
                    if inputs is False: break
                    print(ring_tree(inputs, testing=True, ear=True, image=image, importance_map=importance_map))
                    save_ring_tree_to_drive()

            # 3. After testing finishes, pause before reloading new config
            print("[system1_main] Testing complete. Waiting 3s before reloading config...")
            time.sleep(2)
            continue  # restart loop, reload config

        # If no active mode, let System2 take over (destabilizing mechanism)
        else:
            print("[system1_main] No active mode -> System2 fallback")
            print(ring_tree(destabilizing_mechanism=True))
            time.sleep(0.5)

if __name__ == "__main__":
    system1_main()
