# imagination to finalize and compare output nodes
# hashing will help to organize nodes within the ring tree by giving meaningful reprocess nodes

import os
import json
import hashlib
import json
from typing import Any, List


class Imagination:
    def __init__(self, cleanup_threshold_nodes: int=5,cleanup_threshold: int = 10, priority_threshold: int = 10):
        # Data directory
        self.data_dir = r"D:\artist\brainX\CRX\Properties\Final_outputs"
        self.stored_nodes_file = os.path.join(self.data_dir, "stored_nodes.json")
        self.activation_counts_file = os.path.join(self.data_dir, "activation_counts.json")
        self.activation_counts_nodes_file = os.path.join(self.data_dir, "activation_counts_nodes.json")
        self.pre_priority_nodes_file = os.path.join(self.data_dir, "pre_priority_nodes.json")
            
        # Thresholds
        self.cleanup_threshold = cleanup_threshold
        self.cleanup_threshold_nodes = cleanup_threshold_nodes
        self.priority_threshold = priority_threshold

        # Load saved data (or initialize empty if not present)
        self.load_data()

        # Runtime-only
        self.nodes_to_reprocess = []

        # Ensure directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
    def save_data(self):
        """Save all data to JSON files"""
        try:
            # Save stored_nodes
            with open(self.stored_nodes_file, 'w') as f:
                json.dump(self.stored_nodes, f, indent=2)
            
            # Save activation_counts (convert keys to strings for JSON)
            activation_counts_str = {str(k): v for k, v in self.activation_counts.items()}
            with open(self.activation_counts_file, 'w') as f:
                json.dump(activation_counts_str, f, indent=2)
            
            # Save activation_counts (convert keys to strings for JSON)
            activation_counts_str_ = {str(k): v for k, v in self.activation_counts_nodes.items()}
            with open(self.activation_counts_nodes_file, 'w') as f:
                json.dump(activation_counts_str_, f, indent=2)

            # Save pre_priority_nodes
            with open(self.pre_priority_nodes_file, 'w') as f:
                json.dump(self.pre_priority_nodes, f, indent=2)
                
           # print("Data saved successfully")
            
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def load_data(self):
        """Load all data from JSON files if they exist, otherwise keep empty defaults"""
        try:
            # Load stored_nodes
            if os.path.exists(self.stored_nodes_file):
                with open(self.stored_nodes_file, 'r') as f:
                    self.stored_nodes = json.load(f)
            else:
                self.stored_nodes = []

            # Load activation_counts
            if os.path.exists(self.activation_counts_file):
                with open(self.activation_counts_file, 'r') as f:
                    data = json.load(f)
                    # Convert string keys back to original format if needed
                    self.activation_counts = {k: v for k, v in data.items()}
            else:
                self.activation_counts = {}

            # Load activation_counts_nodes
            if os.path.exists(self.activation_counts_nodes_file):
                with open(self.activation_counts_nodes_file, 'r') as f:
                    self.activation_counts_nodes = json.load(f)
            else:
                self.activation_counts_nodes = {}

            # Load pre_priority_nodes
            if os.path.exists(self.pre_priority_nodes_file):
                with open(self.pre_priority_nodes_file, 'r') as f:
                    self.pre_priority_nodes = json.load(f)
            else:
                self.pre_priority_nodes = []

           # print("Data loaded successfully")

        except Exception as e:
            print(f"Error loading data: {e}")
            # Fallback to empty if anything goes wrong
            self.stored_nodes = []
            self.activation_counts = {}
            self.activation_counts_nodes = {}
            self.pre_priority_nodes = []

    def _flatten_ordered(self, obj: Any, result: List[str]) -> None:
        """
        Flatten nested lists/tuples/dicts into an ordered sequence of leaf values (strings).
        Order is preserved, structure is ignored.
        """
        if isinstance(obj, dict):
            # For dicts, process keys in sorted order to stay consistent
            for k in sorted(obj.keys()):
                self._flatten_ordered(obj[k], result)
        elif isinstance(obj, (list, tuple, set)):
            for x in obj:
                self._flatten_ordered(x, result)
        else:
            result.append(str(obj))

    def _generate_hash(self, node_data: Any) -> str:
        """
        Generate a hash that respects order but ignores nesting structure.
        Example:
        ['A','B']        → same hash as [['A'],['B']]
        ['B','A']        → different hash
        """
        flattened: List[str] = []
        self._flatten_ordered(node_data, flattened)

        # Convert to canonical string (JSON to ensure stable representation)
        canonical_str = json.dumps(flattened, separators=(',', ':'), ensure_ascii=False)

        # Create hash
        return hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()

    def _find_consecutive_matches(self, new_node, pre_priority_nodes):
        """
        Find consecutive matches between new_node and stored_nodes.
        Returns (matches, indices):
            matches = list of matched subsequences (length >= 2)
            indices = list of lists of indices where matches occurred
        """
        # Flatten stored nodes
        stored_nodes = []
        for node, _ in pre_priority_nodes:
            stored_nodes.extend(node)

        matches = []
        indices = []

        len_new = len(new_node)
        len_stored = len(stored_nodes)

        print(f"Comparing new_node: {len_new} with stored_nodes: {len_stored}")

        # Slide over stored_nodes
        for start in range(len_stored - len_new + 1):
            consecutive_match = []
            consecutive_index = []

            for j in range(len_new):
                if new_node[j] == stored_nodes[start + j]:
                    consecutive_match.append(new_node[j])
                    consecutive_index.append(start + j)
                else:
                    if len(consecutive_match) >= 2:
                        matches.append(consecutive_match[:])
                        indices.append(consecutive_index[:])
                       # print(f"Found consecutive match: {consecutive_match} at indices: {consecutive_index}")
                    consecutive_match = []
                    consecutive_index = []

            # Catch trailing matches at the end of this window
            if len(consecutive_match) >= 2:
                matches.append(consecutive_match[:])
                indices.append(consecutive_index[:])

        return matches, indices

  
    def process_output_node(self, node_data):
        """
        Process incoming output node.
        Returns: (final_output, should_send_to_output_function)
        """
        # Load data before processing
        self.load_data()
        slice_data = None
        print(f"Processing node: {node_data}")
        
        # First check against stored nodes (hashed nodes with consecutive matches)
        if self.stored_nodes:
           # print("entering imagination nodes")
            # Get all unique consecutive lengths from stored nodes
            consecutive_lengths = list(set(item[2] for item in self.stored_nodes))
            slice_hashes = []
            slice_map = {}  # map hash -> slice_data for logging
           # print(f"Consecutive lengths to check: {consecutive_lengths}")
            # Check each consecutive length
            for length in consecutive_lengths:
                if len(node_data) >= length:
                    # Collect all slice hashes for this length
                    for i in range(len(node_data) - length + 1):
                        slice_data = node_data[i:i+length]
                        slice_hash = self._generate_hash(slice_data)
                        slice_hashes.append(slice_hash)
                        slice_map[slice_hash] = slice_data
            #print(f"Slice hashes generated: {slice_hashes}")
            #print(f"Slice map: {slice_map}")
            # Compare each unique slice hash against all stored hashes once
            for slice_hash in set(slice_hashes):
                for stored_hash, stored_nodes_data, stored_length in self.stored_nodes:
                   # print(f"Comparing slice hash: {slice_hash} with stored hash: {stored_hash}")
                    if slice_hash == stored_hash and length == stored_length:
                        print(f"Hash match found! Slice: {slice_map[slice_hash]}")

                        # Update activation count using hash
                        self.activation_counts[stored_hash] = self.activation_counts.get(stored_hash, 0) + 1
                        
                        # Send matched stored nodes back to testing phase
                        self.nodes_to_reprocess.append(stored_nodes_data)
                        
                        # Save data after update
                        self.save_data()
                        slice_map[slice_hash] = slice_data
                       # print(f"final output - {slice_data}")

                        # ✅ Remove slice_data from node_data after processing
                        for val in slice_data:
                            if val in node_data:
                                node_data.remove(val)
                      #  print(f"Updated node_data after removal: {node_data}")
                        
        # Track which nodes got incremented in this round
        incremented_nodes = []
        incremented_index = []
        nodes_to_remove = []
        
        # Check each individual element in node_data against pre-priority nodes
        # Found a match, check consecutive matches for the entire sequences
        consecutive_matches, consecutive_index = self._find_consecutive_matches(node_data, self.pre_priority_nodes)
        if consecutive_matches:  # at least one subsequence of length >= 2
            #print(f"Consecutive match: {consecutive_matches}")

            for match in consecutive_matches:
                incremented_nodes.append(match)
            for index in consecutive_index:
                incremented_index.append(index)

           # print(f"Incremented nodes: {incremented_nodes}")
          #  print(f"Incremented indices: {incremented_index}")
            # Update frequencies for incremented nodes and check for promotion
            for n in consecutive_index:
                # Step 1: Increase frequencies
                for i in n:
                    stored_data, frequency = self.pre_priority_nodes[i]
                    frequency += 1
                    self.pre_priority_nodes[i] = (stored_data, frequency)

                    # Update activation count for pre-priority nodes
                    self.activation_counts_nodes[i] = self.activation_counts_nodes.get(i, 0) + 1
                  #  print(f"Node frequency increased to {frequency}")

                # Step 2: Check if *all* nodes in this consecutive group exceed threshold
                if all(self.pre_priority_nodes[i][1] >= self.priority_threshold for i in n):
                    # ✅ All nodes exceed threshold → promote this group
                    group_data = [self.pre_priority_nodes[i][0] for i in n]

                    # Generate a single hash for the combined data
                    node_hash = self._generate_hash(group_data)

                    # Store once with combined data
                    self.stored_nodes.append((node_hash, group_data, len(n)))
                    self.activation_counts[node_hash] = 1

                   # print("nodes stored into imagination")

                    # Mark these nodes for removal
                    for i in n:
                        nodes_to_remove.append(i)

            # Step 4: Remove nodes (reverse order to preserve indices)
            for i in sorted(set(nodes_to_remove), reverse=True):
                del self.pre_priority_nodes[i]
                if i in self.activation_counts_nodes:
                    del self.activation_counts_nodes[i]

            # Save data after processing
            self.save_data()

            # Add new individual nodes that weren't found consecutive match
            found_elements = []
            streak = []
            # Flatten all consecutive matches into a single set for quick lookup
            matched_set = set(item for subseq in consecutive_matches for item in subseq)

            for n in node_data:
                if n not in matched_set:
                    streak.append(n)
                else:
                    if len(streak) >= 2:
                        for x in streak:
                            if x not in found_elements:  # preserve order, avoid duplicates
                                found_elements.append(str(x))
                    streak = []  # reset streak

            # Catch trailing streak at the end
            if len(streak) >= 2:
                for x in streak:
                    if x not in found_elements:
                        found_elements.append(str(x))

        else:  
            # Loop over each node in node_data and add individually with frequency 1
            for node_element in node_data:
                # Extract the stored node strings from pre_priority_nodes
               # existing_nodes = [item[0][0] for item in self.pre_priority_nodes]  
               # if node_element not in existing_nodes:
               # print(f"New node added to pre-priority: {node_element}")
                self.pre_priority_nodes.append(([node_element], 1))
    
            # Save data after processing
            self.save_data()
            
            # Cleanup if needed
            if len(self.stored_nodes) % 50 == 0:
                self._cleanup_unused_nodes()
        if slice_data:
           # print("returning final output")
            # Return the matched slice as final output
            return slice_data, True
        else:
            return node_data, False

    def _cleanup_unused_nodes(self):
        """Remove nodes with low activation counts"""
        # Cleanup stored nodes (hash-based)
        if self.activation_counts:
            hashes_to_remove = []
            for node_hash, count in self.activation_counts.items():
                if int(count) < self.cleanup_threshold:
                    hashes_to_remove.append(node_hash)
            
            # Remove stored nodes with low activation
            for hash_to_remove in hashes_to_remove:
                # Find and remove the node with this hash
                for i, (stored_hash, stored_data, consecutive_length) in enumerate(self.stored_nodes):
                    if stored_hash == hash_to_remove:
                       # print(f"Cleaning up unused stored node - hash: {hash_to_remove}")
                        del self.stored_nodes[i]
                        del self.activation_counts[hash_to_remove]
                        break
        
        # Cleanup pre-priority nodes (index-based)
        if self.activation_counts_nodes:
            indices_to_remove = []
            for node_index, count in self.activation_counts_nodes.items():
                if int(count) < self.cleanup_threshold_nodes and int(node_index) < len(self.pre_priority_nodes):
                    indices_to_remove.append(node_index)
            
            # Remove pre-priority nodes (in reverse order to maintain indices)
            for index in sorted(indices_to_remove, reverse=True):
                if int(index) < len(self.pre_priority_nodes):
                    stored_data, frequency = self.pre_priority_nodes[int(index)]
                   # print(f"Cleaning up unused pre-priority node at index {index}")
                    del self.pre_priority_nodes[int(index)]
                    del self.activation_counts_nodes[index]
            
            # Reindex activation_counts_nodes after removal
            new_activation_counts_nodes = {}
            for old_index, count in self.activation_counts_nodes.items():
                # Calculate new index after removals
                new_index = int(old_index) - sum(1 for removed in indices_to_remove if int(removed) < int(old_index))
                new_activation_counts_nodes[new_index] = count
            
            self.activation_counts_nodes = new_activation_counts_nodes
        
        # Save data after cleanup
        self.save_data()
                
    def get_nodes_for_testing(self):
        """Get nodes that should be sent back to testing phase"""
        nodes = self.nodes_to_reprocess.copy()
        self.nodes_to_reprocess.clear()  # Clear after getting
        return nodes
    
    def get_stats(self):
        """Get current statistics for debugging"""
        return {
            'stored_nodes_count': len(self.stored_nodes),
            'pre_priority_nodes_count': len(self.pre_priority_nodes),
            'pre_priority_frequencies': {k: v[1] for k, v in self.pre_priority_nodes.items()}
        }
    