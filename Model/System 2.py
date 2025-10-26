# system 2 with in-built simulation & brain regions
# claude code
"""
Virtual Person: 2D Stick Figure + Live Camera/Microphone Integration

Features:
- Webcam frames via OpenCV
- Microphone to text via SpeechRecognition (Google or local recognizer if configured)
- Text-to-speech via pyttsx3
- 2D stick figure drawn/animated in pygame
- Threaded pipeline sending sensory input to your model and executing returned action codes

How to run:
1) Install dependencies (Python 3.9+ recommended)
   pip install pygame opencv-python SpeechRecognition pyttsx3 pyaudio

   Note: On some systems, you may need portaudio installed for pyaudio.
   - macOS: brew install portaudio
   - Ubuntu/Debian: sudo apt-get install portaudio19-dev && pip install pyaudio
   - Windows: install PyAudio wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio if pip fails

2) Run the script:
   python virtual_person.py

3) Quit:
   Close the pygame window or press ESC.
"""
import random
# Graphics / IO
import pygame
import cv2

# Audio: STT + TTS
import speech_recognition as sr
import pyttsx3
import numpy as np

# Added missing imports
import threading
import queue
import time
import math
import json
import os
from typing import Dict, Any, List

from shared_config import SharedConfig
#from Simulation import System2AnimationEngine

from typing import Dict, Any, List, Optional
from collections import defaultdict

############################################################
# Config
############################################################

# Pygame window size
WIN_W, WIN_H = 800, 500

# Target FPS for drawing
FPS = 30

# Camera index (0 is usually default webcam)
CAMERA_INDEX = 0

# STT settings
USE_GOOGLE_ONLINE_STT = True       # set False if you plan to plug in offline recognizer
ENERGY_THRESHOLD = 300              # adjust if your mic is too sensitive or too quiet
PAUSE_THRESHOLD = 0.8               # seconds of silence to consider phrase complete
PHRASE_TIME_LIMIT = 5               # max seconds to listen per phrase

# Model trigger rate
MODEL_MIN_INTERVAL = 0.0           # seconds between model calls (avoid overloading)

############################################################
# Loading outputs from System 1
############################################################

def load_system_output():
    """
    Load system output from Motor and Speech output directories.
    Returns a list of action dictionaries with type annotations.
    """
    import os
    import json
    import re
    
    motor_output_dir = r"D:\artist\brainX\CRX\Properties\Motor_output"
    speech_output_dir = r"D:\artist\brainX\CRX\Properties\Speech_output"
    
    all_actions = []
    
    # Process both directories
    for output_dir, action_type in [(motor_output_dir, "move"), (speech_output_dir, "say")]:
        if not os.path.exists(output_dir):
            continue
        
        # Get JSON files with serial numbers
        output_files = [f for f in os.listdir(output_dir) 
                       if f.endswith('.json') and re.match(r'.*_\d{3}\.json$', f)]
        
        if not output_files:
            continue
        
        # Sort files by serial number (extract number from filename)
        def extract_serial(filename):
            match = re.search(r'_(\d{3})\.json$', filename)
            return int(match.group(1)) if match else 0
        
        sorted_files = sorted(output_files, key=extract_serial)
        
        # Process each file in order
        for filename in sorted_files:
            file_path = os.path.join(output_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Add type field to each action in the file
                if isinstance(data, list):
                    for action in data:
                        if isinstance(action, dict):
                            action['type'] = action_type
                            all_actions.append(action)
                elif isinstance(data, dict):
                    # If it's a single action dict
                    if 'actions' in data and isinstance(data['actions'], list):
                        for action in data['actions']:
                            if isinstance(action, dict):
                                action['type'] = action_type
                                all_actions.append(action)
                    else:
                        # Single action
                        data['type'] = action_type
                        all_actions.append(data)
                
                # Remove the file after reading to avoid processing it again
                os.remove(file_path)
                
            except Exception as e:
                print(f"Error loading file {filename}: {e}")
                continue
    
    return all_actions

def retrieve_json():
    """
    Retrieve the next JSON file based on a retrieval counter.
    Uses 'retrieve_counter.txt' to track which file to read next.

    Args:
        directory (str): Folder path where files are stored.
        base_name (str): Base name for the stored JSON files.

    Returns:
        dict or None: The loaded JSON data, or None if no new file exists.
    """
    directory = r"D:\artist\brainX\CRX\Properties\s1_outputs"
    counter_file = os.path.join(directory, "file_counter.txt")
    retrieve_counter_file = os.path.join(directory, "retrieve_counter.txt")

    if not os.path.exists(counter_file):
        print("[RETRIEVE] No stored files yet.")
        return None

    try:
        with open(counter_file, "r") as f:
            last_store_idx = int(f.read().strip())
    except ValueError:
        last_store_idx = 0

    if os.path.exists(retrieve_counter_file):
        try:
            with open(retrieve_counter_file, "r") as f:
                last_retrieve_idx = int(f.read().strip())
        except ValueError:
            last_retrieve_idx = 0
    else:
        last_retrieve_idx = 0

    next_idx = last_retrieve_idx + 1
    if next_idx > last_store_idx:
        print("[RETRIEVE] No new JSON files to retrieve.")
        return None

    json_filename = f"s1_{next_idx:03d}.json"
    json_path = os.path.join(directory, json_filename)

    if not os.path.exists(json_path):
        print(f"[RETRIEVE] File not found: {json_filename}")
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Delete file after reading
    try:
        os.remove(json_path)
        print(f"[RETRIEVE] Removed JSON after reading: {json_path}")
    except Exception as e:
        print(f"[RETRIEVE] Warning: failed to remove file: {e}")

    # Update retrieval counter
    with open(retrieve_counter_file, "w") as f:
        f.write(str(next_idx))

    print(f"[RETRIEVE] Loaded JSON: {json_path}")
    return data

############################################################
# simulation + Mapper + Regions
############################################################
import pygame
import threading
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional
import queue
import os

import multiprocessing as mp
from multiprocessing import Process, Queue

from collections import deque
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional

# Initialize Pygame (only if not already initialized)
if not pygame.get_init():
    pygame.init()

# Constants for Simulation Window
SIM_WIDTH, SIM_HEIGHT = 650, 700
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (100, 150, 255)
GREEN = (100, 255, 150)
RED = (255, 100, 100)
YELLOW = (255, 200, 100)
PURPLE = (200, 100, 255)
GRAY = (200, 200, 200)
ORANGE = (255, 165, 0)
CYAN = (100, 200, 255)

# Node positions
CENTER = (SIM_WIDTH // 2, SIM_HEIGHT // 2.5)
RADIUS = 180
NODE_RADIUS = 45
STORAGE_RADIUS = 60

@dataclass
class Particle:
    """Represents a data particle traveling between nodes"""
    start: Tuple[int, int]
    end: Tuple[int, int]
    progress: float
    color: Tuple[int, int, int]
    data: str
    particle_type: str  # 'input', 'output', 'mapping'

class Node:
    """Represents a sensory/output node"""
    def __init__(self, name: str, position: Tuple[int, int], color: Tuple[int, int, int]):
        self.name = name
        self.position = position
        self.color = color
        self.activity = 0  # Animation activity level
        self.data_count = 0
        
class Storage:
    """Central storage node (System1) that processes and routes data"""
    def __init__(self, position: Tuple[int, int]):
        self.position = position
        self.input_queue = deque(maxlen=100)
        self.output_queue = deque(maxlen=100)
        self.activity = 0
        self.processing = False
        self.lock = threading.Lock()
        
    def receive_input(self, source: str, data: str):
        """Receive input from a node"""
        with self.lock:
            self.input_queue.append((source, data, time.time()))
            self.activity = min(self.activity + 30, 100)
            self.processing = True
            
    def process_and_send(self, target: str, data: str):
        """Process and send data to a target node"""
        with self.lock:
            self.output_queue.append((target, data, time.time()))
            self.activity = min(self.activity + 30, 100)
            
    def update_activity(self):
        """Decay activity over time"""
        with self.lock:
            if self.activity > 0:
                self.activity = max(0, self.activity - 1.5)
            if self.activity < 10:
                self.processing = False

class MappingRoute:
    """Represents an active mapping route"""
    def __init__(self, source: str, target: str, route_type: str):
        self.source = source
        self.target = target
        self.route_type = route_type  # 'visual->hearing' or 'hearing->visual'
        self.active = True
        self.cooldown = 0
        self.timestamp = time.time()

class System2AnimationEngine:
    """
    Real-time animation engine for System2 visualization.
    Runs in a completely separate PROCESS (not thread).
    """
    
    def __init__(self, event_queue: mp.Queue):
        """
        Initialize animation engine with separate process communication.
        
        Args:
            event_queue: Multiprocessing Queue for receiving events from System2
        """
        # Initialize Pygame in THIS process
        pygame.init()
        print("animation starting")
        
        # Set window position
        os.environ['SDL_VIDEO_WINDOW_POS'] = "100,100"
        
        # Create display surface
        self.screen = pygame.display.set_mode((SIM_WIDTH, SIM_HEIGHT))
        pygame.display.set_caption("System2 Neural Simulation - Brain Regions")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 26)
        self.small_font = pygame.font.Font(None, 23)
        self.tiny_font = pygame.font.Font(None, 16)
        
        # Event queue from System2 (multiprocessing Queue)
        self.event_queue = event_queue
        
        # Calculate node positions in a circle around center
        self.nodes = {
            'Eye': Node('Eye', self._calculate_position(0, 4), BLUE),
            'Ear': Node('Ear', self._calculate_position(1, 4), GREEN),
            'Stickfigure': Node('Motor', self._calculate_position(2, 4), RED),
            'Speech': Node('Speech', self._calculate_position(3, 4), YELLOW)
        }
        
        self.storage = Storage(CENTER)
        self.particles: List[Particle] = []
        self.mapping_routes: List[MappingRoute] = []
        self.running = True
        self.paused = False
        
        # Statistics
        self.stats = {
            'visual_inputs': 0,
            'audio_inputs': 0,
            'mappings_created': 0,
            'routes_active': 0,
            'total_particles': 0,
            'uptime': time.time()
        }
        
        # Control buttons
        self.buttons = self._create_control_buttons()
    
    def _calculate_position(self, index: int, total: int) -> Tuple[int, int]:
        """Calculate position of node in circle"""
        angle = (2 * math.pi * index / total) - math.pi / 2
        x = CENTER[0] + RADIUS * math.cos(angle)
        y = CENTER[1] + RADIUS * math.sin(angle)
        return (int(x), int(y))
    
    def _create_control_buttons(self):
        """Create control buttons for the simulation"""
        button_y = 10
        button_width = 90
        button_height = 35
        spacing = 10
        
        buttons = {
            'pause': pygame.Rect(SIM_WIDTH - (button_width + spacing) * 3, button_y, button_width, button_height),
            'reset': pygame.Rect(SIM_WIDTH - (button_width + spacing) * 2, button_y, button_width, button_height),
            'close': pygame.Rect(SIM_WIDTH - (button_width + spacing), button_y, button_width, button_height)
        }
        return buttons
    
    def process_events(self):
        """Process events from the multiprocessing queue (non-blocking)"""
        if self.paused:
            return
        
        # Process multiple events per frame for better responsiveness
        for _ in range(10):  # Process up to 10 events per frame
            try:
                # Non-blocking get with timeout
                event = self.event_queue.get_nowait()
                self._handle_event(event)
            except:
                break  # Queue is empty
    
    def _handle_event(self, event: Dict):
        """Handle individual events"""
        try:
            event_type = event['type']
            source = event['source']
            target = event.get('target')
            data = event.get('data', '')
            
            if event_type == 'sensory_input':
                self._animate_input_to_storage(source, data)
                
                if source == 'Eye':
                    self.stats['visual_inputs'] += 1
                elif source == 'Ear':
                    self.stats['audio_inputs'] += 1
                    
            elif event_type == 'system1_output':
                self._animate_storage_to_output(target, data)
                
            elif event_type == 'mapping_route':
                route_info = event.get('route_info', {})
                self._animate_mapping_route(source, target, route_info)
                self.stats['mappings_created'] += 1
                
            elif event_type == 'mapping_retrieval':
                route_info = event.get('route_info', {})
                self._animate_cross_modal_retrieval(source, target, route_info)
                
        except Exception as e:
            print(f"[Animation] Error handling event: {e}")
    
    def _animate_input_to_storage(self, source: str, data: str):
        """Animate sensory input traveling to storage"""
        if source in self.nodes:
            node = self.nodes[source]
            node.activity = 100
            node.data_count += 1
            self.storage.receive_input(source, data)
            
            particle = Particle(
                start=node.position,
                end=self.storage.position,
                progress=0.0,
                color=node.color,
                data=f"{source}: {data[:20]}...",
                particle_type='input'
            )
            self.particles.append(particle)
            self.stats['total_particles'] += 1
    
    def _animate_storage_to_output(self, target: str, data: str):
        """Animate data from storage to output node"""
        if target in self.nodes:
            node = self.nodes[target]
            self.storage.process_and_send(target, data)
            
            particle = Particle(
                start=self.storage.position,
                end=node.position,
                progress=0.0,
                color=node.color,
                data=f"To {target}: {data[:20]}...",
                particle_type='output'
            )
            self.particles.append(particle)
            self.stats['total_particles'] += 1
    
    def _animate_mapping_route(self, source: str, target: str, route_info: Dict):
        """Animate mapping route creation"""
        route_type = route_info.get('route_type', 'unknown')
        mapping_route = MappingRoute(source, target, route_type)
        self.mapping_routes.append(mapping_route)
        
        if source in self.nodes and target in self.nodes:
            source_node = self.nodes[source]
            target_node = self.nodes[target]
            
            particle = Particle(
                start=source_node.position,
                end=target_node.position,
                progress=0.0,
                color=PURPLE,
                data=f"Mapping: {source} <-> {target}",
                particle_type='mapping'
            )
            self.particles.append(particle)
            self.stats['total_particles'] += 1
    
    def _animate_cross_modal_retrieval(self, source: str, target: str, route_info: Dict):
        """Animate cross-modal retrieval"""
        function_node = route_info.get('function_node')
        
        if source in self.nodes and function_node in self.nodes:
            source_node = self.nodes[source]
            func_node = self.nodes[function_node]
            
            particle1 = Particle(
                start=source_node.position,
                end=func_node.position,
                progress=0.0,
                color=ORANGE,
                data=f"Query: {source}",
                particle_type='mapping'
            )
            self.particles.append(particle1)
            self.stats['total_particles'] += 1
    
    def update_particles(self):
        """Update particle positions"""
        if self.paused:
            return
        
        particles_to_remove = []
        
        for particle in self.particles:
            particle.progress += 0.018
            
            if particle.progress >= 1.0:
                particles_to_remove.append(particle)
                if particle.particle_type == 'input':
                    self.storage.activity = min(self.storage.activity + 20, 100)
        
        for particle in particles_to_remove:
            self.particles.remove(particle)
    
    def update_mapping_routes(self):
        """Update and clean up mapping routes"""
        if self.paused:
            return
        
        routes_to_remove = []
        
        for route in self.mapping_routes:
            if time.time() - route.timestamp > 15:
                routes_to_remove.append(route)
        
        for route in routes_to_remove:
            self.mapping_routes.remove(route)
        
        self.stats['routes_active'] = len(self.mapping_routes)

    def draw_control_buttons(self):
        """Draw control buttons"""
        mouse_pos = pygame.mouse.get_pos()
        
        # Pause/Resume button
        pause_color = (100, 200, 100) if self.paused else (255, 200, 100)
        if self.buttons['pause'].collidepoint(mouse_pos):
            pause_color = tuple(min(c + 30, 255) for c in pause_color)
        pygame.draw.rect(self.screen, pause_color, self.buttons['pause'], border_radius=5)
        pygame.draw.rect(self.screen, BLACK, self.buttons['pause'], 2, border_radius=5)
        pause_text = self.tiny_font.render("Resume" if self.paused else "Pause", True, BLACK)
        pause_rect = pause_text.get_rect(center=self.buttons['pause'].center)
        self.screen.blit(pause_text, pause_rect)
        
        # Reset button
        reset_color = (200, 200, 255)
        if self.buttons['reset'].collidepoint(mouse_pos):
            reset_color = (230, 230, 255)
        pygame.draw.rect(self.screen, reset_color, self.buttons['reset'], border_radius=5)
        pygame.draw.rect(self.screen, BLACK, self.buttons['reset'], 2, border_radius=5)
        reset_text = self.tiny_font.render("Reset", True, BLACK)
        reset_rect = reset_text.get_rect(center=self.buttons['reset'].center)
        self.screen.blit(reset_text, reset_rect)
        
        # Close button
        close_color = (255, 100, 100)
        if self.buttons['close'].collidepoint(mouse_pos):
            close_color = (255, 130, 130)
        pygame.draw.rect(self.screen, close_color, self.buttons['close'], border_radius=5)
        pygame.draw.rect(self.screen, BLACK, self.buttons['close'], 2, border_radius=5)
        close_text = self.tiny_font.render("Close", True, BLACK)
        close_rect = close_text.get_rect(center=self.buttons['close'].center)
        self.screen.blit(close_text, close_rect)
    
    def handle_button_click(self, pos):
        """Handle button clicks"""
        if self.buttons['pause'].collidepoint(pos):
            self.paused = not self.paused
            print(f"[Animation] {'Paused' if self.paused else 'Resumed'}")
        elif self.buttons['reset'].collidepoint(pos):
            self.reset_simulation()
        elif self.buttons['close'].collidepoint(pos):
            self.running = False
    
    def reset_simulation(self):
        """Reset simulation"""
        self.particles.clear()
        self.mapping_routes.clear()
        self.stats = {
            'visual_inputs': 0,
            'audio_inputs': 0,
            'mappings_created': 0,
            'routes_active': 0,
            'total_particles': 0,
            'uptime': time.time()
        }
        for node in self.nodes.values():
            node.data_count = 0
            node.activity = 0
        self.storage.activity = 0
        self.storage.processing = False
        print("[Animation] Simulation reset")
    
    def draw_node(self, node: Node):
        """Draw a sensory/output node"""
        pygame.draw.circle(self.screen, node.color, node.position, NODE_RADIUS, 4)
        
        if node.activity > 0:
            glow_radius = NODE_RADIUS + int(node.activity / 4)
            glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            alpha = int(node.activity * 0.6)
            pygame.draw.circle(glow_surface, (*node.color, alpha), 
                             (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surface, 
                           (node.position[0] - glow_radius, node.position[1] - glow_radius))
            node.activity = max(0, node.activity - 2)
        
        text = self.small_font.render(node.name, True, BLACK)
        text_rect = text.get_rect(center=node.position)
        self.screen.blit(text, text_rect)
        
        if node.data_count > 0:
            badge_text = self.tiny_font.render(str(node.data_count), True, WHITE)
            badge_rect = pygame.Rect(node.position[0] + 30, node.position[1] - 40, 30, 20)
            pygame.draw.rect(self.screen, RED, badge_rect, border_radius=10)
            badge_text_rect = badge_text.get_rect(center=badge_rect.center)
            self.screen.blit(badge_text, badge_text_rect)
        
        pygame.draw.line(self.screen, GRAY, node.position, self.storage.position, 1)
    
    def draw_storage(self):
        """Draw the central storage node (System1)"""
        pygame.draw.circle(self.screen, BLACK, self.storage.position, STORAGE_RADIUS, 5)
        
        if self.storage.activity > 0:
            pulse_radius = STORAGE_RADIUS + int(self.storage.activity / 2.5)
            glow_surface = pygame.Surface((pulse_radius * 2, pulse_radius * 2), pygame.SRCALPHA)
            alpha = int(self.storage.activity * 0.9)
            pygame.draw.circle(glow_surface, (100, 100, 255, alpha), 
                             (pulse_radius, pulse_radius), pulse_radius)
            self.screen.blit(glow_surface, 
                           (self.storage.position[0] - pulse_radius, 
                            self.storage.position[1] - pulse_radius))
        
        if self.storage.processing:
            angle = (time.time() * 5) % (2 * math.pi)
            indicator_radius = STORAGE_RADIUS - 15
            for i in range(8):
                indicator_angle = angle + (i * math.pi / 4)
                x = self.storage.position[0] + indicator_radius * math.cos(indicator_angle)
                y = self.storage.position[1] + indicator_radius * math.sin(indicator_angle)
                alpha = int(255 * (1 - i / 8))
                dot_surface = pygame.Surface((12, 12), pygame.SRCALPHA)
                pygame.draw.circle(dot_surface, (0, 255, 255, alpha), (6, 6), 4)
                self.screen.blit(dot_surface, (int(x) - 6, int(y) - 6))
        
        self.storage.update_activity()
        
        text = self.font.render("System1", True, BLACK)
        text_rect = text.get_rect(center=self.storage.position)
        self.screen.blit(text, text_rect)
        
        queue_text = self.tiny_font.render(
            f"In:{len(self.storage.input_queue)} Out:{len(self.storage.output_queue)}", 
            True, WHITE
        )
        queue_rect = queue_text.get_rect(center=(self.storage.position[0], 
                                                  self.storage.position[1] + 25))
        self.screen.blit(queue_text, queue_rect)
    
    def draw_particle(self, particle: Particle):
        """Draw a data particle"""
        t = particle.progress
        t = t * t * (3 - 2 * t)
        
        x = particle.start[0] + (particle.end[0] - particle.start[0]) * t
        y = particle.start[1] + (particle.end[1] - particle.start[1]) * t
        
        size = 10 if particle.particle_type == 'mapping' else 8
        
        pygame.draw.circle(self.screen, particle.color, (int(x), int(y)), size)
        pygame.draw.circle(self.screen, WHITE, (int(x), int(y)), size - 4)
        
        trail_length = 6
        for i in range(1, trail_length):
            trail_t = max(0, t - i * 0.04)
            trail_x = particle.start[0] + (particle.end[0] - particle.start[0]) * trail_t
            trail_y = particle.start[1] + (particle.end[1] - particle.start[1]) * trail_t
            alpha = int(180 * (1 - i / trail_length))
            trail_surface = pygame.Surface((14, 14), pygame.SRCALPHA)
            pygame.draw.circle(trail_surface, (*particle.color, alpha), (7, 7), 5)
            self.screen.blit(trail_surface, (int(trail_x) - 7, int(trail_y) - 7))
    
    def draw_mapping_routes(self):
        """Draw active mapping routes"""
        for route in self.mapping_routes:
            if route.source in self.nodes and route.target in self.nodes:
                source_pos = self.nodes[route.source].position
                target_pos = self.nodes[route.target].position
                
                steps = 20
                for i in range(steps):
                    if i % 2 == 0:
                        t1 = i / steps
                        t2 = (i + 1) / steps
                        x1 = source_pos[0] + (target_pos[0] - source_pos[0]) * t1
                        y1 = source_pos[1] + (target_pos[1] - source_pos[1]) * t1
                        x2 = source_pos[0] + (target_pos[0] - source_pos[0]) * t2
                        y2 = source_pos[1] + (target_pos[1] - source_pos[1]) * t2
                        
                        pygame.draw.line(self.screen, PURPLE, 
                                       (int(x1), int(y1)), (int(x2), int(y2)), 2)
    
    def draw_stats_panel(self):
        """Draw statistics panel"""
        panel_height = 140
        panel_rect = pygame.Rect(0, SIM_HEIGHT - panel_height, SIM_WIDTH, panel_height)
        
        panel_surface = pygame.Surface((SIM_WIDTH, panel_height), pygame.SRCALPHA)
        pygame.draw.rect(panel_surface, (240, 240, 240, 240), panel_surface.get_rect())
        self.screen.blit(panel_surface, (0, SIM_HEIGHT - panel_height))
        
        pygame.draw.line(self.screen, BLACK, (0, SIM_HEIGHT - panel_height), 
                        (SIM_WIDTH, SIM_HEIGHT - panel_height), 3)
        
        uptime = int(time.time() - self.stats['uptime'])
        
        stats_text = [
            f"System2 Real-Time Impulse Simulation",
            f"Visual Inputs: {self.stats['visual_inputs']} | Audio Inputs: {self.stats['audio_inputs']} | Uptime: {uptime}s",
            f"Mappings Created: {self.stats['mappings_created']} | Active Routes: {self.stats['routes_active']}",
            f"Active Particles: {len(self.particles)} | Total Processed: {self.stats['total_particles']}",
            f"System1 Activity: {int(self.storage.activity)}% | Status: {'PAUSED' if self.paused else 'Processing' if self.storage.processing else 'Idle'}"
        ]
        
        y_offset = SIM_HEIGHT - panel_height + 10
        for i, line in enumerate(stats_text):
            if i == 0:
                text = self.font.render(line, True, BLACK)
            else:
                text = self.small_font.render(line, True, BLACK)
            self.screen.blit(text, (10, y_offset))
            y_offset += 26
    
    def draw_legend(self):
        """Draw color legend"""
        legend_x = 10
        legend_y = 50
        
        legend_items = [
            ("Eye (Visual)", BLUE),
            ("Ear (Audio)", GREEN),
            ("Mapping", PURPLE),
            ("Retrieval", ORANGE)
        ]
        
        for i, (label, color) in enumerate(legend_items):
            pygame.draw.rect(self.screen, color, 
                           (legend_x, legend_y + i * 25, 20, 20))
            pygame.draw.rect(self.screen, BLACK, 
                           (legend_x, legend_y + i * 25, 20, 20), 2)
            
            text = self.tiny_font.render(label, True, BLACK)
            self.screen.blit(text, (legend_x + 25, legend_y + i * 25 + 3))
     
    def run(self):
        """Main animation loop - runs in separate PROCESS"""
        print("[System2 Animation] Starting simulation window in separate process...")
        
        try:
            while self.running:
                # Initialize Pygame in THIS process
                pygame.init()
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:
                            self.handle_button_click(event.pos)
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                        elif event.key == pygame.K_SPACE:
                            self.paused = not self.paused
                        elif event.key == pygame.K_r:
                            self.reset_simulation()
                
                # Process events from System2
                self.process_events()
                
                # Update
                self.update_particles()
                self.update_mapping_routes()
                
                # Draw
                self.screen.fill(WHITE)
                self.draw_mapping_routes()
                
                for node in self.nodes.values():
                    self.draw_node(node)
                
                self.draw_storage()
                
                for particle in self.particles:
                    self.draw_particle(particle)
                
                self.draw_control_buttons()
                self.draw_stats_panel()
                self.draw_legend()
                
                pygame.display.flip()
                self.clock.tick(FPS)
        
        except Exception as e:
            print(f"[Animation] Error in animation loop: {e}")
        finally:
            print("[System2 Animation] Simulation window closed")
            pygame.quit()

class MappingCoordinator:
    """
    Coordinates cross-modal mappings between visual and hearing inputs.
    Creates bidirectional routes to retrieve associated inputs across modalities.
    """
    
    def __init__(self, map_storage_dir: str):
        """
        Initialize the mapping coordinator.
        
        Args:
            map_storage_dir: Directory to store mapping data and counters
        """
        self.map_storage_dir = map_storage_dir
        os.makedirs(map_storage_dir, exist_ok=True)
        
        # Mapping dictionary: stores routes between visual and hearing
        # Format: {
        #   "visual_id": {"hearing_ref": "hearing_id", "last_used": timestamp, "cooldown": False},
        #   "hearing_id": {"visual_ref": "visual_id", "last_used": timestamp, "cooldown": False}
        # }
        self.mapping_dict = self._load_mapping_dict()
        
        # Track route usage for cooldown (2 iterations)
        self.route_usage_counter = defaultdict(int)
        self.cooldown_iterations = 2
        
    def _load_mapping_dict(self) -> Dict[str, Dict[str, Any]]:
        """Load existing mapping dictionary from disk."""
        map_file = os.path.join(self.map_storage_dir, "mapping_dict.json")
        if os.path.exists(map_file):
            try:
                with open(map_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[MappingCoordinator] Error loading mapping dict: {e}")
                return {}
        return {}
    
    def _save_mapping_dict(self):
        """Save mapping dictionary to disk."""
        map_file = os.path.join(self.map_storage_dir, "mapping_dict.json")
        try:
            with open(map_file, "w", encoding="utf-8") as f:
                json.dump(self.mapping_dict, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[MappingCoordinator] Error saving mapping dict: {e}")
    
    def create_mapping(self, visual_id: str, hearing_id: str):
        """
        Create bidirectional mapping between visual and hearing inputs.
        
        Args:
            visual_id: Identifier for visual input (e.g., "frame_001")
            hearing_id: Identifier for hearing input (e.g., "hearing_001")
        """
        # Create dual routes
        self.mapping_dict[visual_id] = {
            "hearing_ref": hearing_id,
            "last_used": None,
            "cooldown": False,
            "route": "visual->hearing_fn->hearing_output->s1"
        }
        
        self.mapping_dict[hearing_id] = {
            "visual_ref": visual_id,
            "last_used": None,
            "cooldown": False,
            "route": "hearing->visual_fn->visual_output->s1"
        }
        
        self._save_mapping_dict()
        print(f"[MappingCoordinator] Created bidirectional mapping: {visual_id} <-> {hearing_id}")
    
    def check_and_route(self, input_id: str, input_type: str) -> Optional[Dict[str, Any]]:
        """
        Check if input matches mapping dictionary and return routing information.
        
        Args:
            input_id: Identifier of the input to check
            input_type: Type of input ("visual" or "hearing")
        
        Returns:
            Dict with routing info if match found and route available, None otherwise
        """
        if input_id not in self.mapping_dict:
            print(f"[MappingCoordinator] No mapping found for {input_id}")
            return None
        
        mapping_entry = self.mapping_dict[input_id]
        
        # Check cooldown status
        if mapping_entry.get("cooldown", False):
            print(f"[MappingCoordinator] Route for {input_id} is in cooldown period")
            return None
        
        # Prepare routing information
        if input_type == "visual":
            ref_key = "hearing_ref"
            target_type = "hearing"
            function_name = "hearing_function"
        else:  # hearing
            ref_key = "visual_ref"
            target_type = "visual"
            function_name = "visual_function"
        
        if ref_key not in mapping_entry:
            print(f"[MappingCoordinator] No {ref_key} found in mapping for {input_id}")
            return None
        
        routing_info = {
            "source_id": input_id,
            "source_type": input_type,
            "target_id": mapping_entry[ref_key],
            "target_type": target_type,
            "function_name": function_name,
            "route": mapping_entry["route"]
        }
        
        # Mark route as used and set cooldown
        self._mark_route_used(input_id)
        
        return routing_info
    
    def _mark_route_used(self, input_id: str):
        """Mark a route as used and apply cooldown."""
        if input_id in self.mapping_dict:
            self.mapping_dict[input_id]["last_used"] = time.time()
            self.mapping_dict[input_id]["cooldown"] = True
            self.route_usage_counter[input_id] = 0  # Reset counter
            self._save_mapping_dict()
            print(f"[MappingCoordinator] Route {input_id} marked as used, cooldown active")
    
    def update_cooldowns(self):
        """
        Update cooldown status for all routes (call once per iteration).
        Routes become available again after cooldown_iterations.
        """
        for input_id in list(self.route_usage_counter.keys()):
            self.route_usage_counter[input_id] += 1
            
            if self.route_usage_counter[input_id] >= self.cooldown_iterations:
                if input_id in self.mapping_dict:
                    self.mapping_dict[input_id]["cooldown"] = False
                    print(f"[MappingCoordinator] Route {input_id} cooldown ended")
                del self.route_usage_counter[input_id]
        
        self._save_mapping_dict()

class RegionsFunction:
    """
    Stores and retrieves cross-modal input associations.
    Acts as memory for visual-hearing pairs.
    """
    
    def __init__(self, storage_dir: str):
        """
        Initialize regions function storage.
        
        Args:
            storage_dir: Directory to store region associations
        """
        self.storage_dir = storage_dir
        self.visual_dir = os.path.join(storage_dir, "visual_regions")
        self.hearing_dir = os.path.join(storage_dir, "hearing_regions")
        
        os.makedirs(self.visual_dir, exist_ok=True)
        os.makedirs(self.hearing_dir, exist_ok=True)
    
    def visual_function(self, hearing_id: str) -> Optional[str]:
        """
        Retrieve visual input associated with hearing input.
        
        Args:
            hearing_id: Hearing input identifier
        
        Returns:
            Path to associated visual data, or None if not found
        """
        # Load hearing region data to find associated visual
        hearing_file = os.path.join(self.hearing_dir, f"{hearing_id}.json")
        
        if not os.path.exists(hearing_file):
            print(f"[RegionsFunction] Hearing region file not found: {hearing_id}")
            return None
        
        try:
            with open(hearing_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                visual_ref = data.get("visual_ref")
                
                if visual_ref:
                    visual_path = os.path.join(self.visual_dir, f"{visual_ref}.json")
                    if os.path.exists(visual_path):
                        print(f"[RegionsFunction] Retrieved visual {visual_ref} for hearing {hearing_id}")
                        return visual_path
        except Exception as e:
            print(f"[RegionsFunction] Error in visual_function: {e}")
        
        return None
    
    def hearing_function(self, visual_id: str) -> Optional[str]:
        """
        Retrieve hearing input associated with visual input.
        
        Args:
            visual_id: Visual input identifier
        
        Returns:
            Path to associated hearing data, or None if not found
        """
        # Load visual region data to find associated hearing
        visual_file = os.path.join(self.visual_dir, f"{visual_id}.json")
        
        if not os.path.exists(visual_file):
            print(f"[RegionsFunction] Visual region file not found: {visual_id}")
            return None
        
        try:
            with open(visual_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                hearing_ref = data.get("hearing_ref")
                
                if hearing_ref:
                    hearing_path = os.path.join(self.hearing_dir, f"{hearing_ref}.json")
                    if os.path.exists(hearing_path):
                        print(f"[RegionsFunction] Retrieved hearing {hearing_ref} for visual {visual_id}")
                        return hearing_path
        except Exception as e:
            print(f"[RegionsFunction] Error in hearing_function: {e}")
        
        return None
    
    def store_association(self, visual_id: str, hearing_id: str, 
                        visual_data: Any, hearing_data: Any):
        """
        Store bidirectional association between visual and hearing inputs.
        
        Args:
            visual_id: Visual input identifier
            hearing_id: Hearing input identifier
            visual_data: Visual input data
            hearing_data: Hearing input data
        """
        # Store visual region with hearing reference
        visual_file = os.path.join(self.visual_dir, f"{visual_id}.json")
        with open(visual_file, "w", encoding="utf-8") as f:
            json.dump({
                "visual_id": visual_id,
                "hearing_ref": hearing_id,
                "data": visual_data,
                "timestamp": time.time()
            }, f, ensure_ascii=False, indent=2)
        
        # Store hearing region with visual reference
        hearing_file = os.path.join(self.hearing_dir, f"{hearing_id}.json")
        with open(hearing_file, "w", encoding="utf-8") as f:
            json.dump({
                "hearing_id": hearing_id,
                "visual_ref": visual_id,
                "data": hearing_data,
                "timestamp": time.time()
            }, f, ensure_ascii=False, indent=2)
        
        print(f"[RegionsFunction] Stored association: {visual_id} <-> {hearing_id}")

# Standalone function to run in separate process
def run_animation_process(event_queue: mp.Queue):
    """
    Function to run in separate process.
    This is necessary because you can't pickle class methods easily.
    """
    engine = System2AnimationEngine(event_queue)
    engine.run()

###########################################################
# Model that co-ordinates the inputs
# keyword -> action mapper
###########################################################

class System2:
    """
    Model that:
    - Saves raw sensory inputs (vision frames + hearing text)
    - Reads processed outputs from System 1 (motor_output + speech_output)
    - Assigns new actions based on System 1 outputs
    - with realtime animation
    """
    def __init__(self):
        self.running = True
        self._last_text = None
        self._shutdown_event = threading.Event()  # Added shutdown event
        self.shared_config = SharedConfig()  # Add shared config
        self.training_mode = False

    def set_training_mode(self, training_enabled):
        """Enable/disable training mode - pauses physical camera capture during training"""
        self.training_mode = training_enabled

    def stop(self):
        """Stop System2 gracefully."""
        self.running = False
        self._shutdown_event.set()  # Signal shutdown to any waiting threads
        return "System2 stopped"

    # Modified System2 process method with mapping integration
    def process(self, sensory_input: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Check shutdown at the beginning of each process call
        if not self.running or self._shutdown_event.is_set():
            print("System2 stopped for shutdown.")
            return []
        actions = []

        # Initialize animation engine if not exists (runs in separate PROCESS)
        if not hasattr(self, 'animation_process'):
            print("[System2] Initializing animation in separate process...")
            
            # Create multiprocessing Queue for communication
            self.animation_queue = mp.Queue(maxsize=1000)
            
            # Create and start separate process
            self.animation_process = mp.Process(
                target=run_animation_process,
                args=(self.animation_queue,),
                daemon=True,
                name="System2AnimationProcess"
            )
            self.animation_process.start()
            print(f"[System2] Animation process started (PID: {self.animation_process.pid})")
            
            # Give it a moment to initialize
            time.sleep(0.5)
            
        # Initialize mapping coordinator and regions function if not exists
        if not hasattr(self, 'mapping_coordinator'):
            mapping_dir = r"D:\artist\brainX\CRX\Properties\Mappings"
            self.mapping_coordinator = MappingCoordinator(mapping_dir)
            
            regions_dir = r"D:\artist\brainX\CRX\Properties\Regions"
            self.regions_function = RegionsFunction(regions_dir)
        
        # Update cooldowns at start of each iteration
        self.mapping_coordinator.update_cooldowns()
        
        # --- Step 1: Save raw sensory inputs ---
        vision_dir = r"D:\artist\brainX\CRX\Properties\Eye_input"
        hearing_dir = r"D:\artist\brainX\CRX\Properties\Ear_input"
        latest_dir = r"D:\artist\brainX\CRX\Properties\latest_images"

        os.makedirs(vision_dir, exist_ok=True)
        os.makedirs(hearing_dir, exist_ok=True)
        os.makedirs(latest_dir, exist_ok=True)

        # Track current input IDs for mapping
        current_visual_id = None
        current_hearing_id = None

        # Save image
        vision_bgr = sensory_input.get("vision_bgr", None)
        if vision_bgr is not None and isinstance(vision_bgr, np.ndarray) and vision_bgr.size > 0:
            vision_counter_file = os.path.join(vision_dir, "last_image_counter.txt")

            if not os.path.exists(vision_counter_file):
                with open(vision_counter_file, "w") as f:
                    f.write("0")
                last_idx = 0
            else:
                try:
                    with open(vision_counter_file, "r") as f:
                        last_idx = int(f.read().strip())
                except ValueError:
                    last_idx = 0

            new_idx = last_idx + 1
            image_filename = f"frame_{new_idx:03d}.png"
            current_visual_id = f"frame_{new_idx:03d}"  # Store ID for mapping
            
            # 🔥 Send event to animation process (non-blocking)
            try:
                self.animation_queue.put_nowait({
                    'type': 'sensory_input',
                    'source': 'Eye',
                    'data': f"Frame {new_idx:03d}",
                    'timestamp': time.time()
                })
            except:
                pass  # Queue full, skip this animation event

            image_path = os.path.join(vision_dir, image_filename)
            cv2.imwrite(image_path, sensory_input["vision_bgr"])

            latest_image_path = os.path.join(latest_dir, image_filename)
            cv2.imwrite(latest_image_path, vision_bgr)

            hear = sensory_input.get("hearing_text", None)
            if hear is not None:
                ear_ = True
            else:
                ear_ = False
            if self.training_mode:
                self.shared_config.update_sensor_status(eye=True)
                print(f"[System2] Image saved, signaled System1 (eye=True)")
            else:
                self.shared_config.update_sensor_status(eye=True, ear=ear_)
                print(f"[System2] Image saved, signaled System1 (eye=True, ear = {ear_})")

            with open(vision_counter_file, "w") as f:
                f.write(str(new_idx))
            sensory_input["vision_bgr"] = None
        
        # Save hearing text
        if "hearing_text" in sensory_input and sensory_input["hearing_text"]:
            hearing_counter_file = os.path.join(hearing_dir, "last_text_counter.txt")

            if not os.path.exists(hearing_counter_file):
                with open(hearing_counter_file, "w") as f:
                    f.write("0")
                last_idx = 0
            else:
                try:
                    with open(hearing_counter_file, "r") as f:
                        last_idx = int(f.read().strip())
                except ValueError:
                    last_idx = 0

            new_idx = last_idx + 1
            text_filename = f"hearing_{new_idx:03d}.json"
            current_hearing_id = f"hearing_{new_idx:03d}"  # Store ID for mapping
            text_path = os.path.join(hearing_dir, text_filename)

            # 🔥 Send event to animation process (non-blocking)
            try:
                self.animation_queue.put_nowait({
                    'type': 'sensory_input',
                    'source': 'Ear',
                    'data': sensory_input["hearing_text"][:50],
                    'timestamp': time.time()
                })
            except:
                pass

            with open(text_path, "w", encoding="utf-8") as f:
                json.dump({"hearing_text": sensory_input["hearing_text"]},
                        f, ensure_ascii=False, indent=2)
            
            eye = sensory_input.get("vision_bgr", None)
            if eye is not None:
                eye_ = True
            else:
                eye_ = False
            if self.training_mode:
                self.shared_config.update_sensor_status(ear=True)
                print(f"[System2] Audio saved, signaled System1 (ear=True)")
            else:
                self.shared_config.update_sensor_status(ear=True, eye=eye_)
                print(f"[System2] Audio saved, signaled System1 (ear=True, eye = {eye_})")

            with open(hearing_counter_file, "w") as f:
                f.write(str(new_idx))
            sensory_input["hearing_text"] = ""

        # --- Create mapping when both inputs are present ---
        if current_visual_id and current_hearing_id:
            self.mapping_coordinator.create_mapping(current_visual_id, current_hearing_id)
            self.regions_function.store_association(
                current_visual_id, current_hearing_id,
                {"path": image_path}, {"path": text_path}
            )
            print(f"[System2] Created cross-modal association: {current_visual_id} <-> {current_hearing_id}")

            # 🔥 Send mapping event to animation process
            try:
                self.animation_queue.put_nowait({
                    'type': 'mapping_route',
                    'source': 'Eye',
                    'target': 'Ear',
                    'data': f"Mapping: {current_visual_id} <-> {current_hearing_id}",
                    'route_info': {'route_type': 'visual<->hearing'},
                    'timestamp': time.time()
                })
            except:
                pass

        # --- Step 2: Load outputs from System 1 ---
        system_1_output = []
        max_wait = 2
        wait_interval = 1

        if not self.running or self._shutdown_event.is_set():
            print("System2 stopped for shutdown.")
            return []
        
        waited = 0
        while not system_1_output and waited < max_wait and self.running and not self._shutdown_event.is_set():
            system_1_output = load_system_output()
            print(f"system_1_output - {system_1_output}")
            if not system_1_output:
                print(f"[System2] Waiting for System 1 output... ({waited}/{max_wait}s)")
                if self._shutdown_event.wait(timeout=wait_interval):
                    print("System2 shutdown signal received during wait.")
                    return []
                waited += wait_interval

        if not self.running or self._shutdown_event.is_set():
            print("System2 stopped for shutdown.")
            return []

        if not system_1_output:
            print("[System2] No System 1 output found after waiting.")
            return actions

        # --- Step 2.5: Check outputs in mapping dictionary and route if needed ---
        print("[System2] Checking System 1 outputs against mapping dictionary...")
        system_1_output_for_regions = []
        system_1_output_for_regions = retrieve_json()
        for out in system_1_output_for_regions:
            if not self.running or self._shutdown_event.is_set():
                return actions
            
            # Try to identify input ID from output (you may need to adjust based on your output format)
            input_id = out.get("input_id") or out.get("source_id")
            input_type = out.get("input_type")  # "visual" or "hearing"
            
            if input_id and input_type:
                # Check if this output matches a mapping route
                routing_info = self.mapping_coordinator.check_and_route(input_id, input_type)
                
                if routing_info:
                    print(f"[System2] Following route: {routing_info['route']}")
                    
                    # Execute the routing
                    if routing_info["function_name"] == "hearing_function":
                        # Visual input -> retrieve associated hearing
                        hearing_path = self.regions_function.hearing_function(routing_info["source_id"])
                        if hearing_path:
                            # Load hearing data and send to S1
                            # 🔥 Trigger animation: Cross-modal retrieval (Visual -> Hearing)
                            self.animation_engine.add_event(
                                event_type='mapping_retrieval',
                                source='Eye',
                                target='Ear',
                                data="Cross-modal retrieval",
                                route_info={
                                    'function_node': 'Hearing_Function',
                                    'route_type': 'visual->hearing'
                                }
                            )
                    
                            # sending to S1
                            hearing_counter_file = os.path.join(hearing_dir, "last_text_counter.txt")

                            if not os.path.exists(hearing_counter_file):
                                with open(hearing_counter_file, "w") as f:
                                    f.write("0")
                                last_idx = 0
                            else:
                                try:
                                    with open(hearing_counter_file, "r") as f:
                                        last_idx = int(f.read().strip())
                                except ValueError:
                                    last_idx = 0

                            new_idx = last_idx + 1
                            text_filename = f"hearing_{new_idx:03d}.json"
                            current_hearing_id = f"hearing_{new_idx:03d}"  # Store ID for mapping
                            text_path = os.path.join(hearing_dir, text_filename)

                            with open(text_path, "w", encoding="utf-8") as f:
                                json.dump({"hearing_text": sensory_input["hearing_text"]},
                                        f, ensure_ascii=False, indent=2)
                            
                            eye = sensory_input.get("vision_bgr", None)
                            if eye is not None:
                                eye_ = True
                            else:
                                eye_ = False
                            if self.training_mode:
                                self.shared_config.update_sensor_status(ear=True)
                                print(f"[System2] Audio saved, signaled System1 (ear=True)")
                            else:
                                self.shared_config.update_sensor_status(ear=True, eye=eye_)
                                print(f"[System2] Audio saved, signaled System1 (ear=True, eye = {eye_})")

                            with open(hearing_counter_file, "w") as f:
                                f.write(str(new_idx))
                            sensory_input["hearing_text"] = ""

                            print(f"[System2] Routed to hearing function, retrieved: {hearing_path}")
                    
                    elif routing_info["function_name"] == "visual_function":
                        # Hearing input -> retrieve associated visual
                        visual_path = self.regions_function.visual_function(routing_info["source_id"])
                        if visual_path:
                            # Load visual data and send to S1
                            # 🔥 Trigger animation: Cross-modal retrieval (Hearing -> Visual)
                            self.animation_engine.add_event(
                                event_type='mapping_retrieval',
                                source='Ear',
                                target='Eye',
                                data="Cross-modal retrieval",
                                route_info={
                                    'function_node': 'Visual_Function',
                                    'route_type': 'hearing->visual'
                                }
                            )
        
                            # sending to S1
                            vision_counter_file = os.path.join(vision_dir, "last_image_counter.txt")

                            if not os.path.exists(vision_counter_file):
                                with open(vision_counter_file, "w") as f:
                                    f.write("0")
                                last_idx = 0
                            else:
                                try:
                                    with open(vision_counter_file, "r") as f:
                                        last_idx = int(f.read().strip())
                                except ValueError:
                                    last_idx = 0

                            new_idx = last_idx + 1
                            image_filename = f"frame_{new_idx:03d}.png"
                            current_visual_id = f"frame_{new_idx:03d}"  # Store ID for mapping
                            
                            image_path = os.path.join(vision_dir, image_filename)
                            cv2.imwrite(image_path, sensory_input["vision_bgr"])

                            latest_image_path = os.path.join(latest_dir, image_filename)
                            cv2.imwrite(latest_image_path, vision_bgr)

                            hear = sensory_input.get("hearing_text", None)
                            if hear is not None:
                                ear_ = True
                            else:
                                ear_ = False
                            if self.training_mode:
                                self.shared_config.update_sensor_status(eye=True)
                                print(f"[System2] Image saved, signaled System1 (eye=True)")
                            else:
                                self.shared_config.update_sensor_status(eye=True, ear=ear_)
                                print(f"[System2] Image saved, signaled System1 (eye=True, ear = {ear_})")

                            with open(vision_counter_file, "w") as f:
                                f.write(str(new_idx))
                            sensory_input["vision_bgr"] = None
                            
                            print(f"[System2] Routed to visual function, retrieved: {visual_path}")

        # --- Step 3: Decide new actions based on system_1_output ---
        print("[System2] Processing System 1 output...")
        for out in system_1_output:
            if not self.running or self._shutdown_event.is_set():
                return actions
            
            action_type = out.get("type", "").lower()
            
            if action_type == "say":
                text = (out.get("text") or out.get("word") or "").lower().strip()
                if text and text != self._last_text and text != "[unknown word]":
                    actions.append({"type": "SAY", "text": text})
                    self._last_text = text
            
                    # 🔥 Trigger animation: System1 output to Speech
                    self.animation_engine.add_event(
                        event_type='system1_output',
                        source='Storage',
                        target='Speech',
                        data=text
                    )
            
            elif action_type == "move":
                code = out.get("code") or out.get("word", "")
                if code == "WALK":
                    actions.append({"type": "MOVE", "code": "WALK", "steps": 6})
                    actions.append({"type": "SAY", "text": "Walking now."})
                elif code == "RUN":
                    actions.append({"type": "MOVE", "code": "RUN"})
                    actions.append({"type": "SAY", "text": "Running fast!"})
                elif code == "CENTER":
                    actions.append({"type": "MOVE", "code": "CENTER"})
                    actions.append({"type": "SAY", "text": "Resetting position."})

                if code in ["WALK", "RUN", "CENTER"]:
                # 🔥 Trigger animation: System1 output to Stickfigure
                    self.animation_engine.add_event(
                        event_type='system1_output',
                        source='Storage',
                        target='Stickfigure',
                        data=code
                    )

        # --- Step 4: Idle liveness movement ---
        if time.time() % 3 < 0.02:
            actions.append({"type": "MOVE", "code": "MOVE_RIGHT_ARM_DOWN"})

        return actions

############################################################
# Virtual Person (2D stick figure)
############################################################

class StickFigure:
    def __init__(self, cx: int, cy: int,
                 mental_dir: str = r"D:\artist\brainX\CRX\Properties\Mental_representation",
                 counter_file: str = "last_counter.json"):
        
        # Animation state - IMPROVED
        self.current_action = None  # Track current action
        self.action_start_time = 0  # When action started
        self.action_duration = 0    # How long action should last
        self.is_animating = False   # Whether currently animating
        self.animation_frame = 0    # Current animation frame
        self.animation_speed = 8    # Frames per second for animation
        
        # anchor position
        self.center_x = cx
        self.center_y = cy

        # Default/resting pose angles
        self.default_arm_left_angle = -45.0
        self.default_arm_right_angle = 45.0
        self.default_leg_left_angle = -10.0  
        self.default_leg_right_angle = 10.0
        
        # Current pose angles
        self.arm_left_angle = self.default_arm_left_angle
        self.arm_right_angle = self.default_arm_right_angle
        self.leg_left_angle = self.default_leg_left_angle  
        self.leg_right_angle = self.default_leg_right_angle
        
        # position offset (for walking movement)
        self.offset_x = 0.0
        self.target_offset_x = 0.0  # Target position for smooth movement

        # movement parameters
        self.arm_step = 10.0
        self.leg_step = 10.0
        self.walk_step_px = 3.0  # Reduced for smoother movement
        self.run_step_px = 6.0   # Faster movement for running

        # --- Mental representation setup (unchanged) ---
        self.mental_dir = mental_dir
        self.counter_path = os.path.join(mental_dir, counter_file)
        self.current_index = self._load_counter()
        self.mental_images = self._load_images()
        print(f"Loaded {len(self.mental_images)} mental images, starting at index {self.current_index}")
        
        # current surface for display
        self.current_mental_surface = None
        self.last_image_time = 0
        self.image_display_duration = 2.0
        
        # Initialize first image
        self.update_mental_image()

    # -------------------- Mental Image Handling (unchanged) --------------------
    def _load_counter(self) -> int:
        """Load the last viewed counter from file."""
        try:
            os.makedirs(self.mental_dir, exist_ok=True)
            with open(self.counter_path, "r") as f:
                data = json.load(f)
                return data.get("last_index", 0)
        except (FileNotFoundError, json.JSONDecodeError):
            return 0

    def _save_counter(self):
        """Save the last viewed counter to file."""
        try:
            os.makedirs(self.mental_dir, exist_ok=True)
            with open(self.counter_path, "w") as f:
                json.dump({"last_index": self.current_index}, f)
        except Exception as e:
            print(f"Warning: could not save counter: {e}")

    def _load_images(self):
        """Load sorted image file list (oldest→newest)."""
        if not os.path.exists(self.mental_dir):
            os.makedirs(self.mental_dir, exist_ok=True)
            return []
        
        imgs = []
        for f in os.listdir(self.mental_dir):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                full_path = os.path.join(self.mental_dir, f)
                imgs.append(full_path)
        
        imgs.sort(key=os.path.getctime)  # oldest first
        return imgs
    
    def update_mental_image(self):
        """Update mental image display - called from main thread to avoid blocking."""
        now = time.time()

        # If we have an image and haven't displayed it long enough, keep showing it
        if self.current_mental_surface and (now - self.last_image_time < self.image_display_duration):
            return  # still showing current image
        
        # Refresh the image list to catch new images
        self.mental_images = self._load_images()
        
        # Reset index if it's beyond the current list
        if self.current_index >= len(self.mental_images):
            self.current_index = 0
            
        if self.current_index < len(self.mental_images):
            print(f"Displaying mental image {self.current_index + 1}/{len(self.mental_images)}")
            img_path = self.mental_images[self.current_index]
            try:
                img = pygame.image.load(img_path)
                # scale down to fit small panel (e.g. 120x120)
                self.current_mental_surface = pygame.transform.scale(img, (120, 120))
                self.last_image_time = now
                
                # Move to next image
                self.current_index += 1
                self._save_counter()
                
                # Remove the displayed image file
                try:
                    os.remove(img_path)
                    print(f"Removed displayed image: {img_path}")
                except Exception as e:
                    print(f"Warning: could not remove {img_path}: {e}")
                    
            except Exception as e:
                print(f"Error loading image: {img_path}: {e}")
                self.current_mental_surface = None
                # Skip this image and try the next one
                self.current_index += 1
                self._save_counter()
        else:
            # No images available
            self.current_mental_surface = None

    # -------------------- Body Logic - IMPROVED --------------------
    def reset_to_default(self):
        """Reset stick figure to default standing position."""
        self.arm_left_angle = self.default_arm_left_angle
        self.arm_right_angle = self.default_arm_right_angle
        self.leg_left_angle = self.default_leg_left_angle
        self.leg_right_angle = self.default_leg_right_angle
        self.offset_x = 0.0
        self.target_offset_x = 0.0
        self.current_action = None
        self.is_animating = False
        print("[StickFigure] Reset to default position")

    def start_action(self, action_type: str, duration: float):
        """Start a new action with specified duration."""
        self.current_action = action_type
        self.action_start_time = time.time()
        self.action_duration = duration
        self.is_animating = True
        self.animation_frame = 0
        print(f"[StickFigure] Started action: {action_type} for {duration}s")

    def execute_action(self, action: dict):
        if action.get("type") != "MOVE":
            return
            
        code = action.get("code", "")
        
        # Single-frame movements
        if code == "MOVE_LEFT_ARM_UP":
            self.arm_left_angle -= self.arm_step
        elif code == "MOVE_LEFT_ARM_DOWN":
            self.arm_left_angle += self.arm_step
        elif code == "MOVE_RIGHT_ARM_UP":
            self.arm_right_angle -= self.arm_step
        elif code == "MOVE_RIGHT_ARM_DOWN":
            self.arm_right_angle += self.arm_step
        elif code == "STEP_LEFT":
            self.leg_left_angle += self.leg_step
        elif code == "STEP_RIGHT":
            self.leg_right_angle += self.leg_step
            
        # Multi-frame animated movements
        elif code == "WALK":
            if not self.is_animating or self.current_action != "WALK":
                self.start_action("WALK", duration=3.0)  # Walk for 3 seconds
                
        elif code == "RUN":
            if not self.is_animating or self.current_action != "RUN":
                self.start_action("RUN", duration=2.5)   # Run for 2.5 seconds
                
        elif code == "CENTER":
            self.reset_to_default()

    def update_animation(self, dt: float):
        """Update animation state - call this every frame."""
        if not self.is_animating or not self.current_action:
            return
            
        current_time = time.time()
        elapsed = current_time - self.action_start_time
        
        # Check if action should end
        if elapsed >= self.action_duration:
            print(f"[StickFigure] Action {self.current_action} completed")
            self.reset_to_default()
            return
            
        # Update animation frame based on time
        self.animation_frame = int(elapsed * self.animation_speed) % 4  # 4-frame cycle
        
        # Animate based on current action
        if self.current_action == "WALK":
            self._animate_walk()
        elif self.current_action == "RUN":
            self._animate_run()

    def _animate_walk(self):
        """Animate walking motion."""
        # Leg animation - alternating steps
        if self.animation_frame == 0:
            self.leg_left_angle = 25.0   # Left forward
            self.leg_right_angle = -15.0  # Right back
        elif self.animation_frame == 1:
            self.leg_left_angle = 15.0   # Left slowing
            self.leg_right_angle = -5.0   # Right coming forward
        elif self.animation_frame == 2:
            self.leg_left_angle = -15.0  # Left back
            self.leg_right_angle = 25.0   # Right forward
        elif self.animation_frame == 3:
            self.leg_left_angle = -5.0   # Left coming forward
            self.leg_right_angle = 15.0   # Right slowing
            
        # Arm animation - opposite to legs for natural walking
        if self.animation_frame in [0, 1]:
            self.arm_left_angle = -25.0   # Left arm back
            self.arm_right_angle = 65.0   # Right arm forward
        else:
            self.arm_left_angle = -65.0   # Left arm forward
            self.arm_right_angle = 25.0   # Right arm back
            
        # Gradual horizontal movement
        self.target_offset_x += self.walk_step_px
        self.offset_x = self.target_offset_x

    def _animate_run(self):
        """Animate running motion - more exaggerated than walking."""
        # More extreme leg positions for running
        if self.animation_frame == 0:
            self.leg_left_angle = 35.0   # Left forward (more extreme)
            self.leg_right_angle = -25.0  # Right back
        elif self.animation_frame == 1:
            self.leg_left_angle = 20.0   # Left slowing
            self.leg_right_angle = -10.0  # Right coming forward
        elif self.animation_frame == 2:
            self.leg_left_angle = -25.0  # Left back
            self.leg_right_angle = 35.0   # Right forward (more extreme)
        elif self.animation_frame == 3:
            self.leg_left_angle = -10.0  # Left coming forward
            self.leg_right_angle = 20.0   # Right slowing
            
        # More pronounced arm swinging for running
        if self.animation_frame in [0, 1]:
            self.arm_left_angle = -15.0   # Left arm back (less extreme than walking)
            self.arm_right_angle = 75.0   # Right arm forward
        else:
            self.arm_left_angle = -75.0   # Left arm forward
            self.arm_right_angle = 15.0   # Right arm back
            
        # Faster horizontal movement for running
        self.target_offset_x += self.run_step_px
        self.offset_x = self.target_offset_x

    def draw(self, surf: pygame.Surface, input_box: "TextInputBox"):
        BLACK = (0, 0, 0)
        cx = int(self.center_x + self.offset_x)
        cy = int(self.center_y)

        # Head
        pygame.draw.circle(surf, BLACK, (cx, cy - 60), 20, 2)

        # Body
        pygame.draw.line(surf, BLACK, (cx, cy - 40), (cx, cy + 40), 2)

        # Arms
        ARM_LEN = 55
        def endpoint_from_angle(base_x, base_y, deg):
            rad = math.radians(deg)
            ex = base_x + int(ARM_LEN * math.cos(rad))
            ey = base_y + int(ARM_LEN * math.sin(rad))
            return ex, ey

        left_arm_base = (cx, cy - 30)
        right_arm_base = (cx, cy - 30)
        la_end = endpoint_from_angle(left_arm_base[0], left_arm_base[1], 180 + self.arm_left_angle)
        ra_end = endpoint_from_angle(right_arm_base[0], right_arm_base[1], 0 + self.arm_right_angle)

        pygame.draw.line(surf, BLACK, left_arm_base, la_end, 2)
        pygame.draw.line(surf, BLACK, right_arm_base, ra_end, 2)

        # Legs
        LEG_LEN = 65
        hip = (cx, cy + 40)
        # Use 90 degrees as base (straight down) and add the angle offsets
        ll_end = endpoint_from_angle(hip[0], hip[1], 90 + self.leg_left_angle)
        rl_end = endpoint_from_angle(hip[0], hip[1], 90 + self.leg_right_angle)

        pygame.draw.line(surf, BLACK, hip, ll_end, 2)
        pygame.draw.line(surf, BLACK, hip, rl_end, 2)
        
        # ---- Draw mental image panel ----
        panel_rect = pygame.Rect(self.center_x + 150, self.center_y - 220, 220, 120)
        pygame.draw.rect(surf, (200, 200, 200), panel_rect)  # light gray background
        pygame.draw.rect(surf, (0, 0, 0), panel_rect, 2)  # black border

        if self.current_mental_surface:
            # Center the image in the panel
            img_rect = self.current_mental_surface.get_rect()
            img_rect.center = panel_rect.center
            surf.blit(self.current_mental_surface, img_rect)
            
        # ---- Draw command box (positioned stationary) ---- (FIXED)
        # Keep input box in fixed position, not following stick figure
        box_width = 350
        box_height = 35
        box_x = self.center_x - box_width // 2  # Center horizontally at original position
        box_y = self.center_y + 120  # Position below the original stick figure position

        # Update the input box rectangle
        input_box.rect.x = box_x
        input_box.rect.y = box_y
        input_box.rect.width = box_width
        input_box.rect.height = box_height
        
        input_box.draw(surf)
        
        # Draw animation status (for debugging)
        if self.is_animating:
            font = pygame.font.SysFont(None, 24)
            status_text = f"Action: {self.current_action} | Frame: {self.animation_frame}"
            status_surf = font.render(status_text, True, (255, 0, 0))  # Red text
            surf.blit(status_surf, (cx - 100, cy - 120))

############################################################
# Camera thread
############################################################

class CameraThread(threading.Thread):
    def __init__(self, camera_index: int, frame_queue: "queue.Queue"):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.cap = None
        self.frame_queue = frame_queue
        self.running = True
        self.cam_enabled = False 
        self.training_mode = False  # Add training mode flag

    def _open_camera(self):
        """Open the camera if not already open"""
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)
            if self.cap.isOpened():
                # Reduce capture resolution for speed
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
                print("[Camera] Camera opened")
            else:
                print("[Camera] Failed to open camera")

    def _close_camera(self):
        """Close the camera and release resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("[Camera] Camera closed and released")

    def set_training_mode(self, training_enabled):
        """Enable/disable training mode - pauses physical camera capture during training"""
        self.training_mode = training_enabled
        if training_enabled:
            print("[Camera] Training mode ON - Physical camera capture paused")
        else:
            print("[Camera] Training mode OFF - Physical camera capture resumed")

    def submit_image(self, frame):
        """Submit an image frame to the queue (similar to submit_text for text input)"""
        if frame is not None:
            # Clear frame queue to make room for the new frame
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            # Put the image frame into the frame queue
            self.frame_queue.put(frame)
            print(f"[IMG] Submitted training image frame to queue")
            return True
        else:
            print(f"[IMG] Failed to submit image - frame is None")
            return False
        
    def run(self):
        while self.running:
            # If in training mode, don't capture from physical camera
            if self.training_mode:
                time.sleep(0.1)  # Sleep while training is active
                continue
                
            if not self.cam_enabled:
                # Camera is turned off - close it if open
                if self.cap is not None:
                    self._close_camera()
                time.sleep(0.1)
                continue

            # Camera is enabled - open it if not already open
            if self.cap is None or not self.cap.isOpened():
                self._open_camera()
                if not self.cap.isOpened():
                    time.sleep(0.1)
                    continue

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
                
            # Keep only the most recent frame (clear queue)
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            self.frame_queue.put(frame)

        # Clean shutdown
        self._close_camera()

    def stop(self):
        self.running = False
        self._close_camera()
        return "Camera thread stopped"
            
class CamButton:
    def __init__(self, x, y, w=100, h=40):
        self.rect = pygame.Rect(x, y, w, h)
        self.is_on = False
        self.font = pygame.font.SysFont(None, 24)

    def draw(self, surf):
        color = (0, 200, 200) if self.is_on else (100, 100, 100)  # cyan/gray
        pygame.draw.rect(surf, color, self.rect)
        pygame.draw.rect(surf, (0, 0, 0), self.rect, 2)  # border
        text = "Cam ON" if self.is_on else "Cam OFF"
        label = self.font.render(text, True, (255, 255, 255))
        label_rect = label.get_rect(center=self.rect.center)
        surf.blit(label, label_rect)

    def handle_event(self, event, cam_thread: CameraThread):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.is_on = not self.is_on
                cam_thread.cam_enabled = self.is_on
                print(f"[UI] Camera toggled {'ON' if self.is_on else 'OFF'}")

############################################################
# Microphone → Text thread (Speech Recognition)
############################################################

class MicSTTThread(threading.Thread):
    def __init__(self, text_queue: "queue.Queue"):
        super().__init__(daemon=True)
        self.text_queue = text_queue
        self.running = True
        self.mic_enabled = False
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = ENERGY_THRESHOLD
        self.recognizer.pause_threshold = PAUSE_THRESHOLD

    def run(self):
        mic = None
        try:
            mic = sr.Microphone()
        except Exception as e:
            print(f"[STT] Could not access microphone: {e}")
            self.running = False

        if not mic:
            return

        with mic as source:
            # Calibrate with ambient noise for a second (optional)
            try:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            except Exception as e:
                print(f"[STT] Ambient noise calibration failed: {e}")

        print("[STT] Listening... (speak in short phrases)")
        while self.running:
            if not self.mic_enabled:
                time.sleep(0.1)
                continue  # skip loop when mic is OFF

            try:
                with mic as source:
                    audio = self.recognizer.listen(source, phrase_time_limit=PHRASE_TIME_LIMIT)
                text = None
                if USE_GOOGLE_ONLINE_STT:
                    try:
                        text = self.recognizer.recognize_google(audio)
                    except sr.UnknownValueError:
                        text = None
                    except sr.RequestError as e:
                        print(f"[STT] API request error: {e}")
                        text = None
                else:
                    # Placeholder: plug offline STT here
                    text = None

                if text and self.running and self.mic_enabled:  # Check running state before putting in queue
                    print(f"[STT] Heard: {text}")
                    self.text_queue.put(text)

            except Exception as e:
                if self.running:  # Only print errors if we're still supposed to be running
                    print(f"[STT] Error: {e}")
                time.sleep(0.2)

    def stop(self):
        self.running = False
        return "STT thread stopped"

class MicButton:
    def __init__(self, x, y, w=100, h=40):
        self.rect = pygame.Rect(x, y, w, h)
        self.is_on = False
        self.font = pygame.font.SysFont(None, 24)

    def draw(self, surf):
        color = (0, 200, 0) if self.is_on else (200, 0, 0)  # green/red
        pygame.draw.rect(surf, color, self.rect)
        pygame.draw.rect(surf, (0, 0, 0), self.rect, 2)  # border
        text = "Mic ON" if self.is_on else "Mic OFF"
        label = self.font.render(text, True, (255, 255, 255))
        label_rect = label.get_rect(center=self.rect.center)
        surf.blit(label, label_rect)

    def handle_event(self, event, stt_thread: MicSTTThread):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.is_on = not self.is_on
                stt_thread.mic_enabled = self.is_on
                print(f"[UI] Mic toggled {'ON' if self.is_on else 'OFF'}")

############################################################
# Word-to-text (shared, thread-safe enqueue) - IMPROVED
############################################################

class TextInputBox:
    def __init__(self, x, y, w, h, text_queue):
        self.rect = pygame.Rect(x, y, w, h)
        self.color_inactive = (200, 200, 200)
        self.color_active = (255, 255, 255)
        self.color_border = (0, 0, 0)
        self.color_border_active = (0, 100, 200)
        self.color = self.color_inactive
        self.text = ""
        self.font = pygame.font.Font(None, 24)
        self.active = False
        self.text_queue = text_queue
        self.cursor_visible = True
        self.cursor_timer = 0
        
        # Label for the input box
        self.label_font = pygame.font.Font(None, 20)
        self.label_text = "Type command and press Enter:"

    def handle_event(self, event):
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Toggle active state if user clicks the box
            if self.rect.collidepoint(event.pos):
                self.active = True
            else:
                self.active = False
            self.color = self.color_active if self.active else self.color_inactive

        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                if self.text.strip():
                    print(f"[CMD] Typed command: {self.text}")
                    self.text_queue.put(self.text.strip())
                self.text = ""
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.unicode.isprintable():  # Only add printable characters
                if len(self.text) < 100:  # Limit text length
                    self.text += event.unicode

    def submit_text(self):
        """Submit the current text to the queue and clear the input"""
        if self.text.strip():
            print(f"[CMD] Submitted text: {self.text}")
            self.text_queue.put(self.text.strip())
        self.text = ""

    def set_text(self, text):
        """Set text programmatically (for dataset feeding)"""
        self.text = text[:100]  # Respect the length limit

    def update(self, dt):
        """Update cursor blinking animation"""
        self.cursor_timer += dt
        if self.cursor_timer >= 500:  # Blink every 500ms
            self.cursor_visible = not self.cursor_visible
            self.cursor_timer = 0

    def draw(self, screen):
        # Draw label above the box
        label_surface = self.label_font.render(self.label_text, True, (60, 60, 60))
        screen.blit(label_surface, (self.rect.x, self.rect.y - 25))
        
        # Draw the input box
        pygame.draw.rect(screen, self.color, self.rect)
        border_color = self.color_border_active if self.active else self.color_border
        pygame.draw.rect(screen, border_color, self.rect, 2)

        # Render text
        if self.text or not self.active:
            display_text = self.text if self.text else "Click here to type..."
            text_color = (0, 0, 0) if self.text else (120, 120, 120)
            txt_surface = self.font.render(display_text, True, text_color)
            # Center text vertically in the box
            text_y = self.rect.y + (self.rect.height - txt_surface.get_height()) // 2
            screen.blit(txt_surface, (self.rect.x + 8, text_y))

        # Draw blinking cursor when active and has text
        if self.active and self.cursor_visible:
            text_width = self.font.size(self.text)[0] if self.text else 0
            cursor_x = self.rect.x + 8 + text_width
            cursor_y = self.rect.y + 6
            pygame.draw.line(screen, (0, 0, 0), 
                           (cursor_x, cursor_y), 
                           (cursor_x, cursor_y + self.rect.height - 12), 2)
            
############################################################
# Text-to-Speech (shared, thread-safe enqueue)
############################################################

class TTSWorker(threading.Thread):
    """
    Text-to-Speech worker with support for both queued natural speech
    and priority interrupts (cutting off mid-sentence if needed).
    """

    def __init__(self):
        super().__init__(daemon=True)
        self.engine = pyttsx3.init()
        self.normal_queue = queue.Queue()
        self.priority_queue = queue.Queue()
        self.running = True
        self.lock = threading.Lock()
        self.current_text = None

    def run(self):
        while self.running:
            try:
                # Priority queue checked first
                if not self.priority_queue.empty():
                    text = self.priority_queue.get_nowait()
                    self._speak(text)
                elif not self.normal_queue.empty():
                    text = self.normal_queue.get_nowait()
                    self._speak(text)
                else:
                    time.sleep(0.05)  # idle wait
            except Exception as e:
                print(f"[TTSWorker] Error: {e}")

    def _speak(self, text: str):
        with self.lock:
            self.current_text = text   
        self.engine.say(text)
        self.engine.runAndWait()
        with self.lock:
            self.current_text = None

    def say(self, text: str, priority: bool = False):
        """
        Queue text for speech.
        - Normal (default): goes into FIFO queue, spoken in turn.
        - Priority=True: interrupts current speech and speaks immediately.
        """
        if priority:
            # Interrupt current speech
            self.engine.stop()

            # Clear any existing priority items
            while not self.priority_queue.empty():
                try:
                    self.priority_queue.get_nowait()
                except queue.Empty:
                    break

            # Insert new high-priority phrase
            self.priority_queue.put(text)
        else:
            self.normal_queue.put(text)

    def stop(self):
        """Stop the TTS worker cleanly."""
        self.running = False
        self.engine.stop()
        self.join(timeout=2.0)

############################################################
# Model worker: gathers latest sensory input and calls your model
# PNS-crx of the virtual person, sends and receives message to CNS-crx
############################################################

class ModelWorker(threading.Thread):
    def __init__(self, frame_queue: "queue.Queue", text_queue: "queue.Queue",
                 action_queue: "queue.Queue", model: System2):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.text_queue = text_queue
        self.action_queue = action_queue
        self.model = model
        self.running = True

        self.latest_frame = None
        self.latest_text = None
        self.last_call_t = 0.0
        self.training_mode = False  # Add training mode flag
        self.normal_mode = False

    def set_training_mode(self, training_enabled):
        """Enable/disable training mode - pauses physical camera capture during training"""
        self.training_mode = training_enabled

    def set_normal_mode(self, normal_enabled):
        """Enable/disable normal mode - pauses physical camera capture during normal mode"""
        self.normal_mode = normal_enabled

    def run(self):
        while self.running:
            # Drain queues to get freshest inputs
            if self.training_mode:
                last_frame_time = None
                try:
                    while True:
                        self.latest_frame = self.frame_queue.get_nowait()
                        last_frame_time = time.time()  # mark when frame arrived 
                except queue.Empty:
                    pass

                try:
                    while True:
                        self.latest_text = self.text_queue.get_nowait()
                        print(f"Latest text updated to: {self.latest_text}")
                except queue.Empty:
                    pass
                
                now = time.time()
                
                if (self.latest_frame is not None and self.running and 
                    (now - self.last_call_t) >= MODEL_MIN_INTERVAL):
                    # --- Wait logic for audio (2 sec max) ---
                    if last_frame_time is not None:
                        waited = now - last_frame_time
                        if self.latest_text is None and waited < 2.0:
                            # No new text yet, keep waiting
                            time.sleep(0.05)
                            continue
                        
                    print("model running")
                
                    sensory_input = {
                        "vision_bgr": self.latest_frame,
                        "hearing_text": self.latest_text,
                        "timestamp": now
                    }
                
                    try:
                        actions = self.model.process(sensory_input) or []
                        for a in actions:
                            if self.running:  # Check before putting actions
                                self.action_queue.put(a)
                            else:
                                break

                        # Clear the inputs after processing to prevent reuse
                        self.latest_text = None  # ADD THIS LINE
                        self.latest_frame = None  # ADD THIS LINE
                        last_frame_time = None

                    except Exception as e:
                        if self.running:  # Only print errors if still running
                            print(f"[Model] Error during processing: {e}")
                    self.last_call_t = now

            elif self.normal_mode:
                # Drain queues to get freshest inputs
                last_frame_time = None
                try:
                    while True:
                        self.latest_frame = self.frame_queue.get_nowait()
                        last_frame_time = time.time()
                except queue.Empty:
                    pass

                try:
                    while True:
                        self.latest_text = self.text_queue.get_nowait()
                        print(f"Latest text updated to: {self.latest_text}")
                except queue.Empty:
                    pass

                now = time.time()
                if (now - self.last_call_t) >= MODEL_MIN_INTERVAL:
                    # Build sensory input according to availability
                    if self.latest_frame is not None or self.latest_text is not None:
                        sensory_input = {
                            "vision_bgr": self.latest_frame,
                            "hearing_text": self.latest_text,
                            "timestamp": now
                        }
                    else:
                        # both missing
                        sensory_input = {
                            "vision_bgr": None,
                            "hearing_text": None,
                            "timestamp": now
                        }

                    try:
                        actions = self.model.process(sensory_input) or []
                        for a in actions:
                            if self.running:  # Check before putting actions
                                self.action_queue.put(a)
                            else:
                                break

                        # Clear the inputs after processing to prevent reuse
                        self.latest_text = None  # ADD THIS LINE
                        self.latest_frame = None  # ADD THIS LINE
                        last_frame_time = None

                    except Exception as e:
                        if self.running:  # Only print errors if still running
                            print(f"[Model] Error during processing: {e}")
                    self.last_call_t = now

           # time.sleep(0.02)

    def stop(self):
        self.running = False
        return "Model thread stopped"

###########################################################
# Dataset feeding function
###########################################################

class DatasetFeeder():
    def __init__(self, dataset_path, image_word_dataset_path=None):
        self.dataset_path = dataset_path
        self.image_word_dataset_path = image_word_dataset_path
        self.words = []
        self.image_word_pairs = []
        self.current_index = 0
        self.current_image_word_index = 0
        self.load_dataset()
        if image_word_dataset_path:
            self.load_image_word_dataset()
    
    def load_dataset(self):
        """Load all words from dataset file once"""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as file:
                self.words = [line.strip() for line in file if line.strip()]
            random.shuffle(self.words)  # Shuffle for variety
            print(f"Loaded {len(self.words)} words from dataset")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.words = []
    
    def load_image_word_dataset(self):
        """Load image-word pairs from the specified directory"""
        if not self.image_word_dataset_path or not os.path.exists(self.image_word_dataset_path):
            print(f"Image-word dataset path not found: {self.image_word_dataset_path}")
            return
        
        try:
            # Get all files in the directory
            files = os.listdir(self.image_word_dataset_path)
            
            # Find all image files and their corresponding text files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
            
            for file in files:
                name, ext = os.path.splitext(file)
                if ext.lower() in image_extensions:
                    image_path = os.path.join(self.image_word_dataset_path, file)
                    text_path = os.path.join(self.image_word_dataset_path, f"{name}.txt")
                    
                    # Check if corresponding text file exists
                    if os.path.exists(text_path):
                        try:
                            with open(text_path, 'r', encoding='utf-8') as f:
                                word = f.read().strip()
                            
                            if word:  # Only add if word is not empty
                                self.image_word_pairs.append({
                                    'image_path': image_path,
                                    'word': word,
                                    'name': name
                                })
                        except Exception as e:
                            print(f"Error reading text file {text_path}: {e}")
            
            # Sort by filename to maintain order (01, 02, 03, etc.)
            self.image_word_pairs.sort(key=lambda x: int(x['name']) if x['name'].isdigit() else 0)
            
            print(f"Loaded {len(self.image_word_pairs)} image-word pairs from dataset")
            
        except Exception as e:
            print(f"Error loading image-word dataset: {e}")
            self.image_word_pairs = []
    
    def load_image_as_opencv_frame(self, image_path):
        """Load image as OpenCV frame (BGR format) for frame queue"""
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Could not load image: {image_path}")
                return None
            
            # Resize to standard camera resolution for consistency
            frame_resized = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            return frame_resized
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def load_image_as_surface(self, image_path):
        """Load image and convert to pygame surface for display"""
        try:
            # Load image using OpenCV
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                print(f"Could not load image: {image_path}")
                return None
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Resize image to reasonable size for display
            img_resized = cv2.resize(img_rgb, (100, 100), interpolation=cv2.INTER_AREA)
            
            # Convert to pygame surface
            surface = pygame.surfarray.make_surface(img_resized.swapaxes(0, 1))
            return surface
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def feed_next_batch(self, input_box, batch_size=100):
        """Feed next batch of words to the model (original functionality)"""
        if self.current_index >= len(self.words):
            print("Word dataset exhausted")
            return False
        
        # Get next batch
        end_index = min(self.current_index + batch_size, len(self.words))
        batch_words = self.words[self.current_index:end_index]
        
        # Create batch text
        batch_text = ' '.join(batch_words)
        
        # Set text and submit it
        input_box.set_text(batch_text)
        input_box.submit_text()
        
        print(f"Fed word batch {self.current_index//batch_size + 1}: {len(batch_words)} words")
        
        self.current_index = end_index
        return self.current_index < len(self.words)
    
    def feed_next_image_word_pair(self, input_box, camera_thread):
        """Feed next single image-word pair to the model using camera thread's submit_image"""
        if not self.image_word_pairs or self.current_image_word_index >= len(self.image_word_pairs):
            print("Image-word dataset exhausted")
            return False
        
        # Get the next pair
        pair = self.image_word_pairs[self.current_image_word_index]
        
        # Load the image as OpenCV frame
        frame = self.load_image_as_opencv_frame(pair['image_path'])
        
        # Submit image through camera thread
        image_submitted = False
        if frame is not None:
            image_submitted = camera_thread.submit_image(frame)
            if image_submitted:
                print(f"Sent image to camera thread: {pair['name']}")
            else:
                print(f"Failed to submit image through camera thread: {pair['name']}")
        else:
            print(f"Failed to load image: {pair['image_path']}")
        
        # Send the associated word through text input
        input_box.set_text(pair['word'])
        input_box.submit_text()
        
        print(f"Fed image-word pair {self.current_image_word_index + 1}: {pair['name']} -> '{pair['word']}'")
        
        # Store current pair for display purposes
        self.current_display_pair = pair
        
        self.current_image_word_index += 1
        return self.current_image_word_index < len(self.image_word_pairs)
    
    def get_current_display_pair(self):
        """Get the current pair for display"""
        return getattr(self, 'current_display_pair', None)
    
    def reset_datasets(self):
        """Reset both datasets to start from beginning"""
        self.current_index = 0
        self.current_image_word_index = 0
        print("Dataset indices reset to beginning")
    
    def get_dataset_status(self):
        """Get status information about both datasets"""
        word_progress = f"{self.current_index}/{len(self.words)}" if self.words else "0/0"
        image_word_progress = f"{self.current_image_word_index}/{len(self.image_word_pairs)}" if self.image_word_pairs else "0/0"
        
        return {
            'words_loaded': len(self.words),
            'image_pairs_loaded': len(self.image_word_pairs),
            'word_progress': word_progress,
            'image_word_progress': image_word_progress,
            'words_exhausted': self.current_index >= len(self.words),
            'image_words_exhausted': self.current_image_word_index >= len(self.image_word_pairs)
        }
    
############################################################
# Main app
############################################################

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Virtual Person - 2D Stick Figure (Camera + Mic + Text Input)")
    clock = pygame.time.Clock()

    # Queues for inter-thread communication
    frame_q: queue.Queue = queue.Queue(maxsize=2)
    text_q: queue.Queue = queue.Queue(maxsize=10)
    action_q: queue.Queue = queue.Queue(maxsize=50)

    # Create components
    stick = StickFigure(WIN_W // 2, WIN_H // 2 + 40)
    
    # Create input box once (positioned dynamically under stick figure)
    input_box = TextInputBox(x=200, y=400, w=350, h=45, text_queue=text_q)
    
    # Initialize dataset feeder with both word dataset and image-word dataset
    dataset_feeder = DatasetFeeder(
        dataset_path=r"D:\artist\brainX\CRX\Datasets\words\3k_common_english_words.txt",
        image_word_dataset_path=r"D:\artist\brainX\CRX\Datasets\Images\images_and_words"
    )
    
    # Training mode flags
    training_mode = False
    image_word_training_mode = False
    normal_mode = False
    last_feed_time = 0
    feed_interval = 5000  # Feed every 5 seconds during image-word training (slower for better association)
    word_feed_interval = 5000  # Feed every 5 seconds for word training

    model = System2()

    tts = TTSWorker()
    tts.start()
    
    cam = CameraThread(CAMERA_INDEX, frame_q)

    cam.start()
    stt = MicSTTThread(text_q)
    stt.start()
    mdl = ModelWorker(frame_q, text_q, action_q, model)
    mdl.start()
    
    font = pygame.font.SysFont(None, 20)
    
    mic_button = MicButton(WIN_W - 120, 20)
    cam_button = CamButton(WIN_W - 240, 20)
    training_button_rect = pygame.Rect(WIN_W - 350, 30, 80, 30)
    normal_button_rect = pygame.Rect(WIN_W - 350, 110, 80, 30)
    image_training_button_rect = pygame.Rect(WIN_W - 350, 70, 80, 30)
    reset_button_rect = pygame.Rect(WIN_W - 100, 200, 60, 30)
    
    def draw_camera_preview(surf: pygame.Surface, frame_bgr):
        """Draw a small camera preview at top-left."""
        if frame_bgr is None:
            return
        # Convert BGR (OpenCV) to RGB (pygame)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Resize to a thumbnail
        thumb = cv2.resize(frame_rgb, (320, 180), interpolation=cv2.INTER_AREA)
        # Convert to pygame surface
        frame_surface = pygame.surfarray.make_surface(thumb.swapaxes(0, 1))
        surf.blit(frame_surface, (10, 10))

        # Border
        pygame.draw.rect(surf, (0, 0, 0), pygame.Rect(10, 10, 320, 180), 2)

    def draw_current_training_pair(surf: pygame.Surface):
        """Draw current training image-word pair"""
        if not image_word_training_mode:
            return
        
        current_pair = dataset_feeder.get_current_display_pair()
        if not current_pair:
            return
        
        # Draw image at bottom left
        image_surface = dataset_feeder.load_image_as_surface(current_pair['image_path'])
        if image_surface:
            # Position at bottom left
            image_x = 10
            image_y = WIN_H - 190
            
            surf.blit(image_surface, (image_x, image_y))
            
            # Draw border
            pygame.draw.rect(surf, (255, 0, 0), pygame.Rect(image_x, image_y, 100, 100), 3)
            
            # Draw word label
            word_text = font.render(f"Training: {current_pair['word']}", True, (255, 0, 0))
            surf.blit(word_text, (image_x + 110, image_y + 40))

    running = True
    latest_frame_for_preview = None
    
    try:
        while running:
            dt = clock.tick(FPS)
            current_time = pygame.time.get_ticks()

            # Handle all pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if training_button_rect.collidepoint(event.pos):
                        training_mode = not training_mode
                        image_word_training_mode = False
                        # Update camera training mode
                        cam.set_training_mode(training_mode or image_word_training_mode)
                        mdl.set_training_mode(training_mode or image_word_training_mode)
                        model.set_training_mode(training_mode or image_word_training_mode)
                        print(f"Word training mode: {'ON' if training_mode else 'OFF'}")
                    elif image_training_button_rect.collidepoint(event.pos):
                        image_word_training_mode = not image_word_training_mode
                        training_mode = False
                        # Update camera training mode
                        cam.set_training_mode(training_mode or image_word_training_mode)
                        mdl.set_training_mode(training_mode or image_word_training_mode)
                        model.set_training_mode(training_mode or image_word_training_mode)
                        print(f"Image-word training mode: {'ON' if image_word_training_mode else 'OFF'}")
                    elif normal_button_rect.collidepoint(event.pos):
                        normal_mode = not normal_mode
                        training_mode = False
                        image_word_training_mode = False
                        # Update camera training mode
                        mdl.set_normal_mode(normal_enabled=normal_mode)
                        print(f"Normal mode: {'ON' if normal_mode else 'OFF'}")
                    elif reset_button_rect.collidepoint(event.pos):
                        dataset_feeder.reset_datasets()

                # Handle input box events (only if not in any training mode)
                if not training_mode and not image_word_training_mode:
                    input_box.handle_event(event)
                
                mic_button.handle_event(event, stt)
                cam_button.handle_event(event, cam)

            # Auto-feed dataset during training modes
            if training_mode and current_time - last_feed_time > word_feed_interval:
                has_more = dataset_feeder.feed_next_batch(input_box, batch_size=100)
                last_feed_time = current_time
                
                if not has_more:
                    training_mode = False
                    # Update camera training mode
                    cam.set_training_mode(training_mode or image_word_training_mode)
                    mdl.set_training_mode(training_mode or image_word_training_mode)
                    model.set_training_mode(training_mode or image_word_training_mode)
                    print("Word training completed - dataset exhausted")
            
            elif image_word_training_mode and current_time - last_feed_time > feed_interval:
                # Pass the camera thread instead of frame queue
                has_more = dataset_feeder.feed_next_image_word_pair(input_box, cam)
                last_feed_time = current_time
                
                if not has_more:
                    time.sleep(1.0)  # Give some time for last image to be processed
                    image_word_training_mode = False
                    # Update camera training mode
                    cam.set_training_mode(training_mode or image_word_training_mode)
                    mdl.set_training_mode(training_mode or image_word_training_mode)
                    model.set_training_mode(training_mode or image_word_training_mode)
                    print("Image-word training completed - dataset exhausted")

            # Update stick figure animation
            stick.update_animation(dt / 1000.0)  # Convert milliseconds to seconds

            # Update input box (for cursor blinking)
            input_box.update(dt)

            # Background
            screen.fill((245, 245, 245))

            # Draw training mode indicators
            if training_mode:
                training_text = font.render("WORD TRAINING MODE", True, (255, 255, 0))
                screen.blit(training_text, (10, WIN_H - 30))
            elif image_word_training_mode:
                training_text = font.render("IMAGE-WORD TRAINING MODE", True, (0, 255, 255))
                screen.blit(training_text, (10, WIN_H - 30))

            # Draw training buttons
            word_button_color = (0, 255, 0) if training_mode else (100, 100, 100)
            image_button_color = (0, 255, 255) if image_word_training_mode else (100, 100, 100)
            normal_button_color = (0, 200, 200) if normal_mode else (100, 100, 100)
            
            pygame.draw.rect(screen, word_button_color, training_button_rect)
            pygame.draw.rect(screen, (0, 0, 0), training_button_rect, 2)

            pygame.draw.rect(screen, image_button_color, image_training_button_rect)
            pygame.draw.rect(screen, (0, 0, 0), image_training_button_rect, 2)

            pygame.draw.rect(screen, normal_button_color, normal_button_rect)
            pygame.draw.rect(screen, (0, 0, 0), normal_button_rect, 2)

            pygame.draw.rect(screen, (255, 100, 100), reset_button_rect)
            pygame.draw.rect(screen, (0, 0, 0), reset_button_rect, 2)

            word_text = font.render("Words", True, (255, 255, 255))
            image_text = font.render("Images", True, (255, 255, 255))
            normal_text = font.render("Normal", True, (255, 255, 255))
            reset_text = font.render("Reset", True, (255, 255, 255))

            # Center the text inside the button rectangle
            text_rect = word_text.get_rect(center=training_button_rect.center)
            
            image_text_rect = image_text.get_rect(center=image_training_button_rect.center)
            normal_text_rect = normal_text.get_rect(center=normal_button_rect.center)
            reset_text_rect = reset_text.get_rect(center=reset_button_rect.center)

            screen.blit(word_text, text_rect)
            screen.blit(image_text, image_text_rect)
            screen.blit(normal_text, normal_text_rect)
            screen.blit(reset_text, reset_text_rect)

            # Draw current training pair
            draw_current_training_pair(screen)

            # Draw dataset status
            status = dataset_feeder.get_dataset_status()
            status_lines = [
                f"Words: {status['word_progress']} | I/W loaded: {status['image_pairs_loaded']}"
            ]
            for i, line in enumerate(status_lines):
                status_surf = font.render(line, True, (100, 100, 100))
                screen.blit(status_surf, (10, WIN_H - 80 + i*18))

            # Drain one frame for preview (do not block)
            try:
                while True:
                    latest_frame_for_preview = frame_q.get_nowait()
            except queue.Empty:
                pass

            # Draw camera preview if available
            draw_camera_preview(screen, latest_frame_for_preview)

            # Execute pending actions (MOVE/SAY)
            try:
                while True:
                    action = action_q.get_nowait()
                    if action["type"] == "MOVE":
                        stick.execute_action(action)
                    elif action["type"] == "SAY":
                        text = action["text"]

                        # Define emergency / interrupt rules here
                        emergency_keywords = ["stop", "danger", "emergency", "alert", "warning"]

                        # If any keyword is in the message → interrupt
                        if any(word.lower() in text.lower() for word in emergency_keywords):
                            tts.say(text, priority=True)
                        else:
                            tts.say(text)

            except queue.Empty:
                pass

            # Update mental image display (moved to main thread)
            stick.update_mental_image()

            # Draw stick figure with input box
            stick.draw(screen, input_box)

            with tts.lock:
                display_text = tts.current_text

            if display_text:
                txt_surf = font.render(f"Speaking: {display_text}", True, (0, 0, 0))
                screen.blit(txt_surf, (10, 200))

            mic_button.draw(screen)  
            cam_button.draw(screen)

            # Draw simple legend
            legend = [
                "Say 'hello', 'walk', 'run', 'center' to control the figure.",
                "Or type commands in the text box below the stick figure and press Enter."
            ]
            for i, line in enumerate(legend):
                lsurf = font.render(line, True, (0, 0, 0))
                screen.blit(lsurf, (10, WIN_H - 50 + i*18))

            pygame.display.flip()

    except KeyboardInterrupt:
        print("KeyboardInterrupt received, shutting down...")
    finally:
        # Graceful shutdown - stop all threads in correct order
        print("Shutting down...")
        running = False
        
        # Stop components in reverse order of dependency
        try: 
            result = model.stop()
            print(result)
        except Exception as e: 
            print(f"Model stop error: {e}")
            
        try: 
            result = mdl.stop()
            print(result)
        except Exception as e:
            print(f"Model worker stop error: {e}")
           
        try: 
           result = stt.stop()
           print(result)
        except Exception as e: 
           print(f"STT stop error: {e}")
           
        try: 
           result = cam.stop()
           print(result)
        except Exception as e: 
           print(f"Camera stop error: {e}")
           
        try: 
           result = tts.stop()
           print(result)
        except Exception as e: 
           print(f"TTS stop error: {e}")

        # Wait for threads to finish (with timeout)
        threads_to_join = [mdl, stt, cam, tts]
        for thread in threads_to_join:
           try:
               thread.join(timeout=2.0)  # Wait up to 2 seconds for each thread
               if thread.is_alive():
                   print(f"Warning: {thread.__class__.__name__} thread did not stop gracefully")
           except Exception as e:
               print(f"Error joining {thread.__class__.__name__} thread: {e}")

        pygame.quit()
        print("Application shutdown complete.")

if __name__ == "__main__":
    main()
   
   