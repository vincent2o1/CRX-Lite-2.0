import json
import os
import threading
import time
from typing import Dict, Any

class SharedConfig:
    def __init__(self, config_path: str = r"D:\artist\brainX\CRX\Properties\shared_config.json"):
        self.config_path = config_path
        self.lock = threading.Lock()  # For thread safety
        self._ensure_config_exists()
    
    def _ensure_config_exists(self):
        """Create default config if it doesn't exist"""
        if not os.path.exists(self.config_path):
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            default_config = {
                "eye": False,
                "ear": False,
                "destabilizing_mechanism": False,
                "training": True,
                "organizing": True,
                "testing": True,
                "initial_grouping": True,
                "last_updated": time.time()
            }
            self.save_config(default_config)
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading config: {e}")
            return {"eye": False, "ear": False, "destabilizing_mechanism": False, "training": True}
    
    def save_config(self, config: Dict[str, Any]):
        """Save configuration to file"""
        try:
            config["last_updated"] = time.time()
            with self.lock:  # Thread-safe writing
                with open(self.config_path, 'w') as f:
                    json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def update_sensor_status(self, eye: bool = None, ear: bool = None):
        """Update sensor status"""
        config = self.load_config()
        if eye is not None:
            config["eye"] = eye
        if ear is not None:
            config["ear"] = ear
        self.save_config(config)
    
    def get_sensor_status(self) -> tuple:
        """Get current sensor status"""
        config = self.load_config()
        return config.get("eye", False), config.get("ear", False)
   
