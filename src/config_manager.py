"""
AQ_Net2 Project Configuration Manager
Centralized management of temporal parameters and configuration validation
"""
import yaml
from typing import Dict, Any

class ConfigManager:
    """Centralized configuration management with temporal parameter conversion"""
    
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self._validate_config()
        self._compute_temporal_parameters()
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _validate_config(self):
        """Validate required configuration parameters"""
        required_keys = [
            ('data', 'time_step'),
            ('data', 'history_days'), 
            ('data', 'forecast_days'),
            ('train', 'epochs'),
            ('train', 'batch_size')
        ]
        
        for section, key in required_keys:
            if section not in self.config or key not in self.config[section]:
                raise ValueError(f"Missing required config parameter: {section}.{key}")
    
    def _compute_temporal_parameters(self):
        """Convert temporal parameters from days to hours"""
        data_config = self.config['data']
        
        # Convert days to hours
        time_step = data_config['time_step']  # Should be 1 hour
        history_days = data_config['history_days']
        forecast_days = data_config['forecast_days']
        
        # Computed parameters in hours
        self.time_history_hours = history_days * 24
        self.time_future_hours = forecast_days * 24
        
        # Store in config for easy access
        data_config['time_history'] = self.time_history_hours
        data_config['time_future'] = self.time_future_hours
        
        print(f"Temporal configuration:")
        print(f"  Time step: {time_step} hour(s)")
        print(f"  History: {history_days} days = {self.time_history_hours} hours")
        print(f"  Forecast: {forecast_days} days = {self.time_future_hours} hours")
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration with computed temporal parameters"""
        return self.config['data']
    
    def get_train_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.config['train']
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config['model']
    
    def get_system_config(self) -> Dict[str, Any]:
        """Get system configuration"""
        return self.config['system']
    
    def print_summary(self):
        """Print configuration summary"""
        print("\n=== AQ_Net2 Configuration Summary ===")
        
        # Training parameters
        train_config = self.get_train_config()
        print(f"Training:")
        print(f"  Epochs: {train_config['epochs']}")
        print(f"  Batch size: {train_config['batch_size']}")
        print(f"  Learning rate: {train_config['learning_rate']}")
        print(f"  Patience: {train_config['patience']}")
        
        # Data parameters
        data_config = self.get_data_config()
        print(f"\nData:")
        print(f"  Train years: {data_config['train_years']}")
        print(f"  Val years: {data_config['val_years']}")
        print(f"  Test years: {data_config['test_years']}")
        print(f"  Variables: {len(data_config['variables'])} ({', '.join(data_config['variables'])})")
        
        # Temporal parameters
        print(f"\nTemporal:")
        print(f"  Time step: {data_config['time_step']} hour(s)")
        print(f"  History: {data_config['history_days']} days ({data_config['time_history']} hours)")
        print(f"  Forecast: {data_config['forecast_days']} days ({data_config['time_future']} hours)")
        
        print("=" * 38)
