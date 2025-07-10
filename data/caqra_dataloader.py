import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from datetime import datetime, timedelta
import glob
from typing import List, Tuple, Dict, Optional
import warnings
from scipy.ndimage import zoom
warnings.filterwarnings('ignore')

class CAQRADataset(Dataset):
    """
    Dataset pour les données CAQRA adaptées à ClimaX avec fine-tuning
    Compatible avec le checkpoint pré-entraîné checkpoints/climax_1.40625deg.ckpt
    """
    
    def __init__(
        self,
        data_path: str = "/scratch/project_462000640/ammar/data_rossice/",
        years: List[int] = [2013, 2014, 2015, 2016],
        variables: Optional[List[str]] = None,
        target_variables: Optional[List[str]] = None,
        time_history: int = 6,  # 6 heures d'historique
        time_future: int = 12,  # Prédire 12 heures à l'avance
        spatial_subsample: int = 1,  # Sous-échantillonnage spatial
        normalize: bool = True,
        target_resolution: Tuple[int, int] = (128, 256),  # Résolution compatible ClimaX
        use_pretrained_vars: bool = True  # Utiliser variables compatibles checkpoint
    ):
        self.data_path = data_path
        self.years = years
        self.time_history = time_history
        self.time_future = time_future
        self.spatial_subsample = spatial_subsample
        self.normalize = normalize
        self.target_resolution = target_resolution
        self.use_pretrained_vars = use_pretrained_vars
        
        # Variables d'entrée - compatibles avec checkpoint pré-entraîné ClimaX
        if variables is None:
            if use_pretrained_vars:
                # Variables météo standards utilisées par ClimaX pré-entraîné
                self.variables = ['u', 'v', 'temp', 'rh', 'psfc']
            else:
                self.variables = ['u', 'v', 'temp', 'rh', 'psfc']
        else:
            self.variables = variables
            
        # Variables cibles (polluants à prédire)
        if target_variables is None:
            self.target_variables = ['pm25', 'pm10', 'so2', 'no2', 'co', 'o3']
        else:
            self.target_variables = target_variables
        
        print(f"Variables d'entrée: {self.variables}")
        print(f"Variables cibles: {self.target_variables}")
        print(f"Résolution cible: {self.target_resolution}")
        
        # Créer la liste des séquences temporelles valides
        self.sequences = self._create_sequences()
        print(f"Séquences créées: {len(self.sequences)}")
        
        # Calculer statistiques pour normalisation
        if self.normalize:
            self.stats = self._compute_stats()
    
    def _create_sequences(self) -> List[Dict]:
        """
        Crée une liste de toutes les séquences temporelles valides
        Structure: data_rossice/YYYYMM/CN-Reanalysis*.nc
        """
        sequences = []
        
        for year in self.years:
            # Pour chaque année, chercher tous les dossiers de mois (YYYYMM)
            year_files = []
            for month in range(1, 13):  # 12 mois
                month_dir = f"{year}{month:02d}"  # Format YYYYMM (ex: 201301)
                month_path = os.path.join(self.data_path, month_dir)
                
                if os.path.exists(month_path):
                    month_files = glob.glob(os.path.join(month_path, "*.nc"))
                    year_files.extend(month_files)
                    print(f"Mois {month_dir}: {len(month_files)} fichiers trouvés")
                else:
                    print(f"Attention: {month_path} n'existe pas")
            
            year_files.sort()
            print(f"Année {year}: {len(year_files)} fichiers au total")
            
            # Extraire les timestamps des noms de fichiers
            timestamps = []
            for file in year_files:
                filename = os.path.basename(file)
                # Format: CN-Reanalysis2013010100.nc -> 2013010100
                try:
                    timestamp_str = filename.split('CN-Reanalysis')[1].split('.nc')[0]
                    dt = datetime.strptime(timestamp_str, '%Y%m%d%H')
                    timestamps.append(dt)
                except:
                    continue
            
            timestamps.sort()
            print(f"Timestamps valides pour {year}: {len(timestamps)}")
            
            # Créer les séquences avec historique + futur
            sequence_count = 0
            for i in range(len(timestamps) - self.time_history - self.time_future + 1):
                start_time = timestamps[i]
                
                # Vérifier la continuité temporelle (pas de trous > 1h)
                sequence_times = timestamps[i:i + self.time_history + self.time_future]
                if self._check_continuity(sequence_times):
                    sequences.append({
                        'start_time': start_time,
                        'input_times': timestamps[i:i + self.time_history],
                        'target_times': timestamps[i + self.time_history:i + self.time_history + self.time_future]
                    })
                    sequence_count += 1
            
            print(f"Séquences valides pour {year}: {sequence_count}")
        
        print(f"Total séquences créées: {len(sequences)}")
        return sequences
    
    def _check_continuity(self, timestamps: List[datetime]) -> bool:
        """Vérifie qu'il n'y a pas de trous dans la série temporelle"""
        for i in range(1, len(timestamps)):
            diff = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600
            if diff > 1.1:  # Plus de 1h de gap
                return False
        return True
    
    def _get_file_path(self, dt: datetime) -> str:
        """Construit le chemin du fichier pour un timestamp donné"""
        # Structure: data_rossice/YYYYMM/CN-ReanalysisYYYYMMDDHH.nc
        year_month = dt.strftime('%Y%m')  # Format YYYYMM
        filename = f"CN-Reanalysis{dt.strftime('%Y%m%d%H')}.nc"
        return os.path.join(self.data_path, year_month, filename)
    
    def _load_single_file(self, dt: datetime) -> Dict[str, np.ndarray]:
        """Charge un seul fichier NetCDF et redimensionne selon target_resolution"""
        file_path = self._get_file_path(dt)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fichier non trouvé: {file_path}")
        
        data = {}
        try:
            with xr.open_dataset(file_path) as ds:
                # Charger les variables d'entrée
                for var in self.variables:
                    if var in ds:
                        arr = ds[var].values
                        
                        # Gérer les dimensions
                        if arr.ndim == 3:  # (time, lat, lon) -> prendre le premier temps
                            arr = arr[0]
                        elif arr.ndim == 2:  # (lat, lon) -> OK
                            pass
                        else:
                            print(f"Dimension inattendue pour {var}: {arr.shape}")
                            continue
                        
                        # Sous-échantillonnage spatial si nécessaire
                        if self.spatial_subsample > 1:
                            arr = arr[::self.spatial_subsample, ::self.spatial_subsample]
                        
                        # Redimensionner vers target_resolution
                        arr = self._resize_array(arr, self.target_resolution)
                        data[var] = arr
                    else:
                        print(f"Variable {var} non trouvée dans {file_path}")
                
                # Charger les variables cibles
                for var in self.target_variables:
                    if var in ds:
                        arr = ds[var].values
                        
                        if arr.ndim == 3:
                            arr = arr[0]
                        elif arr.ndim == 2:
                            pass
                        else:
                            continue
                        
                        if self.spatial_subsample > 1:
                            arr = arr[::self.spatial_subsample, ::self.spatial_subsample]
                        
                        arr = self._resize_array(arr, self.target_resolution)
                        data[var] = arr
                    else:
                        print(f"Variable cible {var} non trouvée dans {file_path}")
                
        except Exception as e:
            print(f"Erreur lors du chargement de {file_path}: {e}")
            raise
        
        return data
    
    def _resize_array(self, arr: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Redimensionne un array 2D vers la forme cible"""
        if arr.shape == target_shape:
            return arr
        
        # Calculer les facteurs de zoom
        zoom_factors = (target_shape[0] / arr.shape[0], target_shape[1] / arr.shape[1])
        
        # Redimensionner avec interpolation
        resized = zoom(arr, zoom_factors, order=1, mode='nearest')
        
        return resized
    
    def _compute_stats(self) -> Dict[str, Dict[str, float]]:
        """Calcule les statistiques pour la normalisation"""
        print("Calcul des statistiques pour normalisation...")
        
        stats = {}
        all_vars = self.variables + self.target_variables
        
        # Initialiser les accumulateurs
        sums = {var: 0.0 for var in all_vars}
        squares = {var: 0.0 for var in all_vars}
        counts = {var: 0 for var in all_vars}
        
        # Calculer sur un sous-ensemble pour l'efficacité
        sample_sequences = self.sequences[::max(1, len(self.sequences) // 500)]
        
        for seq in sample_sequences:
            try:
                for dt in seq['input_times'] + seq['target_times']:
                    data = self._load_single_file(dt)
                    
                    for var in all_vars:
                        if var in data:
                            arr = data[var].flatten()
                            valid_mask = ~np.isnan(arr)
                            valid_data = arr[valid_mask]
                            
                            if len(valid_data) > 0:
                                sums[var] += np.sum(valid_data)
                                squares[var] += np.sum(valid_data ** 2)
                                counts[var] += len(valid_data)
            except:
                continue
        
        # Calculer moyennes et écarts-types
        for var in all_vars:
            if counts[var] > 0:
                mean = sums[var] / counts[var]
                variance = (squares[var] / counts[var]) - (mean ** 2)
                std = np.sqrt(max(variance, 1e-8))
                stats[var] = {'mean': mean, 'std': std}
                print(f"{var}: mean={mean:.4f}, std={std:.4f}")
            else:
                stats[var] = {'mean': 0.0, 'std': 1.0}
                print(f"{var}: pas de données valides")
        
        print("Statistiques calculées")
        return stats
    
    def _normalize_data(self, data: np.ndarray, var: str) -> np.ndarray:
        """Normalise les données"""
        if self.normalize and var in self.stats:
            mean = self.stats[var]['mean']
            std = self.stats[var]['std']
            return (data - mean) / std
        return data
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retourne:
        - input_tensor: (time_history, channels, H, W) - variables d'entrée
        - target_tensor: (time_future, target_channels, H, W) - variables cibles
        """
        seq = self.sequences[idx]
        
        # Charger les données d'entrée (historique)
        input_data = []
        for dt in seq['input_times']:
            try:
                data = self._load_single_file(dt)
                # Empiler les variables d'entrée
                timestep_data = []
                for var in self.variables:
                    if var in data:
                        normalized = self._normalize_data(data[var], var)
                        timestep_data.append(normalized)
                    else:
                        # Remplir avec des zéros si variable manquante
                        timestep_data.append(np.zeros(self.target_resolution))
                
                input_data.append(np.stack(timestep_data, axis=0))  # (channels, H, W)
            except Exception as e:
                print(f"Erreur lors du chargement {dt}: {e}")
                # Créer des données par défaut
                shape = (len(self.variables), self.target_resolution[0], self.target_resolution[1])
                input_data.append(np.zeros(shape))
        
        # Charger les données cibles (futur)
        target_data = []
        for dt in seq['target_times']:
            try:
                data = self._load_single_file(dt)
                timestep_data = []
                for var in self.target_variables:
                    if var in data:
                        normalized = self._normalize_data(data[var], var)
                        timestep_data.append(normalized)
                    else:
                        timestep_data.append(np.zeros(self.target_resolution))
                
                target_data.append(np.stack(timestep_data, axis=0))
            except Exception as e:
                print(f"Erreur lors du chargement target {dt}: {e}")
                shape = (len(self.target_variables), self.target_resolution[0], self.target_resolution[1])
                target_data.append(np.zeros(shape))
        
        # Convertir en tenseurs
        input_tensor = torch.from_numpy(np.stack(input_data, axis=0)).float()  # (time_history, channels, H, W)
        target_tensor = torch.from_numpy(np.stack(target_data, axis=0)).float()  # (time_future, target_channels, H, W)
        
        return input_tensor, target_tensor

def create_caqra_dataloaders(
    data_path: str = "/scratch/project_462000640/ammar/data_rossice/",
    train_years: List[int] = [2013, 2014, 2015, 2016],
    val_years: List[int] = [2017],
    test_years: List[int] = [2018],
    batch_size: int = 4,
    num_workers: int = 4,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crée les dataloaders pour train/val/test
    """
    
    # Créer les datasets
    train_dataset = CAQRADataset(data_path=data_path, years=train_years, **dataset_kwargs)
    val_dataset = CAQRADataset(data_path=data_path, years=val_years, **dataset_kwargs)
    test_dataset = CAQRADataset(data_path=data_path, years=test_years, **dataset_kwargs)
    
    # Partager les statistiques de normalisation
    if hasattr(train_dataset, 'stats'):
        val_dataset.stats = train_dataset.stats
        test_dataset.stats = train_dataset.stats
    
    # Créer les dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

# Test rapide
if __name__ == "__main__":
    # Test du dataloader
    dataset = CAQRADataset(
        data_path="/scratch/project_462000640/ammar/data_rossice/",
        years=[2013],
        time_history=3,
        time_future=6,
        spatial_subsample=4,  # Pour test rapide
        target_resolution=(64, 128)  # Résolution réduite pour test
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        input_tensor, target_tensor = dataset[0]
        print(f"Input shape: {input_tensor.shape}")   # (time_history, channels, H, W)
        print(f"Target shape: {target_tensor.shape}") # (time_future, target_channels, H, W)
        print("Test réussi !")
    else:
        print("ERREUR: Dataset vide")