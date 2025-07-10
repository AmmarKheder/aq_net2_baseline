#!/usr/bin/env python3
"""
Script de preprocessing et d'analyse des données CAQRA
Adapté pour la structure rossice
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import glob
from typing import List, Dict, Tuple
import argparse

class CAQRAAnalyzer:
    """Classe pour analyser les données CAQRA dans la structure rossice"""
    
    def __init__(self, data_path: str = "/scratch/project_462000640/ammar/data_rossice/"):
        self.data_path = data_path
        self.variables = ['u', 'v', 'temp', 'rh', 'psfc', 'pm25', 'pm10', 'so2', 'no2', 'co', 'o3']
        self.meteo_vars = ['u', 'v', 'temp', 'rh', 'psfc']
        self.pollutant_vars = ['pm25', 'pm10', 'so2', 'no2', 'co', 'o3']
        
    def scan_data_availability(self, years: List[int] = None) -> pd.DataFrame:
        """Scan la disponibilité des données par année avec structure YYYYMM"""
        if years is None:
            years = [2013, 2014, 2015, 2016, 2017, 2018]
        
        availability = []
        
        print("=== Scan de la disponibilité des données ===")
        for year in years:
            total_files = 0
            months_found = 0
            
            # Chercher tous les dossiers de mois pour cette année
            for month in range(1, 13):
                month_dir = f"{year}{month:02d}"  # Format YYYYMM
                month_path = os.path.join(self.data_path, month_dir)
                
                if os.path.exists(month_path):
                    files = glob.glob(os.path.join(month_path, "*.nc"))
                    total_files += len(files)
                    months_found += 1
                    print(f"  📁 {month_dir}: {len(files)} fichiers")
                
            # Calculer les heures attendues pour l'année
            if year % 4 == 0:  # Année bissextile
                expected_hours = 366 * 24
            else:
                expected_hours = 365 * 24
                
            coverage = total_files / expected_hours * 100 if expected_hours > 0 else 0
            
            status = "excellent" if coverage > 95 else "good" if coverage > 80 else "partial" if coverage > 50 else "poor"
            
            availability.append({
                'year': year,
                'files_count': total_files,
                'months_found': months_found,
                'expected_hours': expected_hours,
                'coverage_percent': coverage,
                'status': status
            })
            
            print(f"📅 Année {year}: {total_files:,} fichiers ({months_found}/12 mois, {coverage:.1f}% de couverture) - {status}")
        
        return pd.DataFrame(availability)
    
    def analyze_file_structure(self, sample_size: int = 5) -> Dict:
        """Analyse la structure des fichiers NetCDF avec structure YYYYMM"""
        print("\n=== Analyse de la structure des fichiers ===")
        
        # Trouver quelques fichiers d'exemple dans différents mois
        sample_files = []
        for year in [2013, 2014, 2015]:
            for month in [1, 6, 12]:  # Janvier, juin, décembre
                month_dir = f"{year}{month:02d}"
                month_path = os.path.join(self.data_path, month_dir)
                if os.path.exists(month_path):
                    files = glob.glob(os.path.join(month_path, "*.nc"))
                    if files:
                        sample_files.append(files[0])  # Prendre le premier fichier du mois
                    if len(sample_files) >= sample_size:
                        break
            if len(sample_files) >= sample_size:
                break
        
        if not sample_files:
            print("❌ Aucun fichier trouvé pour l'analyse")
            return {}
        
        print(f"🔍 Analyse de {len(sample_files)} fichiers d'exemple...")
        
        structure_info = {
            'files_analyzed': [],
            'common_variables': [],
            'dimensions': {},
            'spatial_info': {},
            'data_ranges': {}
        }
        
        for i, file_path in enumerate(sample_files):
            try:
                with xr.open_dataset(file_path) as ds:
                    file_info = {
                        'file': os.path.basename(file_path),
                        'variables': list(ds.data_vars.keys()),
                        'coordinates': list(ds.coords.keys()),
                        'dimensions': dict(ds.dims)
                    }
                    
                    # Analyser les variables
                    for var in self.variables:
                        if var in ds:
                            data = ds[var].values
                            file_info[f'{var}_shape'] = data.shape
                            file_info[f'{var}_dtype'] = str(data.dtype)
                            
                            # Statistiques de base
                            valid_data = data[~np.isnan(data)]
                            if len(valid_data) > 0:
                                file_info[f'{var}_min'] = float(np.min(valid_data))
                                file_info[f'{var}_max'] = float(np.max(valid_data))
                                file_info[f'{var}_mean'] = float(np.mean(valid_data))
                                file_info[f'{var}_nan_percent'] = float(np.isnan(data).sum() / data.size * 100)
                    
                    # Analyser les coordonnées spatiales
                    if 'lat2d' in ds and 'lon2d' in ds:
                        lat = ds['lat2d'].values
                        lon = ds['lon2d'].values
                        
                        file_info['spatial'] = {
                            'lat_range': (float(np.nanmin(lat)), float(np.nanmax(lat))),
                            'lon_range': (float(np.nanmin(lon)), float(np.nanmax(lon))),
                            'grid_shape': lat.shape,
                            'lat_resolution': float(np.mean(np.diff(lat, axis=0))),
                            'lon_resolution': float(np.mean(np.diff(lon, axis=1)))
                        }
                    
                    structure_info['files_analyzed'].append(file_info)
                    
                    print(f"✅ Fichier {i+1}/{len(sample_files)}: {os.path.basename(file_path)}")
                    print(f"   Variables: {len(file_info['variables'])}")
                    print(f"   Dimensions: {file_info['dimensions']}")
                    
            except Exception as e:
                print(f"❌ Erreur avec {file_path}: {e}")
                continue
        
        # Analyser les patterns communs
        if structure_info['files_analyzed']:
            all_vars = []
            for file_info in structure_info['files_analyzed']:
                all_vars.extend(file_info['variables'])
            
            from collections import Counter
            var_counts = Counter(all_vars)
            structure_info['common_variables'] = [var for var, count in var_counts.items() 
                                                if count == len(structure_info['files_analyzed'])]
            
            print(f"\n📊 Variables communes à tous les fichiers: {len(structure_info['common_variables'])}")
            print(f"   {structure_info['common_variables']}")
        
        return structure_info
    
    def compute_statistics(self, years: List[int], sample_size: int = 200) -> Dict:
        """Calcule les statistiques des variables sur un échantillon avec structure YYYYMM"""
        print(f"\n=== Calcul des statistiques sur {sample_size} fichiers ===")
        
        stats = {var: {'values': []} for var in self.variables}
        files_processed = 0
        
        # Collecter les fichiers de toutes les années et tous les mois
        all_files = []
        for year in years:
            for month in range(1, 13):
                month_dir = f"{year}{month:02d}"
                month_path = os.path.join(self.data_path, month_dir)
                if os.path.exists(month_path):
                    files = glob.glob(os.path.join(month_path, "*.nc"))
                    all_files.extend(files)
        
        if not all_files:
            print("❌ Aucun fichier trouvé")
            return {}
        
        # Échantillonner aléatoirement
        np.random.shuffle(all_files)
        sample_files = all_files[:sample_size]
        
        print(f"📈 Traitement de {len(sample_files)} fichiers...")
        
        for i, file_path in enumerate(sample_files):
            try:
                with xr.open_dataset(file_path) as ds:
                    for var in self.variables:
                        if var in ds:
                            data = ds[var].values
                            
                            # Prendre seulement le premier timestep si 3D
                            if data.ndim == 3:
                                data = data[0]
                            
                            # Sous-échantillonner spatialement pour efficacité
                            if data.ndim == 2:
                                data = data[::4, ::4]  # Prendre 1 point sur 16
                            
                            valid_data = data[~np.isnan(data)]
                            if len(valid_data) > 0:
                                # Échantillonner encore pour limiter la mémoire
                                if len(valid_data) > 1000:
                                    indices = np.random.choice(len(valid_data), 1000, replace=False)
                                    valid_data = valid_data[indices]
                                
                                stats[var]['values'].extend(valid_data.tolist())
                
                files_processed += 1
                if files_processed % 50 == 0:
                    print(f"   Fichiers traités: {files_processed}/{len(sample_files)}")
                    
            except Exception as e:
                continue
        
        # Calculer les statistiques finales
        final_stats = {}
        print("\n📊 Statistiques par variable:")
        
        for var in self.variables:
            if stats[var]['values']:
                values = np.array(stats[var]['values'])
                final_stats[var] = {
                    'count': len(values),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'q25': float(np.percentile(values, 25)),
                    'q50': float(np.percentile(values, 50)),
                    'q75': float(np.percentile(values, 75)),
                    'q95': float(np.percentile(values, 95)),
                    'q99': float(np.percentile(values, 99))
                }
                
                s = final_stats[var]
                var_type = "météo" if var in self.meteo_vars else "polluant"
                print(f"   {var:>6} ({var_type:>8}): mean={s['mean']:8.2f}, std={s['std']:8.2f}, "
                      f"range=[{s['min']:8.2f}, {s['max']:8.2f}]")
            else:
                final_stats[var] = None
                print(f"   {var:>6}: ❌ Pas de données valides")
        
        return final_stats
    
    def check_temporal_continuity(self, years: List[int] = [2013, 2014]) -> Dict:
        """Vérifie la continuité temporelle des données avec structure YYYYMM"""
        print(f"\n=== Vérification de la continuité temporelle ===")
        
        continuity_info = {}
        
        for year in years:
            all_files = []
            
            # Collecter tous les fichiers de l'année depuis tous les mois
            for month in range(1, 13):
                month_dir = f"{year}{month:02d}"
                month_path = os.path.join(self.data_path, month_dir)
                if os.path.exists(month_path):
                    files = glob.glob(os.path.join(month_path, "*.nc"))
                    all_files.extend(files)
            
            all_files.sort()
            
            # Extraire les timestamps
            timestamps = []
            for file in all_files:
                filename = os.path.basename(file)
                try:
                    timestamp_str = filename.split('CN-Reanalysis')[1].split('.nc')[0]
                    dt = datetime.strptime(timestamp_str, '%Y%m%d%H')
                    timestamps.append(dt)
                except:
                    continue
            
            timestamps.sort()
            
            # Analyser les gaps
            gaps = []
            for i in range(1, len(timestamps)):
                diff_hours = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600
                if diff_hours > 1.1:  # Gap de plus d'1 heure
                    gaps.append({
                        'start': timestamps[i-1],
                        'end': timestamps[i],
                        'duration_hours': diff_hours
                    })
            
            continuity_info[year] = {
                'total_files': len(all_files),
                'valid_timestamps': len(timestamps),
                'first_timestamp': timestamps[0] if timestamps else None,
                'last_timestamp': timestamps[-1] if timestamps else None,
                'gaps_count': len(gaps),
                'largest_gap_hours': max([g['duration_hours'] for g in gaps]) if gaps else 0,
                'gaps': gaps[:10]  # Garder seulement les 10 premiers gaps
            }
            
            print(f"📅 Année {year}:")
            print(f"   Fichiers: {len(all_files)}, Timestamps valides: {len(timestamps)}")
            print(f"   Période: {timestamps[0]} → {timestamps[-1]}")
            print(f"   Gaps: {len(gaps)} (max: {max([g['duration_hours'] for g in gaps]) if gaps else 0:.1f}h)")
        
        return continuity_info
    
    def create_comprehensive_report(self, output_dir: str = "/scratch/project_462000640/ammar/rossice/outputs/"):
        """Crée un rapport complet d'analyse"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("🚀 === ANALYSE COMPLÈTE DES DONNÉES CAQRA ===")
        
        # 1. Disponibilité des données
        availability = self.scan_data_availability()
        availability.to_csv(os.path.join(output_dir, "data_availability.csv"), index=False)
        
        # 2. Structure des fichiers
        structure_info = self.analyze_file_structure()
        
        # 3. Statistiques des variables
        stats = self.compute_statistics([2013, 2014, 2015])
        
        # 4. Continuité temporelle
        continuity = self.check_temporal_continuity([2013, 2014, 2015])
        
        # Créer le rapport JSON
        report = {
            'analysis_info': {
                'date': datetime.now().isoformat(),
                'data_path': self.data_path,
                'output_dir': output_dir
            },
            'data_availability': availability.to_dict('records'),
            'file_structure': structure_info,
            'variable_statistics': stats,
            'temporal_continuity': continuity
        }
        
        import json
        with open(os.path.join(output_dir, "caqra_comprehensive_report.json"), 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Créer graphiques
        self._create_analysis_plots(stats, availability, output_dir)
        
        # Résumé final
        print("\n✅ === RÉSUMÉ DE L'ANALYSE ===")
        total_files = availability['files_count'].sum()
        avg_coverage = availability['coverage_percent'].mean()
        available_vars = len([v for v in stats if stats.get(v)])
        
        print(f"📁 Total fichiers: {total_files:,}")
        print(f"📊 Couverture moyenne: {avg_coverage:.1f}%")
        print(f"📈 Variables avec données: {available_vars}/{len(self.variables)}")
        print(f"💾 Rapport sauvegardé: {output_dir}")
        
        # Recommandations pour ClimaX
        print("\n💡 === RECOMMANDATIONS POUR CLIMAX ===")
        
        if structure_info.get('files_analyzed'):
            spatial = structure_info['files_analyzed'][0].get('spatial', {})
            if spatial:
                shape = spatial.get('grid_shape', (0, 0))
                print(f"🌍 Résolution spatiale: {shape}")
                print(f"📐 Résolution recommandée pour ClimaX: {self._recommend_climax_resolution(shape)}")
        
        best_years = availability[availability['coverage_percent'] > 90]['year'].tolist()
        print(f"📅 Années recommandées pour entraînement: {best_years}")
        
        if stats:
            pollutant_stats = {var: stats[var] for var in self.pollutant_vars if stats.get(var)}
            print(f"🏭 Polluants avec bonnes données: {list(pollutant_stats.keys())}")
        
        return report
    
    def _recommend_climax_resolution(self, original_shape: Tuple[int, int]) -> str:
        """Recommande une résolution compatible avec ClimaX"""
        if not original_shape or len(original_shape) != 2:
            return "128x256 (par défaut)"
        
        h, w = original_shape
        
        # Trouver des dimensions compatibles avec ViT (divisibles par patch_size)
        # et proches de la résolution originale
        target_ratios = [(1, 2), (2, 3), (3, 4), (1, 1)]
        
        for ratio_h, ratio_w in target_ratios:
            for base_size in [64, 96, 128, 160, 192, 224, 256]:
                new_h = base_size * ratio_h
                new_w = base_size * ratio_w
                
                # Vérifier que c'est divisible par patch_size communs (4, 8, 16)
                if new_h % 16 == 0 and new_w % 16 == 0:
                    # Calculer la différence avec la résolution originale
                    diff = abs(h - new_h) + abs(w - new_w)
                    if diff < min(h, w) * 0.5:  # Pas plus de 50% de différence
                        return f"{new_h}x{new_w}"
        
        return "128x256 (par défaut)"
    
    def _create_analysis_plots(self, stats: Dict, availability: pd.DataFrame, output_dir: str):
        """Crée des graphiques d'analyse"""
        
        # 1. Couverture des données par année
        plt.figure(figsize=(12, 6))
        colors = ['red' if x < 50 else 'orange' if x < 80 else 'yellow' if x < 95 else 'green' 
                 for x in availability['coverage_percent']]
        
        bars = plt.bar(availability['year'], availability['coverage_percent'], color=colors)
        plt.title('Couverture des données CAQRA par année', fontsize=14, fontweight='bold')
        plt.xlabel('Année')
        plt.ylabel('Couverture (%)')
        plt.ylim(0, 105)
        
        # Ajouter labels sur les barres
        for bar, coverage in zip(bars, availability['coverage_percent']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{coverage:.1f}%', ha='center', va='bottom')
        
        # Légende
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Excellent (>95%)'),
            Patch(facecolor='yellow', label='Bon (80-95%)'),
            Patch(facecolor='orange', label='Partiel (50-80%)'),
            Patch(facecolor='red', label='Pauvre (<50%)')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'data_coverage_by_year.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Distribution des polluants
        if any(stats.get(var) for var in self.pollutant_vars):
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, var in enumerate(self.pollutant_vars):
                ax = axes[i]
                
                if stats.get(var):
                    s = stats[var]
                    
                    # Créer une distribution approximative avec les quartiles
                    quartiles = [s['min'], s['q25'], s['q50'], s['q75'], s['max']]
                    
                    # Box plot style avec quartiles
                    bp = ax.boxplot([quartiles[1:4]], positions=[1], widths=0.6, 
                                   patch_artist=True)
                    bp['boxes'][0].set_facecolor('lightblue')
                    
                    ax.set_title(f'{var.upper()}\n(μg/m³)' if var != 'co' else f'{var.upper()}\n(mg/m³)', 
                               fontweight='bold')
                    ax.set_ylabel('Concentration')
                    
                    # Ajouter statistiques
                    stats_text = (f'Moyenne: {s["mean"]:.2f}\n'
                                f'Médiane: {s["q50"]:.2f}\n'
                                f'Std: {s["std"]:.2f}\n'
                                f'Max: {s["max"]:.2f}')
                    
                    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                           verticalalignment='top', fontsize=10,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                    
                    ax.set_xticklabels(['Distribution'])
                    
                else:
                    ax.text(0.5, 0.5, 'Pas de données', ha='center', va='center',
                           transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'{var.upper()} - Non disponible')
            
            plt.suptitle('Distribution des concentrations de polluants', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'pollutant_distributions.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"📊 Graphiques sauvegardés dans {output_dir}")


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description='Analyse complète des données CAQRA')
    parser.add_argument('--data_path', type=str, 
                       default="/scratch/project_462000640/ammar/data_rossice/",
                       help='Chemin vers les données CAQRA')
    parser.add_argument('--output_dir', type=str, 
                       default="/scratch/project_462000640/ammar/rossice/outputs/",
                       help='Dossier de sortie pour les résultats')
    parser.add_argument('--years', nargs='+', type=int, 
                       default=[2013, 2014, 2015, 2016, 2017, 2018],
                       help='Années à analyser')
    parser.add_argument('--sample_size', type=int, default=200,
                       help='Nombre de fichiers pour calcul statistiques')
    
    args = parser.parse_args()
    
    # Vérifier que le dossier de données existe
    if not os.path.exists(args.data_path):
        print(f"❌ Erreur: Dossier de données non trouvé: {args.data_path}")
        return
    
    print(f"🎯 Analyse des données CAQRA")
    print(f"📁 Source: {args.data_path}")
    print(f"💾 Output: {args.output_dir}")
    print(f"📅 Années: {args.years}")
    
    # Créer l'analyseur
    analyzer = CAQRAAnalyzer(args.data_path)
    
    # Lancer l'analyse complète
    report = analyzer.create_comprehensive_report(args.output_dir)
    
    print("\n🎉 Analyse terminée avec succès !")


if __name__ == "__main__":
    main()