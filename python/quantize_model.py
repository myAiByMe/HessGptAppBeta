#!/usr/bin/env python3
"""
Script pour quantifier le modèle HessGPT en INT8
Réduit la taille du modèle de ~75% et accélère l'inférence

Usage:
    python quantize_model.py --input model.ptl --output model_quantized.ptl
"""

import argparse
import torch
import os
from torch.quantization import quantize_dynamic


def quantize_model(input_path, output_path, quantization_type='dynamic'):
    """
    Quantifie le modèle en INT8
    
    Args:
        input_path: Chemin vers le modèle TorchScript (.ptl)
        output_path: Chemin de sortie pour le modèle quantifié
        quantization_type: Type de quantification ('dynamic' ou 'static')
    """
    
    print(f"📥 Chargement du modèle depuis {input_path}")
    model = torch.jit.load(input_path)
    
    original_size = os.path.getsize(input_path) / (1024 * 1024)
    print(f"   Taille originale: {original_size:.2f} MB")
    
    print(f"\n⚙️  Quantification {quantization_type} en cours...")
    
    if quantization_type == 'dynamic':
        # Quantification dynamique (recommandé pour les modèles de langage)
        # Quantifie les poids en INT8, les activations restent en FP32
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            qconfig_spec={torch.nn.Linear},  # Quantifier les couches linéaires
            dtype=torch.qint8
        )
    else:
        # Pour la quantification statique, il faudrait calibrer avec des données
        raise NotImplementedError("La quantification statique nécessite une calibration")
    
    print("💾 Sauvegarde du modèle quantifié...")
    quantized_model.save(output_path)
    
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)
    reduction = ((original_size - quantized_size) / original_size) * 100
    
    print(f"\n✅ Quantification terminée !")
    print(f"   Taille originale:  {original_size:.2f} MB")
    print(f"   Taille quantifiée: {quantized_size:.2f} MB")
    print(f"   Réduction:         {reduction:.1f}%")
    
    # Test rapide
    print("\n🧪 Test du modèle quantifié...")
    test_input = torch.randint(0, 50257, (1, 128))
    
    try:
        with torch.no_grad():
            output = quantized_model(test_input)
        print(f"   Shape de sortie: {output.shape}")
        print("   ✅ Le modèle quantifié fonctionne correctement!")
    except Exception as e:
        print(f"   ⚠️  Erreur lors du test: {e}")
    
    return output_path


def compare_models(original_path, quantized_path, num_tests=10):
    """
    Compare les performances et la précision entre les modèles
    """
    print("\n📊 Comparaison des modèles...")
    
    original = torch.jit.load(original_path)
    quantized = torch.jit.load(quantized_path)
    
    import time
    
    # Test de latence
    test_input = torch.randint(0, 50257, (1, 128))
    
    # Warm-up
    with torch.no_grad():
        _ = original(test_input)
        _ = quantized(test_input)
    
    # Benchmark original
    start = time.time()
    for _ in range(num_tests):
        with torch.no_grad():
            _ = original(test_input)
    original_time = (time.time() - start) / num_tests * 1000
    
    # Benchmark quantifié
    start = time.time()
    for _ in range(num_tests):
        with torch.no_grad():
            _ = quantized(test_input)
    quantized_time = (time.time() - start) / num_tests * 1000
    
    speedup = original_time / quantized_time
    
    print(f"\n⏱️  Latence moyenne (sur {num_tests} tests):")
    print(f"   Original:   {original_time:.2f} ms")
    print(f"   Quantifié:  {quantized_time:.2f} ms")
    print(f"   Speedup:    {speedup:.2f}x")
    
    # Test de précision (différence de sortie)
    with torch.no_grad():
        out_original = original(test_input)
        out_quantized = quantized(test_input)
    
    max_diff = torch.max(torch.abs(out_original - out_quantized)).item()
    mean_diff = torch.mean(torch.abs(out_original - out_quantized)).item()
    
    print(f"\n📏 Différence de sortie:")
    print(f"   Différence max:     {max_diff:.6f}")
    print(f"   Différence moyenne: {mean_diff:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Quantifier HessGPT pour optimiser la taille et la vitesse"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Chemin vers le modèle TorchScript (.ptl)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model_quantized.ptl",
        help="Chemin de sortie pour le modèle quantifié"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="dynamic",
        choices=["dynamic", "static"],
        help="Type de quantification"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Comparer les performances avec le modèle original"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Erreur: Le modèle {args.input} n'existe pas!")
        return
    
    quantize_model(
        input_path=args.input,
        output_path=args.output,
        quantization_type=args.type
    )
    
    if args.compare:
        compare_models(args.input, args.output)


if __name__ == "__main__":
    main()
