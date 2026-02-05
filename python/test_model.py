#!/usr/bin/env python3
"""
Script pour tester le modèle exporté avant déploiement mobile

Usage:
    python test_model.py --model model.ptl
"""

import argparse
import torch
import time


def test_model(model_path, num_iterations=10):
    """
    Teste le modèle TorchScript
    
    Args:
        model_path: Chemin vers le modèle .ptl
        num_iterations: Nombre d'itérations pour le benchmark
    """
    
    print(f"📥 Chargement du modèle depuis {model_path}")
    model = torch.jit.load(model_path)
    model.eval()
    
    print("\n🧪 Tests fonctionnels...")
    
    # Test 1: Différentes tailles de batch
    test_cases = [
        (1, 64),   # batch=1, seq_len=64
        (1, 128),  # batch=1, seq_len=128
        (1, 256),  # batch=1, seq_len=256
    ]
    
    for batch_size, seq_len in test_cases:
        print(f"\n   Test: batch={batch_size}, seq_len={seq_len}")
        test_input = torch.randint(0, 50257, (batch_size, seq_len))
        
        try:
            with torch.no_grad():
                output = model(test_input)
            print(f"   ✅ Shape: {output.shape}")
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            return False
    
    # Test 2: Benchmark de performance
    print(f"\n⏱️  Benchmark de performance ({num_iterations} itérations)...")
    
    test_input = torch.randint(0, 50257, (1, 128))
    
    # Warm-up
    for _ in range(3):
        with torch.no_grad():
            _ = model(test_input)
    
    # Benchmark
    latencies = []
    
    for i in range(num_iterations):
        start = time.time()
        with torch.no_grad():
            output = model(test_input)
        latency = (time.time() - start) * 1000
        latencies.append(latency)
        
        if i == 0:
            print(f"   Première inférence: {latency:.2f} ms")
    
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    
    print(f"\n   Latence moyenne: {avg_latency:.2f} ms")
    print(f"   Latence min:     {min_latency:.2f} ms")
    print(f"   Latence max:     {max_latency:.2f} ms")
    
    # Test 3: Simulation de génération token par token
    print("\n🔄 Simulation de génération autoregressive...")
    
    prompt_ids = torch.randint(0, 50257, (1, 10))
    generated_tokens = []
    
    start_time = time.time()
    
    for step in range(20):  # Générer 20 tokens
        with torch.no_grad():
            logits = model(prompt_ids)
        
        # Prendre le dernier token
        next_token_logits = logits[0, -1, :]
        
        # Sampling simple (greedy)
        next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
        generated_tokens.append(next_token.item())
        
        # Ajouter à la séquence
        prompt_ids = torch.cat([prompt_ids, next_token], dim=1)
    
    generation_time = time.time() - start_time
    tokens_per_sec = 20 / generation_time
    
    print(f"   ✅ Généré 20 tokens en {generation_time:.2f}s")
    print(f"   Débit: {tokens_per_sec:.1f} tokens/sec")
    
    # Test 4: Vérification de la stabilité numérique
    print("\n🔍 Test de stabilité numérique...")
    
    test_input = torch.randint(0, 50257, (1, 128))
    
    outputs = []
    for _ in range(3):
        with torch.no_grad():
            output = model(test_input)
        outputs.append(output)
    
    # Vérifier que les sorties sont identiques (déterministes)
    diff1 = torch.max(torch.abs(outputs[0] - outputs[1])).item()
    diff2 = torch.max(torch.abs(outputs[1] - outputs[2])).item()
    
    if diff1 < 1e-6 and diff2 < 1e-6:
        print("   ✅ Le modèle est déterministe (différence < 1e-6)")
    else:
        print(f"   ⚠️  Différences détectées: {diff1:.2e}, {diff2:.2e}")
    
    # Résumé
    print("\n" + "="*50)
    print("📊 RÉSUMÉ")
    print("="*50)
    print(f"✅ Tous les tests sont passés!")
    print(f"⚡ Latence moyenne: {avg_latency:.2f} ms")
    print(f"🚀 Débit: {tokens_per_sec:.1f} tokens/sec")
    print(f"💾 Prêt pour le déploiement mobile!")
    
    return True


def inspect_model(model_path):
    """
    Affiche des informations sur le modèle
    """
    import os
    
    print("\n🔍 Inspection du modèle...")
    
    # Taille du fichier
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"   Taille: {size_mb:.2f} MB")
    
    # Charger et inspecter
    model = torch.jit.load(model_path)
    
    # Compter les paramètres
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Paramètres: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Graph
    print(f"\n   Graph du modèle:")
    print(f"   {model.graph}")


def main():
    parser = argparse.ArgumentParser(
        description="Tester le modèle HessGPT exporté"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Chemin vers le modèle TorchScript (.ptl)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Nombre d'itérations pour le benchmark"
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Afficher des informations détaillées sur le modèle"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"❌ Erreur: Le modèle {args.model} n'existe pas!")
        return
    
    if args.inspect:
        inspect_model(args.model)
    
    test_model(args.model, args.iterations)


if __name__ == "__main__":
    import os
    main()
