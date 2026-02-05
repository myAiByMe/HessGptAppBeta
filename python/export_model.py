#!/usr/bin/env python3
"""
Export HessGPT vers TorchScript Mobile
Adapté à l'architecture exacte du pretrain_hessgpt.py
"""

import torch
import sys
import os
import argparse

# Ajouter le path vers Core/Model
sys.path.append('./Core/Model')
try:
    from HessGpt import HessGPT
except ImportError as e:
    print("❌ Import HessGPT échoué")
    raise e
print("✅ HessGPT chargé :", HessGPT)


# Configuration EXACTE de votre pretrain
HESSGPT_CONFIG = {
    'vocab_size':    50261,  # GPT2 50257 + 4 ChatLM tokens
    'embed_dim':     1280,
    'num_heads':     20,
    'num_layers':    20,
    'max_seq_len':   512,
    'dropout':       0.0,    # Pas de dropout en inférence
    'use_rope':      True,   # ✨ RoPE activé
}

def load_hessgpt_checkpoint(checkpoint_path):
    """
    Charge le checkpoint HessGPT depuis pretrain_hessgpt.py
    """
    print(f"📥 Chargement checkpoint : {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extraire la config du checkpoint (sauvegardée par pretrain)
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"   ✅ Config du checkpoint :")
        print(f"      Vocab size : {config['vocab_size']}")
        print(f"      Embed dim  : {config['embed_dim']}")
        print(f"      Layers     : {config['num_layers']}")
        print(f"      Heads      : {config['num_heads']}")
        print(f"      Use RoPE   : {config['use_rope']}")
    else:
        # Fallback sur la config par défaut
        config = HESSGPT_CONFIG
        print(f"   ⚠️  Pas de config dans checkpoint, utilisation config par défaut")
    
    # Créer le modèle
    print(f"\n🏗️  Création du modèle HessGPT...")
    model = HessGPT(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        max_seq_len=config['max_seq_len'],
        dropout=0.0,  # Toujours 0 en inférence
        use_rope=config.get('use_rope', True),
    )
    
    # Charger les poids
    print(f"   📦 Chargement des poids...")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Infos training
    if 'global_step' in checkpoint:
        print(f"   📊 Step d'entraînement : {checkpoint['global_step']:,}")
    if 'training_history' in checkpoint:
        history = checkpoint['training_history']
        if history.get('validations'):
            last_val = history['validations'][-1]
            print(f"   📈 Dernière PPL        : {last_val['perplexity']:.2f}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Paramètres totaux   : {total_params / 1e6:.1f}M")
    
    return model, config


class HessGPTMobileWrapper(torch.nn.Module):
    """
    Wrapper optimisé pour PyTorch Mobile
    """
    
    def __init__(self, model, max_seq_len=512):
        super().__init__()
        self.model = model
        self.max_seq_len = max_seq_len
        
        # Mode évaluation + désactivation gradients
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass pour l'inférence mobile
        
        Args:
            input_ids: [batch_size, seq_len] - IDs des tokens
            
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        with torch.no_grad():
            # HessGPT retourne (logits, loss) si targets fourni
            # Sinon juste logits
            output = self.model(input_ids, targets=None)
            
            # Si retour tuple, prendre premier élément
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
                
        return logits


def export_to_torchscript(checkpoint_path, output_path, quantize=False):
    """
    Export complet vers TorchScript Mobile
    """
    
    print("=" * 80)
    print("🚀 EXPORT HESSGPT → PYTORCH MOBILE")
    print("=" * 80)
    
    # 1. Charger le modèle
    model, config = load_hessgpt_checkpoint(checkpoint_path)
    
    # 2. Wrapper pour mobile
    print(f"\n📦 Création wrapper mobile...")
    wrapped = HessGPTMobileWrapper(model, config['max_seq_len'])
    
    # 3. Test du modèle
    print(f"\n🧪 Test du modèle...")
    test_input = torch.randint(0, config['vocab_size'], (1, 128))
    
    with torch.no_grad():
        test_output = wrapped(test_input)
    
    print(f"   ✅ Input shape  : {test_input.shape}")
    print(f"   ✅ Output shape : {test_output.shape}")
    print(f"   ✅ Expected     : [1, 128, {config['vocab_size']}]")
    
    # 4. Conversion TorchScript
    print(f"\n⚙️  Conversion TorchScript...")
    try:
        # Utiliser trace (plus simple et plus compatible mobile)
        traced = torch.jit.trace(wrapped, test_input)
        print(f"   ✅ Tracing réussi")
    except Exception as e:
        print(f"   ❌ Erreur tracing : {e}")
        print(f"   🔄 Tentative avec script...")
        traced = torch.jit.script(wrapped)
    
    # 5. Optimisation mobile
    print(f"\n⚡ Optimisation mobile...")
    optimized = torch.jit.optimize_for_inference(traced)
    
    # 6. Quantification optionnelle
    if quantize:
        print(f"\n🔢 Quantification INT8...")
        try:
            quantized = torch.quantization.quantize_dynamic(
                optimized,
                qconfig_spec={torch.nn.Linear},
                dtype=torch.qint8
            )
            optimized = quantized
            print(f"   ✅ Quantifié")
        except Exception as e:
            print(f"   ⚠️  Quantification échouée : {e}")
    
    # 7. Sauvegarde
    print(f"\n💾 Sauvegarde : {output_path}")
    optimized.save(output_path)
    
    # 8. Vérification
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"   ✅ Taille : {size_mb:.2f} MB")
    
    # 9. Test de chargement
    print(f"\n🧪 Test de chargement...")
    loaded = torch.jit.load(output_path)
    test_output2 = loaded(test_input)
    
    # Vérifier que les sorties sont identiques
    diff = torch.max(torch.abs(test_output - test_output2)).item()
    print(f"   ✅ Différence max : {diff:.2e}")
    
    if diff < 1e-5:
        print(f"   ✅ Export validé !")
    else:
        print(f"   ⚠️  Différence importante, vérifiez le modèle")
    
    print("\n" + "=" * 80)
    print("✅ EXPORT TERMINÉ !")
    print("=" * 80)
    print(f"\n📦 Fichier : {output_path}")
    print(f"📊 Taille  : {size_mb:.2f} MB")
    print(f"🎯 Prêt pour Android !")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Export HessGPT vers PyTorch Mobile"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Chemin vers HessGpt_pretrain.pt'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='model.ptl',
        help='Fichier de sortie .ptl'
    )
    parser.add_argument(
        '--quantize',
        action='store_true',
        help='Quantifier en INT8 (réduit la taille ~75%)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint introuvable : {args.checkpoint}")
        sys.exit(1)
    
    export_to_torchscript(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        quantize=args.quantize
    )
    
    print(f"\n📱 Prochaine étape :")
    print(f"   cp {args.output} ../app/src/main/assets/")


if __name__ == "__main__":
    main()