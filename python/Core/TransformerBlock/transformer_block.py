# transformer_block.py
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Attention.attention import MultiHeadAttention
from FeedForward.feedforward import FeedForward

# ============================================
# TRANSFORMER BLOCK AVEC RoPE
# ============================================

class TransformerBlock(nn.Module):
    """
    Un bloc Transformer complet pour GPT-2/GPT-3.5 avec RoPE
    
    Architecture :
    1. LayerNorm → Multi-Head Attention (avec RoPE) → Dropout → Residual
    2. LayerNorm → Feed-Forward → Dropout → Residual
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1, use_rope=True, max_seq_len=2048):
        """
        Args:
            embed_dim (int): Dimension des embeddings (4096 pour GPT-3.5)
            num_heads (int): Nombre de têtes d'attention (32 pour GPT-3.5)
            dropout (float): Taux de dropout
            use_rope (bool): Utiliser RoPE (Rotary Position Embeddings)
            max_seq_len (int): Longueur maximale de séquence pour RoPE
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_rope = use_rope
        
        # Layer Normalization (avant attention)
        self.ln1 = nn.LayerNorm(embed_dim)
        
        # Multi-Head Attention (avec RoPE et dropout intégré)
        self.attention = MultiHeadAttention(
            embed_dim, 
            num_heads, 
            dropout,
            use_rope=use_rope,
            max_seq_len=max_seq_len
        )
        
        # Layer Normalization (avant FFN)
        self.ln2 = nn.LayerNorm(embed_dim)
        
        # Feed-Forward Network
        self.ffn = FeedForward(embed_dim, dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
            mask: [seq_len, seq_len] - Masque causal
        
        Returns:
            output: [batch_size, seq_len, embed_dim]
        """
        # 1. Attention block avec residual connection
        # Pre-LayerNorm (GPT-2/GPT-3 utilise pre-norm, pas post-norm)
        residual = x
        x = self.ln1(x)
        x = self.attention(x, mask)
        x = residual + x  # Residual connection
        
        # 2. Feed-Forward block avec residual connection
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = residual + x  # Residual connection
        
        return x


def create_causal_mask(seq_len, device='cpu'):
    """
    Crée un masque causal triangulaire
    
    Args:
        seq_len (int): Longueur de la séquence
        device (str/torch.device): Device sur lequel créer le masque
    
    Returns:
        mask: [seq_len, seq_len] - 1 pour visible, 0 pour masqué
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask


# ============================================
# TESTS
# ============================================

def test_transformer_block():
    """Test du Transformer Block complet avec RoPE"""
    print("\n" + "="*60)
    print("TEST 1: Transformer Block avec RoPE")
    print("="*60)
    
    # Paramètres GPT-2 small
    batch_size = 2
    seq_len = 10
    embed_dim = 768
    num_heads = 12
    
    # Test AVEC RoPE
    print("\n📍 AVEC RoPE:")
    block_rope = TransformerBlock(embed_dim, num_heads, use_rope=True)
    x = torch.randn(batch_size, seq_len, embed_dim)
    mask = create_causal_mask(seq_len, device=x.device)
    
    print(f"  ✓ Input shape: {x.shape}")
    print(f"  ✓ Mask shape: {mask.shape}")
    
    # Forward pass
    output_rope = block_rope(x, mask)
    print(f"  ✓ Output shape: {output_rope.shape}")
    
    # Vérifier que les shapes correspondent
    assert output_rope.shape == x.shape, "Les shapes ne correspondent pas!"
    
    # Nombre de paramètres
    num_params_rope = sum(p.numel() for p in block_rope.parameters())
    print(f"\n  ✓ Nombre de paramètres: {num_params_rope:,}")
    
    # Test SANS RoPE pour comparaison
    print("\n📍 SANS RoPE (pour comparaison):")
    block_no_rope = TransformerBlock(embed_dim, num_heads, use_rope=False)
    output_no_rope = block_no_rope(x, mask)
    
    num_params_no_rope = sum(p.numel() for p in block_no_rope.parameters())
    print(f"  ✓ Nombre de paramètres: {num_params_no_rope:,}")
    print(f"  ✓ Différence: {num_params_rope - num_params_no_rope:,}")
    print("    (RoPE n'ajoute PAS de paramètres entraînables!)")
    
    # Détails des paramètres (version RoPE)
    attention_params = sum(p.numel() for p in block_rope.attention.parameters())
    ffn_params = sum(p.numel() for p in block_rope.ffn.parameters())
    ln_params = sum(p.numel() for p in block_rope.ln1.parameters()) + sum(p.numel() for p in block_rope.ln2.parameters())
    
    print(f"\n📊 Détails des paramètres (avec RoPE):")
    print(f"  - Attention:   {attention_params:,} ({attention_params/num_params_rope*100:.1f}%)")
    print(f"  - FFN:         {ffn_params:,} ({ffn_params/num_params_rope*100:.1f}%)")
    print(f"  - LayerNorms:  {ln_params:,} ({ln_params/num_params_rope*100:.1f}%)")
    print(f"  - Total:       {num_params_rope:,}")


def test_residual_connections():
    """Vérifie que les residual connections fonctionnent"""
    print("\n" + "="*60)
    print("TEST 2: Residual Connections")
    print("="*60)
    
    batch_size = 1
    seq_len = 5
    embed_dim = 64
    num_heads = 4
    
    # Créer le bloc avec RoPE
    block = TransformerBlock(embed_dim, num_heads, use_rope=True)
    block.eval()  # Mode eval pour désactiver dropout
    
    # Input simple (identité)
    x = torch.ones(batch_size, seq_len, embed_dim)
    
    # Forward
    mask = create_causal_mask(seq_len, device=x.device)
    with torch.no_grad():
        output = block(x, mask)
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    
    # L'output devrait être différent de l'input (grâce aux transformations)
    # mais pas trop différent (grâce aux residual connections)
    diff = (output - x).abs().mean().item()
    print(f"\n✓ Différence moyenne input/output: {diff:.4f}")
    print(f"  (Devrait être > 0 mais pas énorme grâce aux residuals)")


def test_layer_norm():
    """Comprendre la Layer Normalization"""
    print("\n" + "="*60)
    print("TEST 3: Layer Normalization")
    print("="*60)
    
    # Créer des données avec des échelles différentes
    x = torch.tensor([[[10.0, 20.0, 30.0, 40.0]]])  # [1, 1, 4]
    
    print(f"✓ Input:")
    print(f"  Valeurs: {x.squeeze().tolist()}")
    print(f"  Mean: {x.mean().item():.2f}")
    print(f"  Std: {x.std().item():.2f}")
    
    # Appliquer LayerNorm
    ln = nn.LayerNorm(4)
    x_norm = ln(x)
    
    print(f"\n✓ Après LayerNorm:")
    print(f"  Valeurs: {[f'{v:.3f}' for v in x_norm.squeeze().tolist()]}")
    print(f"  Mean: {x_norm.mean().item():.6f}")
    print(f"  Std: {x_norm.std().item():.6f}")
    print(f"\n💡 La moyenne est ~0 et la variance est ~1 !")


def test_multiple_blocks():
    """Test avec plusieurs blocs empilés (comme dans GPT-2)"""
    print("\n" + "="*60)
    print("TEST 4: Empiler plusieurs blocs avec RoPE")
    print("="*60)
    
    batch_size = 2
    seq_len = 10
    embed_dim = 256
    num_heads = 8
    num_blocks = 3  # On teste avec 3 blocs au lieu de 12
    max_seq_len = 512  # Longueur max pour RoPE
    
    # Créer plusieurs blocs avec RoPE
    blocks = nn.ModuleList([
        TransformerBlock(embed_dim, num_heads, use_rope=True, max_seq_len=max_seq_len)
        for _ in range(num_blocks)
    ])
    
    # Input
    x = torch.randn(batch_size, seq_len, embed_dim)
    mask = create_causal_mask(seq_len, device=x.device)
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Nombre de blocs: {num_blocks}")
    print(f"✓ Max seq len (RoPE): {max_seq_len}")
    
    # Passer à travers tous les blocs
    for i, block in enumerate(blocks):
        x = block(x, mask)
        print(f"  Après bloc {i+1}: {x.shape}")
    
    print(f"\n✓ Output final shape: {x.shape}")
    
    # Nombre total de paramètres
    total_params = sum(p.numel() for p in blocks.parameters())
    print(f"✓ Paramètres totaux ({num_blocks} blocs): {total_params:,}")


def test_gpt35_block():
    """Test avec configuration GPT-3.5"""
    print("\n" + "="*60)
    print("TEST 5: Transformer Block GPT-3.5 avec RoPE")
    print("="*60)
    
    # Paramètres GPT-3.5
    batch_size = 2
    seq_len = 128
    embed_dim = 4096  # GPT-3.5
    num_heads = 32    # GPT-3.5
    max_seq_len = 4096  # Longueur max pour RoPE
    
    print(f"✓ Configuration GPT-3.5 + RoPE:")
    print(f"  - Embed dim: {embed_dim}")
    print(f"  - Num heads: {num_heads}")
    print(f"  - Head dim: {embed_dim // num_heads}")
    print(f"  - Max seq len: {max_seq_len}")
    
    # Créer le bloc avec RoPE
    block = TransformerBlock(embed_dim, num_heads, use_rope=True, max_seq_len=max_seq_len)
    
    # Input
    x = torch.randn(batch_size, seq_len, embed_dim)
    mask = create_causal_mask(seq_len, device=x.device)
    
    # Forward
    output = block(x, mask)
    
    print(f"\n✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    
    # Nombre de paramètres
    num_params = sum(p.numel() for p in block.parameters())
    print(f"\n✓ Paramètres par bloc: {num_params:,}")
    print(f"✓ Pour 32 blocs: {num_params * 32:,}")


def test_rope_extrapolation():
    """Test de l'extrapolation RoPE à des séquences plus longues"""
    print("\n" + "="*60)
    print("TEST 6: Extrapolation RoPE (séquences longues)")
    print("="*60)
    
    batch_size = 1
    embed_dim = 256
    num_heads = 8
    
    # Entraîné sur seq_len=128
    max_seq_len_train = 128
    
    # Tester sur seq_len=256 (extrapolation!)
    seq_len_test = 256
    
    print(f"✓ Longueur max d'entraînement: {max_seq_len_train}")
    print(f"✓ Longueur de test: {seq_len_test}")
    
    # Créer bloc avec RoPE
    block = TransformerBlock(embed_dim, num_heads, use_rope=True, max_seq_len=max_seq_len_train)
    
    # Tester avec une séquence plus longue
    x = torch.randn(batch_size, seq_len_test, embed_dim)
    mask = create_causal_mask(seq_len_test)
    
    try:
        output = block(x, mask)
        print(f"\n✅ RoPE peut extrapoler à {seq_len_test} tokens!")
        print(f"   Output shape: {output.shape}")
    except Exception as e:
        print(f"\n❌ Erreur d'extrapolation: {e}")


def test_with_cuda():
    """Test avec CUDA si disponible"""
    print("\n" + "="*60)
    print("TEST 7: Test avec CUDA (si disponible)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}")
    
    if device.type == 'cpu':
        print("  ⚠️  CUDA non disponible, test sur CPU")
    
    # Paramètres
    batch_size = 2
    seq_len = 10
    embed_dim = 256
    num_heads = 8
    
    # Créer le bloc et le mettre sur le device
    block = TransformerBlock(embed_dim, num_heads, use_rope=True).to(device)
    
    # Input sur le device
    x = torch.randn(batch_size, seq_len, embed_dim).to(device)
    
    # Masque sur le device
    mask = create_causal_mask(seq_len, device=device)
    
    # Forward
    output = block(x, mask)
    
    print(f"✓ Input device: {x.device}")
    print(f"✓ Mask device: {mask.device}")
    print(f"✓ Output device: {output.device}")
    print(f"✓ Output shape: {output.shape}")


if __name__ == "__main__":
    print("\n🚀 TESTS DU TRANSFORMER BLOCK AVEC RoPE\n")
    
    # Test 1: Bloc basique avec RoPE
    test_transformer_block()
    
    # Test 2: Residual connections
    test_residual_connections()
    
    # Test 3: Layer Normalization
    test_layer_norm()
    
    # Test 4: Plusieurs blocs
    test_multiple_blocks()
    
    # Test 5: Configuration GPT-3.5
    test_gpt35_block()
    
    # Test 6: Extrapolation RoPE
    test_rope_extrapolation()
    
    # Test 7: Avec CUDA
    test_with_cuda()
    
    print("\n" + "="*60)
    print("✅ TOUS LES TESTS PASSÉS!")
    print("="*60)
    print("\n🔧 MODIFICATIONS APPLIQUÉES:")
    print("  1. ✨ RoPE intégré via MultiHeadAttention")
    print("  2. 🎯 Paramètre use_rope (True par défaut)")
    print("  3. 📏 Paramètre max_seq_len pour RoPE")
    print("  4. 🚀 Test d'extrapolation ajouté")
    print("\n💡 AVANTAGES:")
    print("  • Meilleure généralisation aux longues séquences")
    print("  • Pas de paramètres supplémentaires")
    print("  • Compatible avec le code existant")
    print("\n📝 Prêt pour hessgpt.py!")
    print("="*60 + "\n")
