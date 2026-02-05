# 🔧 Guide d'Intégration avec HessGPT

Ce guide explique comment intégrer précisément votre modèle HessGPT (du repository HessGpt_RoPE) avec cette application mobile.

## 📦 Structure de votre modèle HessGPT

Basé sur votre architecture avec RoPE (Rotary Position Embeddings), voici comment adapter l'export.

### 1. Comprendre votre architecture

Votre modèle HessGPT utilise probablement:
- **Embeddings de tokens**: Transformation des IDs en vecteurs
- **RoPE**: Rotary Position Embeddings au lieu des embeddings de position classiques
- **Couches Transformer**: Attention multi-têtes + Feed-Forward
- **LM Head**: Projection vers le vocabulaire

### 2. Adapter export_model.py

Modifiez `python/export_model.py` pour charger votre architecture exacte:

```python
import sys
sys.path.append("/path/to/HessGpt_RoPE")  # Votre repo

# Importez votre modèle
from Core.model import HessGPT  # Ou le nom de votre classe principale
# from Core.config import ModelConfig  # Si vous avez une config séparée

def load_hessgpt_model(checkpoint_path):
    """
    Charge votre modèle HessGPT depuis un checkpoint
    """
    # Charger le checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extraire la config (adaptez selon votre structure)
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Créer manuellement la config
        config = {
            'vocab_size': 50257,      # Taille de votre vocabulaire
            'n_embd': 768,            # Dimension d'embedding (pour 0.5B, probablement 768 ou 1024)
            'n_head': 12,             # Nombre de têtes d'attention
            'n_layer': 12,            # Nombre de couches transformer
            'max_seq_len': 2048,      # Longueur max de contexte
            'dropout': 0.0,           # Pas de dropout en inférence
            # Paramètres RoPE
            'rope_theta': 10000.0,    # Base theta pour RoPE
            'rope_scaling': None,     # Facteur de scaling
        }
    
    # Créer le modèle
    model = HessGPT(**config)
    
    # Charger les poids
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model
```

### 3. Wrapper pour l'inférence mobile

Le wrapper doit gérer l'inférence autoregressive:

```python
class HessGPTMobileWrapper(torch.nn.Module):
    """
    Wrapper optimisé pour mobile avec génération autoregressive
    """
    
    def __init__(self, model, max_seq_len=512):
        super().__init__()
        self.model = model
        self.max_seq_len = max_seq_len
        
        # Mettre en mode éval et désactiver gradients
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    @torch.jit.export
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass pour l'inférence
        
        Args:
            input_ids: [batch_size, seq_len] - IDs des tokens
            
        Returns:
            logits: [batch_size, seq_len, vocab_size] - Logits pour chaque position
        """
        with torch.no_grad():
            # Votre modèle retourne probablement (logits, loss) ou juste logits
            output = self.model(input_ids)
            
            # Extraire les logits selon votre structure
            if isinstance(output, tuple):
                logits = output[0]
            elif isinstance(output, dict):
                logits = output['logits']
            else:
                logits = output
                
        return logits
    
    @torch.jit.export
    def generate_next_token(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Génère le prochain token (optimisé pour mobile)
        
        Args:
            input_ids: [1, seq_len]
            
        Returns:
            logits: [vocab_size] - Logits pour le prochain token
        """
        logits = self.forward(input_ids)
        # Retourner uniquement les logits du dernier token
        return logits[0, -1, :]
```

### 4. Export avec votre modèle

Script complet d'export:

```python
def export_hessgpt_to_mobile(checkpoint_path, output_path):
    """
    Export complet de HessGPT vers mobile
    """
    print("🔄 Chargement du modèle HessGPT...")
    model = load_hessgpt_model(checkpoint_path)
    
    print("📦 Création du wrapper mobile...")
    wrapped_model = HessGPTMobileWrapper(model, max_seq_len=512)
    
    print("🔍 Test du modèle...")
    # Test avec une séquence exemple
    test_input = torch.randint(0, model.config.vocab_size, (1, 128))
    test_output = wrapped_model(test_input)
    print(f"   Output shape: {test_output.shape}")
    
    print("⚙️  Conversion vers TorchScript...")
    # Utiliser tracing
    traced = torch.jit.trace(wrapped_model, test_input)
    
    # OU utiliser scripting si votre modèle a des contrôles de flux
    # scripted = torch.jit.script(wrapped_model)
    
    print("⚡ Optimisation pour mobile...")
    optimized = torch.jit.optimize_for_inference(traced)
    
    print(f"💾 Sauvegarde vers {output_path}...")
    optimized.save(output_path)
    
    # Vérifier la taille
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Export terminé ! Taille: {size_mb:.2f} MB")
    
    # Estimer le nombre de paramètres
    params = sum(p.numel() for p in model.parameters())
    print(f"📊 Paramètres: {params:,} ({params/1e9:.2f}B)")
```

### 5. Optimisations spécifiques RoPE

Pour optimiser RoPE sur mobile:

```python
# Dans votre modèle, pré-calculez les fréquences RoPE
@torch.jit.script
def precompute_rope_freqs(
    dim: int,
    max_seq_len: int, 
    theta: float = 10000.0
) -> torch.Tensor:
    """
    Pré-calcule les fréquences RoPE (fait une seule fois)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # e^(i * theta)
```

### 6. Gestion du KV-Cache (optionnel mais recommandé)

Pour accélérer la génération, implémentez le KV-cache:

```python
class HessGPTWithCache(torch.nn.Module):
    """
    Version avec KV-cache pour génération plus rapide
    """
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.cache = None
    
    def forward(self, input_ids, use_cache=False):
        if use_cache and self.cache is not None:
            # Utiliser le cache pour les tokens déjà traités
            output = self.model(input_ids, past_key_values=self.cache)
            self.cache = output.past_key_values
        else:
            output = self.model(input_ids)
            if use_cache:
                self.cache = output.past_key_values if hasattr(output, 'past_key_values') else None
        
        return output.logits
    
    def reset_cache(self):
        self.cache = None
```

### 7. Tester l'export

```bash
cd python

# Export
python export_model.py \
    --checkpoint /path/to/your/hessgpt_checkpoint.pt \
    --output ../app/src/main/assets/model.ptl

# Test
python test_model.py \
    --model ../app/src/main/assets/model.ptl \
    --inspect

# Quantification
python quantize_model.py \
    --input ../app/src/main/assets/model.ptl \
    --output ../app/src/main/assets/model_q8.ptl \
    --compare
```

### 8. Adapter le tokenizer

Si vous utilisez un tokenizer GPT-2 ou similaire:

```bash
# Depuis HuggingFace
python create_tokenizer.py \
    --from-hf gpt2 \
    --output ../app/src/main/assets/tokenizer.json

# Ou depuis votre tokenizer local
python create_tokenizer.py \
    --from-hf /path/to/HessGpt_RoPE/tokenizer \
    --output ../app/src/main/assets/tokenizer.json
```

### 9. Ajustements Android

Dans `ModelManager.kt`, vérifiez que les hyperparamètres correspondent:

```kotlin
// Doit correspondre à votre modèle
private val maxSeqLength = 512  // Votre max_seq_len
private val temperature = 0.7f
private val topK = 40
private val topP = 0.9f
```

## 🎯 Checklist d'intégration

- [ ] Export du modèle réussi sans erreurs
- [ ] Test du modèle avec `test_model.py` OK
- [ ] Tokenizer correspond au modèle
- [ ] Taille du modèle < 200MB (quantifié si nécessaire)
- [ ] Test sur un appareil Android réel
- [ ] Performances acceptables (>2 tokens/sec)
- [ ] Génération cohérente

## 🔍 Debugging

### Le modèle ne charge pas
```bash
# Vérifier la compatibilité TorchScript
python -c "import torch; m = torch.jit.load('model.ptl'); print('OK')"
```

### Les tokens sont incorrects
```python
# Vérifier le tokenizer
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("...")
print(tok.encode("Bonjour"))
print(tok.decode([...]))
```

### Erreur OutOfMemory
- Réduire `max_seq_len` à 256 ou 128
- Quantifier en INT8
- Tester sur appareil avec plus de RAM

## 📞 Support

Pour des questions spécifiques à HessGPT:
1. Vérifiez votre architecture dans `Core/model.py`
2. Regardez les exemples d'entraînement dans `PreTrain.py`
3. Adaptez le wrapper selon votre structure exacte

Bonne intégration ! 🚀
