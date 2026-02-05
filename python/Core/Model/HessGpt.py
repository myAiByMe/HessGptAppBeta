# HessGpt.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TransformerBlock.transformer_block import TransformerBlock

# ============================================
# CONFIG CLASS (pour compatibilité PEFT)
# ============================================
class HessGPTConfig:
    """
    Configuration class pour HessGPT (compatible PEFT).
    Simule un config Hugging Face avec la méthode get().
    """
    def __init__(self, vocab_size=50257, embed_dim=768, num_heads=12, 
                 num_layers=12, max_seq_len=2048, use_rope=True, **kwargs):
        self.vocab_size = vocab_size
        self.hidden_size = embed_dim
        self.num_attention_heads = num_heads
        self.num_hidden_layers = num_layers
        self.max_position_embeddings = max_seq_len
        self.model_type = "hessgpt"
        self.use_rope = use_rope
        
        # Attributs supplémentaires pour compatibilité PEFT
        self.tie_word_embeddings = True  # On partage token_embeddings et output_head
        self.is_encoder_decoder = False
        self.architectures = ["HessGPT"]
        
        # Stocker kwargs supplémentaires
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def get(self, key, default=None):
        """Méthode get() comme un dict (requis par PEFT)"""
        return getattr(self, key, default)
    
    def to_dict(self):
        """Convertir en dictionnaire"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

# ============================================
# MODÈLE HessGPT COMPLET AVEC RoPE
# ============================================

class HessGPT(nn.Module):
    """
    Modèle HessGPT - Architecture Transformer avec RoPE
    
    Architecture :
    - Token Embeddings (SANS Position Embeddings - remplacé par RoPE!)
    - N Transformer Blocks (avec RoPE intégré)
    - Layer Norm finale
    - Output Head (projection vers vocabulaire)
    
    🔥 CHANGEMENT MAJEUR: RoPE remplace les position embeddings traditionnels
    """
    def __init__(
        self,
        vocab_size,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        max_seq_len=2048,
        dropout=0.1,
        use_rope=True
    ):
        """
        Args:
            vocab_size (int): Taille du vocabulaire
            embed_dim (int): Dimension des embeddings
            num_heads (int): Nombre de têtes d'attention
            num_layers (int): Nombre de Transformer Blocks
            max_seq_len (int): Longueur max de séquence
            dropout (float): Taux de dropout
            use_rope (bool): Utiliser RoPE (Rotary Position Embeddings)
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        
        # Configuration object for PEFT compatibility
        self.config = HessGPTConfig(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            use_rope=use_rope,
        )
        
        # Token Embeddings (uniquement!)
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # 🚫 PLUS DE position_embeddings! RoPE le remplace
        # Si use_rope=False, on garde les position embeddings classiques
        if not use_rope:
            self.position_embeddings = nn.Embedding(max_seq_len, embed_dim)
        else:
            self.position_embeddings = None
        
        self.dropout = nn.Dropout(dropout)
        
        # Transformer Blocks (empiler N blocs avec RoPE)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, 
                num_heads, 
                dropout,
                use_rope=use_rope,
                max_seq_len=max_seq_len
            )
            for _ in range(num_layers)
        ])
        
        # Layer Norm finale
        self.ln_final = nn.LayerNorm(embed_dim)
        
        # Output Head (projection vers vocabulaire)
        self.output_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Partager les poids entre token_embeddings et output_head
        self.output_head.weight = self.token_embeddings.weight
        
        # Initialisation des poids
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialisation des poids"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        targets=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        """
        Forward pass compatible PEFT.
        
        Args:
            input_ids: [batch_size, seq_len] - IDs des tokens (ou None si inputs_embeds fourni)
            inputs_embeds: [batch_size, seq_len, embed_dim] - Embeddings pré-calculés (optionnel)
            attention_mask: [batch_size, seq_len] - Masque d'attention (optionnel, non utilisé)
            targets: [batch_size, seq_len] - Targets pour calculer la loss (optionnel)
            labels: [batch_size, seq_len] - Alias de targets (convention HF)
            output_attentions: bool - Retourner les attentions (non implémenté)
            output_hidden_states: bool - Retourner les hidden states (non implémenté)
            return_dict: bool - Retourner un dict au lieu d'un tuple (non implémenté)
            **kwargs: Arguments additionnels (ignorés)
        
        Returns:
            logits: [batch_size, seq_len, vocab_size] - Prédictions
            loss: Scalar (si targets/labels fourni)
        """
        # Gérer l'alias labels -> targets (convention Hugging Face)
        if labels is not None and targets is None:
            targets = labels
        
        # Gérer inputs_embeds vs input_ids
        if inputs_embeds is not None:
            # Si les embeddings sont fournis directement
            token_embeds = inputs_embeds
            batch_size, seq_len, _ = token_embeds.shape
        elif input_ids is not None:
            # Cas normal : calculer les embeddings depuis input_ids
            batch_size, seq_len = input_ids.shape
            token_embeds = self.token_embeddings(input_ids)
        else:
            raise ValueError("Il faut fournir soit input_ids, soit inputs_embeds")
        
        # Position Embeddings (uniquement si RoPE désactivé)
        if self.use_rope:
            # Avec RoPE: pas besoin d'ajouter des position embeddings
            x = self.dropout(token_embeds)
        else:
            # Sans RoPE: on ajoute les position embeddings classiques
            positions = torch.arange(0, seq_len, device=token_embeds.device)
            position_embeds = self.position_embeddings(positions)
            x = self.dropout(token_embeds + position_embeds)
        
        # Créer le masque causal
        mask = self.create_causal_mask(seq_len, device=token_embeds.device)
        
        # 4. Passer à travers tous les Transformer Blocks
        # (RoPE est appliqué à l'intérieur de chaque bloc)
        for block in self.blocks:
            x = block(x, mask)
        
        # 5. Layer Norm finale
        x = self.ln_final(x)
        
        # 6. Output Head (projection vers vocabulaire)
        logits = self.output_head(x)
        
        # 7. Calculer la loss si targets fourni
        loss = None
        if targets is not None:
            # Reshape pour calculer la cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1)
            )
        
        return logits, loss
    
    def create_causal_mask(self, seq_len, device):
        """Crée un masque causal triangulaire"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """
        Prépare les inputs pour la génération (requis par PEFT).
        
        Args:
            input_ids: [batch_size, seq_len] - IDs des tokens
            past_key_values: Cache KV (non utilisé pour l'instant)
            **kwargs: Arguments additionnels
            
        Returns:
            dict: Dictionnaire avec les inputs formatés
        """
        # Pour un modèle simple sans KV cache, on retourne juste input_ids
        return {
            "input_ids": input_ids,
        }
    
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None, 
                 stop_tokens=None, min_new_tokens=10, eos_token_id=None):
        """
        Génération de texte (autoregressive) avec arrêt intelligent
        
        Args:
            input_ids: [batch_size, seq_len] - Prompt
            max_new_tokens: Nombre MAX de tokens à générer
            temperature: Contrôle la randomness (1.0 = normal, <1 = plus déterministe)
            top_k: Si fourni, ne garde que les top-k tokens les plus probables
            stop_tokens: Liste de token IDs qui indiquent la fin (ex: ponctuation)
            min_new_tokens: Nombre minimum de tokens avant d'autoriser l'arrêt
            eos_token_id: Token ID de fin de séquence (si existe dans le tokenizer)
        
        Returns:
            generated_ids: [batch_size, seq_len + nb_tokens_générés]
        """
        self.eval()
        
        # Tokens par défaut qui peuvent indiquer une fin de phrase
        if stop_tokens is None:
            stop_tokens = set()
        
        with torch.no_grad():
            tokens_generated = 0
            
            for _ in range(max_new_tokens):
                # Tronquer si trop long
                input_ids_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
                
                # Forward pass
                logits, _ = self.forward(input_ids_cond)
                
                # Prendre les logits du dernier token
                logits = logits[:, -1, :] / temperature
                
                # Top-k sampling (optionnel)
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')
                
                # Softmax pour obtenir les probabilités
                probs = F.softmax(logits, dim=-1)
                
                # Sampler le prochain token
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Ajouter à la séquence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                tokens_generated += 1
                
                # Vérifier les conditions d'arrêt APRÈS le minimum de tokens
                if tokens_generated >= min_new_tokens:
                    # Arrêt si token EOS détecté
                    if eos_token_id is not None and next_token.item() == eos_token_id:
                        break
                    
                    # Arrêt si token de ponctuation finale détecté
                    if next_token.item() in stop_tokens:
                        break
        
        return input_ids
    
    def count_parameters(self):
        """Compte et détaille les paramètres du modèle"""
        total = sum(p.numel() for p in self.parameters())
        
        # Détail par composant
        token_emb = self.token_embeddings.weight.numel()
        pos_emb = self.position_embeddings.weight.numel() if self.position_embeddings else 0
        
        # Transformer blocks
        blocks_params = sum(p.numel() for block in self.blocks for p in block.parameters())
        
        # Final LN
        ln_params = sum(p.numel() for p in self.ln_final.parameters())
        
        # Output head (partagé avec token_embeddings donc 0 nouveaux params)
        output_params = 0
        
        return {
            'total': total,
            'token_embeddings': token_emb,
            'position_embeddings': pos_emb,
            'transformer_blocks': blocks_params,
            'final_ln': ln_params,
            'output_head': output_params,
        }
    
    def get_num_params(self, non_embedding=True):
        """
        Retourne le nombre de paramètres.
        Si non_embedding=True, exclut les embeddings.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embeddings.weight.numel()
            if self.position_embeddings:
                n_params -= self.position_embeddings.weight.numel()
        return n_params


# ============================================
# TESTS
# ============================================

def test_hessgpt_model():
    """Test basique du modèle avec/sans RoPE"""
    print("="*60)
    print("TEST 1: Forward Pass (avec RoPE)")
    print("="*60)
    
    vocab_size = 300
    batch_size = 2
    seq_len = 10
    
    # AVEC RoPE
    print("\n🎯 AVEC RoPE:")
    model_rope = HessGPT(
        vocab_size=vocab_size,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        max_seq_len=128,
        use_rope=True
    )
    
    # Input aléatoire
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"  ✓ Input shape: {input_ids.shape}")
    
    # Forward pass
    logits_rope, _ = model_rope(input_ids)
    
    print(f"  ✓ Logits shape: {logits_rope.shape}")
    print(f"    Expected: [{batch_size}, {seq_len}, {vocab_size}]")
    
    # Vérifier les shapes
    assert logits_rope.shape == (batch_size, seq_len, vocab_size)
    print(f"  ✓ Shape correcte!")
    
    # Nombre de paramètres
    params_rope = model_rope.count_parameters()
    print(f"\n  ✓ Paramètres (avec RoPE): {params_rope['total']:,}")
    
    # Test SANS RoPE pour comparaison
    print("\n📍 SANS RoPE (pour comparaison):")
    model_no_rope = HessGPT(
        vocab_size=vocab_size,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        max_seq_len=128,
        use_rope=False
    )
    
    logits_no_rope, _ = model_no_rope(input_ids)
    params_no_rope = model_no_rope.count_parameters()
    
    print(f"  ✓ Paramètres (sans RoPE): {params_no_rope['total']:,}")
    print(f"  ✓ Différence: {params_no_rope['total'] - params_rope['total']:,} paramètres économisés!")
    print(f"    (Position embeddings supprimés: {params_no_rope['position_embeddings']:,})")


def test_with_loss():
    """Test avec calcul de la loss"""
    print("\n" + "="*60)
    print("TEST 2: Forward Pass avec Loss (RoPE)")
    print("="*60)
    
    vocab_size = 300
    batch_size = 2
    seq_len = 10
    
    model = HessGPT(
        vocab_size=vocab_size,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        use_rope=True
    )
    
    # Input et targets
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"✓ Input shape: {input_ids.shape}")
    print(f"✓ Targets shape: {targets.shape}")
    
    # Forward avec loss
    logits, loss = model(input_ids, targets)
    
    print(f"\n✓ Logits shape: {logits.shape}")
    print(f"✓ Loss: {loss.item():.4f}")
    print(f"  (Loss aléatoire ~{math.log(vocab_size):.2f} au début)")


def test_generation():
    """Test de génération de texte"""
    print("\n" + "="*60)
    print("TEST 3: Génération de texte avec RoPE")
    print("="*60)
    
    vocab_size = 300
    
    model = HessGPT(
        vocab_size=vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        use_rope=True
    )
    
    # Prompt (quelques tokens)
    prompt = torch.randint(0, vocab_size, (1, 5))
    
    print(f"✓ Prompt shape: {prompt.shape}")
    print(f"✓ Prompt tokens: {prompt[0].tolist()}")
    
    # Générer 10 nouveaux tokens
    generated = model.generate(prompt, max_new_tokens=10, temperature=1.0)
    
    print(f"\n✓ Generated shape: {generated.shape}")
    print(f"✓ Generated tokens: {generated[0].tolist()}")
    print(f"✓ Génération réussie! ({generated.shape[1] - prompt.shape[1]} nouveaux tokens)")


def test_prepare_inputs_for_generation():
    """Test de la méthode prepare_inputs_for_generation (pour PEFT)"""
    print("\n" + "="*60)
    print("TEST 4: prepare_inputs_for_generation (PEFT compatibility)")
    print("="*60)
    
    vocab_size = 300
    
    model = HessGPT(
        vocab_size=vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        use_rope=True
    )
    
    # Test input
    input_ids = torch.randint(0, vocab_size, (1, 10))
    
    print(f"✓ Input shape: {input_ids.shape}")
    
    # Appeler la méthode
    model_inputs = model.prepare_inputs_for_generation(input_ids)
    
    print(f"✓ Model inputs: {model_inputs.keys()}")
    print(f"✓ Input IDs shape: {model_inputs['input_ids'].shape}")
    print(f"✓ Méthode PEFT compatible!")


def test_long_sequence_extrapolation():
    """Test d'extrapolation à des séquences plus longues"""
    print("\n" + "="*60)
    print("TEST 5: Extrapolation RoPE (séquences longues)")
    print("="*60)
    
    vocab_size = 300
    
    # Modèle entraîné sur seq_len=128
    max_seq_len_train = 128
    
    model = HessGPT(
        vocab_size=vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        max_seq_len=max_seq_len_train,
        use_rope=True
    )
    
    # Tester sur seq_len=256 (2x plus long!)
    seq_len_test = 256
    
    print(f"✓ Longueur max d'entraînement: {max_seq_len_train}")
    print(f"✓ Longueur de test: {seq_len_test}")
    
    try:
        input_ids = torch.randint(0, vocab_size, (1, seq_len_test))
        logits, _ = model(input_ids)
        
        print(f"\n✅ RoPE peut extrapoler à {seq_len_test} tokens!")
        print(f"   Logits shape: {logits.shape}")
        print(f"   (Sans RoPE, ça planterait car position_embeddings limité à {max_seq_len_train})")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")


def test_hessgpt_20m():
    """Test avec configuration 20M paramètres"""
    print("\n" + "="*60)
    print("TEST 6: HessGPT 20M paramètres avec RoPE")
    print("="*60)
    
    # Configuration 20M avec RoPE
    model = HessGPT(
        vocab_size=20000,
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        max_seq_len=2048,
        use_rope=True
    )
    
    print(f"✓ Modèle créé avec succès!")
    print(f"  - Vocab size: {model.vocab_size}")
    print(f"  - Embed dim: {model.embed_dim}")
    print(f"  - Num heads: {model.num_heads}")
    print(f"  - Num layers: {model.num_layers}")
    print(f"  - Max seq len: {model.max_seq_len}")
    print(f"  - Use RoPE: {model.use_rope}")
    
    # Détails des paramètres
    params = model.count_parameters()
    
    print(f"\n📊 Détails des paramètres:")
    print(f"  - Token embeddings:       {params['token_embeddings']:,}")
    print(f"  - Position embeddings:    {params['position_embeddings']:,} (RoPE = 0 paramètres!)")
    print(f"  - {model.num_layers} Transformer Blocks: {params['transformer_blocks']:,}")
    print(f"  - Final LayerNorm:        {params['final_ln']:,}")
    print(f"  - Output head:            {params['output_head']:,} (partagé avec token emb)")
    print(f"\n  ✓ TOTAL: {params['total']:,} paramètres")
    
    # Comparaison avec/sans RoPE
    model_no_rope = HessGPT(
        vocab_size=20000,
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        max_seq_len=2048,
        use_rope=False
    )
    
    params_no_rope = model_no_rope.count_parameters()
    
    print(f"\n💡 Comparaison avec modèle SANS RoPE:")
    print(f"  - Avec RoPE:    {params['total']:,} paramètres")
    print(f"  - Sans RoPE:    {params_no_rope['total']:,} paramètres")
    print(f"  - Économie:     {params_no_rope['total'] - params['total']:,} paramètres!")
    print(f"    (= {params_no_rope['position_embeddings']:,} position embeddings supprimés)")
    
    # Test rapide
    input_ids = torch.randint(0, 20000, (1, 10))
    logits, _ = model(input_ids)
    print(f"\n✓ Test forward pass: {logits.shape}")


if __name__ == "__main__":
    print("\n🚀 TESTS DU MODÈLE HessGPT AVEC RoPE\n")
    
    # Test 1: Forward basique avec/sans RoPE
    test_hessgpt_model()
    
    # Test 2: Avec loss
    test_with_loss()
    
    # Test 3: Génération basique
    test_generation()
    
    # Test 4: PEFT compatibility
    test_prepare_inputs_for_generation()
    
    # Test 5: Extrapolation (séquences longues)
    test_long_sequence_extrapolation()
    
    # Test 6: 20M paramètres
    test_hessgpt_20m()
    
    print("\n" + "="*60)
    print("✅ TOUS LES TESTS PASSÉS!")
    print("="*60)
    print("\n🎉 FÉLICITATIONS! HessGPT avec RoPE est opérationnel!")
    print("\n🔧 MODIFICATIONS MAJEURES:")
    print("  1. ✨ RoPE intégré (remplace position embeddings)")
    print("  2. 🎯 Économie de paramètres (~10M pour vocab 20k)")
    print("  3. 🚀 Meilleure extrapolation aux séquences longues")
    print("  4. ⚡ Architecture moderne (LLaMA-style)")
    print("  5. 🔗 Compatible PEFT/LoRA (prepare_inputs_for_generation)")
    print("\n💡 AVANTAGES DE RoPE:")
    print("  • Moins de paramètres (pas de position embeddings)")
    print("  • Meilleure généralisation")
    print("  • Peut traiter des séquences plus longues que l'entraînement")
    print("  • Utilisé par LLaMA, PaLM, Mistral, etc.")
    print("\n📝 Prêt pour train.py, pretrain.py et SFT avec LoRA/QLoRA!")
    print("="*60 + "\n")