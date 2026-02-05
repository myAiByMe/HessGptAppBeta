import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
import os
from tqdm import tqdm
import pickle

# ============================================
# DATASET
# ============================================

class TextDataset(Dataset):
    """
    Dataset pour l'entraînement de GPT-2/HessGPT
    Prend un long texte et le découpe en séquences
    """
    def __init__(self, text, tokenizer, seq_len=128):
        """
        Args:
            text (str): Texte brut
            tokenizer: Votre tokenizer BPE
            seq_len (int): Longueur des séquences (128-256 pour commencer)
        """
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        
        # Encoder tout le texte
        print("Encodage du texte...")
        self.tokens = tokenizer.encoder(text)
        print(f"✓ {len(self.tokens)} tokens encodés")
        
        # Calculer le nombre de séquences possibles
        self.num_sequences = len(self.tokens) // seq_len
        
        # Tronquer pour avoir un multiple de seq_len
        self.tokens = self.tokens[:self.num_sequences * seq_len]
        
    def __len__(self):
        return self.num_sequences - 1  # -1 car on a besoin de input + target
    
    def __getitem__(self, idx):
        """
        Retourne une séquence et son target (décalé de 1)
        """
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len
        
        # Input: tokens[start:end]
        input_ids = torch.tensor(self.tokens[start_idx:end_idx], dtype=torch.long)
        
        # Target: tokens[start+1:end+1] (décalé de 1 pour next token prediction)
        target_ids = torch.tensor(self.tokens[start_idx+1:end_idx+1], dtype=torch.long)
        
        return input_ids, target_ids


# ============================================
# TRAINER
# ============================================

class GPT2Trainer:
    """
    Classe pour entraîner GPT-2/HessGPT avec RoPE
    Gère l'optimisation, la loss, les checkpoints, etc.
    """
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset=None,
        learning_rate=3e-4,
        batch_size=4,
        num_epochs=10,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir='./checkpoints',
        gradient_accumulation_steps=1,
        warmup_steps=100
    ):
        """
        Args:
            model: Votre modèle GPT-2/HessGPT (avec RoPE!)
            train_dataset: Dataset d'entraînement
            val_dataset: Dataset de validation (optionnel)
            learning_rate: Taux d'apprentissage (3e-4 est bon pour GPT)
            batch_size: Taille des batches (4-8 sur Colab gratuit)
            num_epochs: Nombre d'epochs
            device: 'cuda' ou 'cpu'
            checkpoint_dir: Dossier pour sauvegarder les checkpoints
            gradient_accumulation_steps: Accumuler les gradients pour simuler un plus gros batch
            warmup_steps: Nombre de steps pour le warmup du learning rate
        """
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        
        # Créer le dossier de checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0  # 0 pour éviter les problèmes sur Colab
        )
        
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
        
        # Optimizer (AdamW comme dans GPT-2 original)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),  # Valeurs utilisées dans GPT-2
            weight_decay=0.1
        )
        
        # Learning rate scheduler avec warmup
        total_steps = len(self.train_loader) * num_epochs // gradient_accumulation_steps
        
        # Warmup linéaire puis cosine decay
        self.scheduler = self._get_scheduler(learning_rate, total_steps, warmup_steps)
        
        # Historique
        self.train_losses = []
        self.val_losses = []
        self.global_step = 0
        
    def _get_scheduler(self, lr, total_steps, warmup_steps):
        """Crée un scheduler avec warmup linéaire puis cosine decay"""
        def lr_lambda(step):
            if step < warmup_steps:
                # Warmup linéaire
                return step / warmup_steps
            else:
                # Cosine decay après warmup
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self, epoch):
        """Entraîne le modèle pour une epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        for batch_idx, (input_ids, target_ids) in enumerate(pbar):
            # Déplacer sur le device
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            # Forward pass
            logits, loss = self.model(input_ids, target_ids)
            
            # Normaliser la loss par gradient_accumulation_steps
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights tous les gradient_accumulation_steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping (important pour la stabilité)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update weights
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Tracking (multiplier par gradient_accumulation_steps pour avoir la vraie loss)
            actual_loss = loss.item() * self.gradient_accumulation_steps
            total_loss += actual_loss
            avg_loss = total_loss / (batch_idx + 1)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{actual_loss:.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}',
                'step': self.global_step
            })
        
        avg_epoch_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_epoch_loss)
        
        return avg_epoch_loss
    
    def validate(self):
        """Valide le modèle"""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for input_ids, target_ids in tqdm(self.val_loader, desc="Validation"):
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                logits, loss = self.model(input_ids, target_ids)
                total_loss += loss.item()
        
        avg_val_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_val_loss)
        
        return avg_val_loss
    
    def save_checkpoint(self, epoch, loss):
        """Sauvegarde un checkpoint"""
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_epoch_{epoch+1}_loss_{loss:.4f}.pt'
        )
        
        torch.save({
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, checkpoint_path)
        
        print(f"✓ Checkpoint sauvegardé: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Charge un checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.global_step = checkpoint.get('global_step', 0)
        
        print(f"✓ Checkpoint chargé: {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']+1}")
        print(f"  Loss: {checkpoint['loss']:.4f}")
        print(f"  Global step: {self.global_step}")
        
        return checkpoint['epoch']
    
    def train(self, save_every=1):
        """
        Boucle d'entraînement complète
        
        Args:
            save_every (int): Sauvegarder un checkpoint tous les N epochs
        """
        print("\n" + "="*60)
        print("DÉBUT DE L'ENTRAÎNEMENT")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Nombre d'epochs: {self.num_epochs}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.train_loader.batch_size * self.gradient_accumulation_steps}")
        print(f"Batches par epoch: {len(self.train_loader)}")
        
        # Afficher les détails du modèle
        model_params = self.model.count_parameters() if hasattr(self.model, 'count_parameters') else None
        if model_params:
            print(f"\n📊 Paramètres du modèle:")
            print(f"  - Total: {model_params['total']:,}")
            print(f"  - Token embeddings: {model_params['token_embeddings']:,}")
            print(f"  - Position embeddings: {model_params['position_embeddings']:,}")
            if model_params['position_embeddings'] == 0:
                print(f"    ✨ RoPE activé (0 paramètres pour positions!)")
            print(f"  - Transformer blocks: {model_params['transformer_blocks']:,}")
        else:
            print(f"Paramètres du modèle: {sum(p.numel() for p in self.model.parameters()):,}")
        
        print("="*60 + "\n")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            # Entraînement
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss = self.validate()
            
            # Affichage
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            if val_loss is not None:
                print(f"  Val Loss:   {val_loss:.4f}")
            
            # Sauvegarder le checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch, train_loss)
            
            # Sauvegarder le meilleur modèle
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
                torch.save(self.model.state_dict(), best_model_path)
                print(f"  ✓ Meilleur modèle sauvegardé! (val_loss: {val_loss:.4f})")
        
        print("\n" + "="*60)
        print("ENTRAÎNEMENT TERMINÉ!")
        print("="*60)
        print(f"Train Loss finale: {self.train_losses[-1]:.4f}")
        if self.val_losses:
            print(f"Val Loss finale: {self.val_losses[-1]:.4f}")
            print(f"Meilleure Val Loss: {best_val_loss:.4f}")
        print("="*60 + "\n")


# ============================================
# EXEMPLE D'UTILISATION
# ============================================

def example_training():
    """Exemple complet d'entraînement avec RoPE"""
    print("\n🚀 EXEMPLE D'ENTRAÎNEMENT HessGPT AVEC RoPE\n")
    
    # 1. Importer le modèle et le tokenizer (vous devez adapter les imports)
    from Model.HessGpt import HessGPT
    from Tokenizer.Tokenizer import MYBPE
    
    # 2. Charger le tokenizer
    print("Chargement du tokenizer...")
    tokenizer = MYBPE(vocab_size=300)
    tokenizer.load_tokenizer("./Tokenizer/tokenizer_model.bin")
    print("✓ Tokenizer chargé")
    
    # 3. Charger les données
    print("\nChargement des données...")
    with open("./data/train.txt", "r", encoding="utf-8") as f:
        train_text = f.read()
    
    print(f"✓ {len(train_text)} caractères chargés")
    
    # 4. Créer les datasets
    print("\nCréation des datasets...")
    train_dataset = TextDataset(train_text, tokenizer, seq_len=128)
    print(f"✓ {len(train_dataset)} séquences d'entraînement")
    
    # 5. Créer le modèle AVEC RoPE
    print("\nCréation du modèle avec RoPE...")
    model = HessGPT(
        vocab_size=300,
        embed_dim=256,      # Plus petit pour tester (768 pour le vrai)
        num_heads=8,        # Plus petit pour tester (12 pour le vrai)
        num_layers=4,       # Plus petit pour tester (12 pour le vrai)
        max_seq_len=128,
        use_rope=True       # ✨ RoPE activé!
    )
    
    params = model.count_parameters()
    print(f"✓ Modèle créé ({params['total']:,} paramètres)")
    print(f"  - Position embeddings: {params['position_embeddings']:,} (RoPE = 0!)")
    
    # 6. Créer le trainer
    trainer = GPT2Trainer(
        model=model,
        train_dataset=train_dataset,
        learning_rate=3e-4,
        batch_size=4,
        num_epochs=5,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        gradient_accumulation_steps=2,  # Batch effectif = 4 * 2 = 8
        warmup_steps=100
    )
    
    # 7. Entraîner!
    trainer.train(save_every=1)
    
    print("\n✓ Entraînement terminé!")
    print("✓ Les checkpoints sont dans ./checkpoints/")


def example_training_20m():
    """Exemple d'entraînement avec configuration 20M paramètres"""
    print("\n🚀 EXEMPLE D'ENTRAÎNEMENT HessGPT 20M AVEC RoPE\n")
    
    from Model.HessGpt import HessGPT
    from Tokenizer.Tokenizer import MYBPE
    
    # 1. Charger le tokenizer (20k vocab)
    print("Chargement du tokenizer...")
    tokenizer = MYBPE(vocab_size=20000)
    tokenizer.load_tokenizer("./Tokenizer/tokenizer_model.bin")
    print("✓ Tokenizer chargé")
    
    # 2. Charger les données
    print("\nChargement des données...")
    with open("./data/train.txt", "r", encoding="utf-8") as f:
        train_text = f.read()
    
    print(f"✓ {len(train_text)} caractères chargés")
    
    # 3. Créer les datasets
    print("\nCréation des datasets...")
    train_dataset = TextDataset(train_text, tokenizer, seq_len=512)
    print(f"✓ {len(train_dataset)} séquences d'entraînement")
    
    # 4. Créer le modèle 20M avec RoPE
    print("\nCréation du modèle 20M avec RoPE...")
    model = HessGPT(
        vocab_size=20000,
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        max_seq_len=2048,
        use_rope=True  # ✨ RoPE!
    )
    
    params = model.count_parameters()
    print(f"\n✓ Modèle créé:")
    print(f"  - Total: {params['total']:,} paramètres")
    print(f"  - Token embeddings: {params['token_embeddings']:,}")
    print(f"  - Position embeddings: {params['position_embeddings']:,} ✨")
    print(f"  - Transformer blocks: {params['transformer_blocks']:,}")
    
    # 5. Créer le trainer
    trainer = GPT2Trainer(
        model=model,
        train_dataset=train_dataset,
        learning_rate=3e-4,
        batch_size=2,          # Plus petit car modèle plus gros
        num_epochs=10,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        gradient_accumulation_steps=4,  # Batch effectif = 2 * 4 = 8
        warmup_steps=500
    )
    
    # 6. Entraîner!
    trainer.train(save_every=2)
    
    print("\n✓ Entraînement 20M terminé!")


def test_dataset():
    """Test simple du dataset"""
    print("\n" + "="*60)
    print("TEST: Dataset")
    print("="*60)
    
    # Créer un mini texte de test
    test_text = "Bonjour! " * 100  # Répéter pour avoir assez de tokens
    
    # Mock tokenizer pour le test
    class MockTokenizer:
        def encoder(self, text):
            # Simuler l'encodage (1 token par caractère pour simplifier)
            return list(range(len(test_text)))
    
    tokenizer = MockTokenizer()
    
    # Créer le dataset
    dataset = TextDataset(test_text, tokenizer, seq_len=10)
    
    print(f"✓ Dataset créé: {len(dataset)} séquences")
    
    # Tester __getitem__
    input_ids, target_ids = dataset[0]
    
    print(f"✓ Input shape: {input_ids.shape}")
    print(f"✓ Target shape: {target_ids.shape}")
    print(f"✓ Input:  {input_ids.tolist()[:10]}")
    print(f"✓ Target: {target_ids.tolist()[:10]}")
    print(f"  (Target = Input décalé de 1)")


if __name__ == "__main__":
    print("\n🚀 TRAINING LOOP - HessGPT AVEC RoPE\n")
    
    # Test du dataset
    test_dataset()
    
    print("\n" + "="*60)
    print("Pour lancer l'entraînement complet:")
    print("="*60)
    print("1. Assurez-vous d'avoir un fichier data/train.txt")
    print("2. Décommentez example_training() ou example_training_20m() ci-dessous")
    print("3. Ajustez les paramètres (batch_size, epochs, etc.)")
    print("4. Lancez: python Training/training.py")
    print("\n💡 AVANTAGES DE RoPE:")
    print("  • Moins de paramètres à entraîner")
    print("  • Meilleure extrapolation aux longues séquences")
    print("  • Architecture moderne (LLaMA-style)")
    print("="*60)
    
    # Décommentez pour lancer l'entraînement:
    # example_training()          # Petit modèle de test
    # example_training_20m()      # Modèle 20M
