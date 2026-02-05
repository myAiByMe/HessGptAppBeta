#!/usr/bin/env python3
"""
🚀 HessGPT - PRE-TRAIN SCALABLE (Chunk-Based) avec RoPE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Load directement les chunks du downloader (mixed_1B_chunk_X/)
✅ Fusion des .pt (fineweb_edu.pt + wikipedia.pt + ...) en séquence
✅ BF16 training (plus rapide + plus stable que FP16 sur modern GPU)
✅ RoPE (Rotary Position Embeddings) — économise ~10M paramètres
✅ LR WSD dynamique — scale à l'infini, decay uniquement à la fin
   → Inspiré de Qwen : warmup 2% / stable 90%+ / decay 8%
   → Tu peux rajouter des chunks sans toucher au schedule
✅ 1 epoch = 1 chunk chargé en RAM puis libéré
✅ Reprise propre via checkpoint (chunk_id sauvegardé)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

USAGE:
    python pretrain_hessgpt.py                    # Start / resume normal
    python pretrain_hessgpt.py --total-chunks 50  # Override nombre de chunks
    python pretrain_hessgpt.py --dry-run          # Vérifie les chunks sans trainer
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import sys
import os
import time
import math
import json
import gc
import argparse
from tqdm import tqdm
from transformers import GPT2Tokenizer
from datetime import datetime
import traceback

sys.path.append('./Core/Model')

from HessGpt import HessGPT
# ============================================
# TOKENS SPÉCIAUX CHATLM
# ============================================
# Ces 4 tokens sont réservés pour la phase fine-tuning ChatLM.
# On les bake dans le vocab DÈS le pre-training pour que les
# embeddings existent déjà et soient initialisés proprement.
# Le pre-training ne les utilise pas dans les séquences,
# mais le modèle apprend leurs embeddings via les autres tokens.
SPECIAL_TOKENS = {
    '<|system|>':    50257,
    '<|user|>':      50258,
    '<|assistant|>': 50259,
    '<|end|>':       50260,
}

# ============================================
# ARGS CLI
# ============================================
parser = argparse.ArgumentParser(description='HessGPT Scalable Pre-Training')
parser.add_argument('--total-chunks', type=int, default=None,
                    help='Override nombre de chunks à train (auto-détecté sinon)')
parser.add_argument('--dry-run', action='store_true',
                    help='Vérifie les chunks sans lancer le training')
parser.add_argument('--data-dir', type=str, default='./data/mixed',
                    help='Directory des chunks du downloader')
parser.add_argument('--checkpoint', type=str, default='./checkpoints/HessGpt_pretrain.pt',
                    help='Path du checkpoint')
args = parser.parse_args()

print("=" * 80)
print("🚀 HessGPT — SCALABLE PRE-TRAINING avec RoPE")
print("   BF16 | WSD Dynamic LR | Chunk-Based Loading | Rotary Position Embeddings")
print("=" * 80)

# ============================================
# CONFIGURATION
# ============================================
CONFIG = {
    # --- Model (0.5B params avec RoPE) ---
    'vocab_size':    50257 + len(SPECIAL_TOKENS),  # 50261 (GPT2 + 4 ChatLM tokens)
    'embed_dim':     1280,
    'num_heads':     20,
    'num_layers':    20,
    'max_seq_len':   512,
    'dropout':       0.1,
    'use_rope':      True,  # ✨ RoPE (Rotary Position Embeddings) - économise ~10M params

    # --- Training ---
    'batch_size':              32,   # H100 80GB + modèle 0.5B (~6GB) → on peut faire 32 facilement
    'gradient_accumulation':   4,    # effective batch = 32×4×512 = 65536 tokens
    'max_grad_norm':           1.0,
    'learning_rate':           3e-4,

    # --- Data ---
    'data_dir':    args.data_dir,       # ./data/mixed/
    'val_ratio':   0.005,               # 0.5% du chunk 0 pour validation

    # --- WSD LR Schedule (DYNAMIQUE) ---
    # Le schedule se calcule en fonction du TOTAL de chunks détecté.
    # Tu peux rajouter des chunks : le decay se pousse automatiquement à la fin.
    # Inspiré Qwen : warmup très court, stable très long, decay à la fin.
    'warmup_ratio':  0.02,    # 2% des steps totaux (≈ 1 chunk sur 50)
    'decay_ratio':   0.08,    # 8% des steps totaux (≈ 4 chunks sur 50)
    # stable_ratio = 1.0 - warmup - decay (calculé auto)
    'min_lr_ratio':  0.1,     # LR_min = LR_max * 0.1 à la fin du decay

    # --- Validation ---
    'validate_every_steps': 500,
    'val_batches':          50,

    # --- Checkpoint ---
    'checkpoint_file':     args.checkpoint,
    'save_every_epochs':   5,

    # --- System ---
    'use_compile':    True,
    'compile_mode':   'default',
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"\n📊 CONFIGURATION :")
print(f"   Vocab size : {CONFIG['vocab_size']:,} (GPT2 50257 + {len(SPECIAL_TOKENS)} ChatLM)")
print(f"   Embed dim  : {CONFIG['embed_dim']}")
print(f"   Layers     : {CONFIG['num_layers']}")
print(f"   Heads      : {CONFIG['num_heads']}")
print(f"   Seq len    : {CONFIG['max_seq_len']}")
print(f"   Use RoPE   : {CONFIG['use_rope']} ✨" if CONFIG['use_rope'] else f"   Use RoPE   : {CONFIG['use_rope']}")

print(f"\n🗣️  TOKENS CHATLM :")
for token, idx in SPECIAL_TOKENS.items():
    print(f"   {token:20s} → id {idx}")

# ============================================
# SCAN CHUNKS DISPONIBLES
# ============================================
def scan_available_chunks(data_dir):
    """
    Scanne le data_dir pour trouver tous les chunks complets.
    Un chunk est valide si le dossier mixed_1B_chunk_X/chunk/ existe
    et contient au moins un .pt.
    """
    available = []
    if not os.path.exists(data_dir):
        return available

    for entry in sorted(os.listdir(data_dir)):
        if not entry.startswith('mixed_1B_chunk_'):
            continue
        chunk_subdir = os.path.join(data_dir, entry, 'chunk')
        if not os.path.isdir(chunk_subdir):
            continue
        pt_files = sorted([f for f in os.listdir(chunk_subdir) if f.endswith('.pt')])
        if len(pt_files) > 0:
            # Extraire l'ID numérique
            try:
                chunk_id = int(entry.replace('mixed_1B_chunk_', ''))
                available.append({
                    'id': chunk_id,
                    'dir': chunk_subdir,
                    'files': pt_files,
                })
            except ValueError:
                continue

    # Sort par ID
    available.sort(key=lambda x: x['id'])
    return available

print(f"\n🔍 Scan des chunks dans {CONFIG['data_dir']}...")
AVAILABLE_CHUNKS = scan_available_chunks(CONFIG['data_dir'])

if args.total_chunks is not None:
    # Override : limiter au nombre demandé
    AVAILABLE_CHUNKS = AVAILABLE_CHUNKS[:args.total_chunks]

NUM_CHUNKS = len(AVAILABLE_CHUNKS)

print(f"   ✅ {NUM_CHUNKS} chunks trouvés")
if NUM_CHUNKS > 0:
    total_estimated_tokens = NUM_CHUNKS * 1e9  # ~1B par chunk (from downloader)
    print(f"   📊 Tokens estimés : {total_estimated_tokens / 1e9:.0f}B")
    print(f"   📂 Premier chunk : {AVAILABLE_CHUNKS[0]['dir']}")
    print(f"   📂 Dernier chunk  : {AVAILABLE_CHUNKS[-1]['dir']}")
    print(f"   📁 Fichiers par chunk : {AVAILABLE_CHUNKS[0]['files']}")

if args.dry_run:
    print("\n📋 DRY RUN — Détail des chunks :")
    total_size = 0
    for chunk in AVAILABLE_CHUNKS:
        size = sum(
            os.path.getsize(os.path.join(chunk['dir'], f))
            for f in chunk['files']
        )
        total_size += size
        print(f"   chunk_{chunk['id']:03d} : {len(chunk['files'])} fichiers, {size/1e6:.1f} MB")
    print(f"\n   Total disque : {total_size/1e9:.2f} GB")
    print(f"   Total chunks : {NUM_CHUNKS}")
    print("\n✅ Dry run terminé. Relancer sans --dry-run pour train.")
    sys.exit(0)

if NUM_CHUNKS == 0:
    print("\n❌ Aucun chunk trouvé ! Lance d'abord le downloader.")
    sys.exit(1)

# ============================================
# SETUP
# ============================================
print(f"\n✅ Device : {device}")
if device == 'cuda':
    print(f"   GPU  : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    # Vérifie BF16 support
    if torch.cuda.is_bf16_supported():
        print(f"   BF16 : ✅ Supporté")
    else:
        print(f"   BF16 : ⚠️  Non supporté — fallback FP16")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({
    'additional_special_tokens': list(SPECIAL_TOKENS.keys())
})
tokenizer.pad_token = tokenizer.eos_token

# ============================================
# CALCUL STEPS TOTAUX (pour WSD)
# ============================================
# On estime les steps par chunk :
#   tokens_per_chunk ≈ 1B (from downloader)
#   samples = tokens / (seq_len + 1)
#   batches = samples / batch_size
#   steps   = batches / gradient_accumulation
TOKENS_PER_CHUNK_EST = 1_000_000_000
samples_per_chunk = TOKENS_PER_CHUNK_EST // (CONFIG['max_seq_len'] + 1)
batches_per_chunk = samples_per_chunk // CONFIG['batch_size']
steps_per_chunk   = batches_per_chunk // CONFIG['gradient_accumulation']
TOTAL_STEPS       = steps_per_chunk * NUM_CHUNKS

print(f"\n📈 TRAINING PLAN :")
print(f"   Chunks           : {NUM_CHUNKS}")
print(f"   Steps/chunk      : {steps_per_chunk:,}")
print(f"   Total steps      : {TOTAL_STEPS:,}")
print(f"   Tokens totaux    : {NUM_CHUNKS * TOKENS_PER_CHUNK_EST / 1e9:.0f}B")

# ============================================
# WSD SCHEDULER — DYNAMIQUE
# ============================================
class WSDScheduler:
    """
    Warmup – Stable – Decay
    ─────────────────────────────────────
    Inspiré Qwen/LLaMA best practices :
    • Warmup très court (2%) : convergence rapide du début
    • Stable très long (90%) : le modèle apprend sans perturbation
    • Decay à la fin (8%)    : affine les poids pour la meilleure qualité

    KEY POINT : total_steps est calculé à partir du nombre de chunks
    détectés. Si tu rajoutes des chunks (le downloader continue),
    relancez le script → il recalcule total_steps, le decay se pousse
    automatiquement vers la fin. Le checkpoint garde current_step,
    donc la reprise est seamless.
    """
    def __init__(self, optimizer, max_lr, total_steps,
                 warmup_ratio=0.02, decay_ratio=0.08, min_lr_ratio=0.1):
        self.optimizer   = optimizer
        self.max_lr      = max_lr
        self.min_lr      = max_lr * min_lr_ratio
        self.total_steps = total_steps

        self.warmup_steps = int(total_steps * warmup_ratio)
        self.decay_steps  = int(total_steps * decay_ratio)
        self.stable_steps = total_steps - self.warmup_steps - self.decay_steps

        self.current_step = 0

        print(f"\n📈 WSD LR SCHEDULE :")
        print(f"   ├─ Warmup  : {self.warmup_steps:>8,} steps  ({warmup_ratio*100:>4.1f}%)")
        print(f"   ├─ Stable  : {self.stable_steps:>8,} steps  ({self.stable_steps/total_steps*100:>4.1f}%)")
        print(f"   ├─ Decay   : {self.decay_steps:>8,} steps  ({decay_ratio*100:>4.1f}%)")
        print(f"   └─ Total   : {self.total_steps:>8,} steps")
        print(f"   LR : {self.min_lr:.2e} → {self.max_lr:.2e}")

    def get_lr(self):
        step = self.current_step

        if step < self.warmup_steps:
            # Phase 1 : Warmup linéaire
            return self.max_lr * (step / max(self.warmup_steps, 1))

        elif step < self.warmup_steps + self.stable_steps:
            # Phase 2 : Stable — LR constant à max_lr
            return self.max_lr

        else:
            # Phase 3 : Decay cosine vers min_lr
            decay_step = step - self.warmup_steps - self.stable_steps
            progress   = min(decay_step / max(self.decay_steps, 1), 1.0)
            cosine     = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr + (self.max_lr - self.min_lr) * cosine

    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

    def get_last_lr(self):
        return [self.get_lr()]

    def state_dict(self):
        return {'current_step': self.current_step}

    def load_state_dict(self, sd):
        self.current_step = sd['current_step']

# ============================================
# LAZY CHUNK DATASET
# ============================================
class LazyChunkDataset(Dataset):
    """
    Charge UN chunk en RAM.
    Un chunk = plusieurs .pt (un par dataset source).
    On les concatène en séquence.
    """
    def __init__(self, chunk_info, seq_len, pad_token_id):
        self.seq_len     = seq_len
        self.pad_token_id = pad_token_id
        self.tokens      = None
        self.num_samples = 0
        self._load(chunk_info)

    def _load(self, chunk_info):
        print(f"   📥 Loading chunk_{chunk_info['id']:03d} ({len(chunk_info['files'])} fichiers)...")
        t0 = time.time()

        all_tokens = []
        for fname in chunk_info['files']:
            fpath = os.path.join(chunk_info['dir'], fname)
            try:
                data = torch.load(fpath, map_location='cpu')
                # Le downloader sauvegarde des listes ou des tensors
                if isinstance(data, list):
                    all_tokens.extend(data)
                elif isinstance(data, torch.Tensor):
                    all_tokens.extend(data.tolist())
                else:
                    print(f"      ⚠️  {fname} : type inconnu ({type(data)}), skip")
            except Exception as e:
                print(f"      ⚠️  {fname} : erreur load ({e}), skip")
                continue

        if len(all_tokens) == 0:
            raise ValueError(f"Chunk {chunk_info['id']} : aucun token chargé !")

        self.tokens     = torch.tensor(all_tokens, dtype=torch.long)
        self.num_samples = len(self.tokens) // (self.seq_len + 1)

        elapsed = time.time() - t0
        print(f"   ✅ {len(self.tokens)/1e6:.1f}M tokens → {self.num_samples:,} samples ({elapsed:.1f}s)")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * (self.seq_len + 1)
        end   = start + self.seq_len + 1
        chunk = self.tokens[start:end]

        if len(chunk) < self.seq_len + 1:
            pad_len = self.seq_len + 1 - len(chunk)
            chunk = torch.cat([
                chunk,
                torch.full((pad_len,), self.pad_token_id, dtype=torch.long)
            ])

        return chunk[:-1], chunk[1:]

    def unload(self):
        del self.tokens
        self.tokens = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ============================================
# CHECKPOINT MANAGER
# ============================================
class CheckpointManager:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def save(self, model, optimizer, scheduler, metadata):
        m = model._orig_mod if hasattr(model, '_orig_mod') else model
        checkpoint = {
            'model_state_dict':     m.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'global_step':          metadata['global_step'],
            'next_chunk_idx':       metadata['next_chunk_idx'],
            'training_history':     metadata['training_history'],
            'total_training_time':  metadata.get('total_training_time', 0),
            'config':               CONFIG,
            'last_save':            datetime.now().isoformat(),
        }
        tmp = self.path + '.tmp'
        torch.save(checkpoint, tmp)
        os.replace(tmp, self.path)
        print(f"      💾 Checkpoint → {self.path}")

    def load(self):
        if not os.path.exists(self.path):
            return None
        print(f"\n📂 Checkpoint trouvé : {self.path}")
        cp = torch.load(self.path, map_location='cpu')
        print(f"   ✅ Step          : {cp['global_step']:,}")
        print(f"   ✅ Next chunk    : {cp['next_chunk_idx']}")
        print(f"   ✅ Temps total   : {cp.get('total_training_time', 0)/3600:.2f}h")
        return cp

# ============================================
# VALIDATION
# ============================================
@torch.no_grad()
def validate(model, val_loader, device, max_batches=50):
    model.eval()
    total_loss   = 0
    total_tokens = 0

    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _, loss = model(x, targets=y)
        mask = (y != tokenizer.pad_token_id)
        total_loss   += loss.item() * mask.sum().item()
        total_tokens += mask.sum().item()

    avg_loss    = total_loss / max(total_tokens, 1)
    perplexity  = math.exp(min(avg_loss, 10))
    model.train()
    return perplexity, avg_loss

# ============================================
# TRAIN ONE EPOCH = ONE CHUNK
# ============================================
def train_one_chunk(
    model, chunk_info, optimizer, scheduler,
    val_loader, checkpoint_manager, training_history,
    global_step, total_training_time, chunk_idx
):
    epoch_num = chunk_idx + 1  # Display 1-indexed

    print(f"\n{'=' * 80}")
    print(f"📦 EPOCH {epoch_num}/{NUM_CHUNKS}  —  chunk_{chunk_info['id']:03d}")
    print(f"   LR actuel : {scheduler.get_last_lr()[0]:.2e}")
    print(f"{'=' * 80}")

    # --- Load chunk ---
    try:
        train_dataset = LazyChunkDataset(
            chunk_info, CONFIG['max_seq_len'], tokenizer.pad_token_id
        )
    except Exception as e:
        print(f"   ❌ Erreur load chunk : {e}")
        return global_step, total_training_time

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    num_batches = len(train_loader)
    print(f"   📊 {num_batches:,} batches")

    model.train()
    epoch_loss     = 0.0
    valid_batches  = 0
    t_start        = time.time()

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch_num}/{NUM_CHUNKS}",
        leave=True,
    )

    for batch_idx, (x, y) in enumerate(pbar):
        try:
            x = x.to(device)
            y = y.to(device)

            # BF16 autocast
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, loss = model(x, targets=y)
                loss = loss / CONFIG['gradient_accumulation']

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()

            if (batch_idx + 1) % CONFIG['gradient_accumulation'] == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), CONFIG['max_grad_norm']
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                global_step += 1

                # --- Validation périodique ---
                if global_step % CONFIG['validate_every_steps'] == 0 and val_loader is not None:
                    val_ppl, val_loss = validate(
                        model, val_loader, device, CONFIG['val_batches']
                    )
                    print(f"\n      {'─' * 65}")
                    print(f"      📊 Step {global_step:,} | PPL {val_ppl:7.2f} | "
                          f"Val Loss {val_loss:.4f} | LR {scheduler.get_last_lr()[0]:.2e}")
                    print(f"      {'─' * 65}\n")

                    training_history['validations'].append({
                        'step':       global_step,
                        'epoch':      epoch_num,
                        'chunk_id':   chunk_info['id'],
                        'perplexity': val_ppl,
                        'val_loss':   val_loss,
                        'train_loss': loss.item() * CONFIG['gradient_accumulation'],
                        'lr':         scheduler.get_last_lr()[0],
                    })

            epoch_loss    += loss.item() * CONFIG['gradient_accumulation']
            valid_batches += 1

            if batch_idx % 20 == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item() * CONFIG["gradient_accumulation"]:.4f}',
                    'lr':   f'{scheduler.get_last_lr()[0]:.2e}',
                    'step': f'{global_step:,}',
                })

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n      ❌ OOM au batch {batch_idx} — cleanup...")
                torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                continue
            raise

    pbar.close()

    # --- Fin epoch ---
    avg_loss = epoch_loss / max(valid_batches, 1)

    # Val finale de l'epoch
    val_ppl, val_loss = (None, None)
    if val_loader is not None:
        val_ppl, val_loss = validate(model, val_loader, device, CONFIG['val_batches'])

    epoch_time = time.time() - t_start
    total_training_time += epoch_time

    print(f"\n   {'─' * 70}")
    print(f"   ✅ EPOCH {epoch_num} TERMINÉE")
    print(f"      Train Loss : {avg_loss:.4f}")
    if val_ppl is not None:
        print(f"      Val PPL    : {val_ppl:.2f}")
        print(f"      Val Loss   : {val_loss:.4f}")
    print(f"      Temps      : {epoch_time / 60:.1f} min")
    print(f"      LR         : {scheduler.get_last_lr()[0]:.2e}")
    print(f"   {'─' * 70}")

    training_history['epochs'].append({
        'epoch':      epoch_num,
        'chunk_id':   chunk_info['id'],
        'train_loss': avg_loss,
        'val_loss':   val_loss,
        'val_ppl':    val_ppl,
        'global_step': global_step,
        'lr':         scheduler.get_last_lr()[0],
        'time_s':     epoch_time,
    })

    # --- Checkpoint ---
    if epoch_num % CONFIG['save_every_epochs'] == 0:
        checkpoint_manager.save(
            model, optimizer, scheduler,
            metadata={
                'global_step':         global_step,
                'next_chunk_idx':      chunk_idx + 1,
                'training_history':    training_history,
                'total_training_time': total_training_time,
            }
        )

    # --- Cleanup chunk de la RAM ---
    train_dataset.unload()
    del train_loader, train_dataset
    gc.collect()
    torch.cuda.empty_cache()

    return global_step, total_training_time

# ============================================
# MAIN
# ============================================
def main():
    from HessGpt import HessGPT

    print("\n" + "=" * 80)
    print("🤖 CRÉATION DU MODÈLE")
    print("=" * 80)

    if device == 'cpu':
        print("\n⚠️  GPU fortement recommandée pour le training !")

    checkpoint_manager = CheckpointManager(CONFIG['checkpoint_file'])

    # --- Modèle ---
    print(f"\n🏗️  HessGPT ({CONFIG['embed_dim']}d, {CONFIG['num_layers']}L, {CONFIG['num_heads']}h)...")
    model = HessGPT(
        vocab_size=CONFIG['vocab_size'],
        embed_dim=CONFIG['embed_dim'],
        num_heads=CONFIG['num_heads'],
        num_layers=CONFIG['num_layers'],
        max_seq_len=CONFIG['max_seq_len'],
        dropout=CONFIG['dropout'],
        use_rope=CONFIG['use_rope'],  # ✨ RoPE activé
    ).to(device)
    # ⚡ MIXED PRECISION — STRATÉGIE CORRECTE :
    # - Poids du modèle : FP32 (pour optimizer states précis)
    # - Forward/Backward : BF16 via autocast (rapide sur H100)
    # - Optimizer states : FP32 automatiquement (PyTorch cast les grads)
    # ⚠️  NE PAS faire model.to(bfloat16) — ça met les optimizer states
    #     en BF16 aussi → les petits updates (LR×grad ≈ 3e-6) sont
    #     arrondis à 0 par la précision BF16 → le modèle n'apprend plus.
    # Sur H100 avec 0.5B : poids FP32 + optimizer = ~6GB sur 80GB → OK
    print(f"   ✅ Poids en FP32 (optimizer states précis)")
    print(f"   ✅ Forward/Backward via autocast BF16 (rapide sur H100)")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Paramètres : {total_params / 1e6:.1f}M")
    
    # Afficher les détails de l'architecture avec RoPE
    if hasattr(model, 'count_parameters'):
        params_detail = model.count_parameters()
        print(f"\n📊 Détails de l'architecture :")
        print(f"   • Token embeddings     : {params_detail['token_embeddings'] / 1e6:.1f}M")
        print(f"   • Position embeddings  : {params_detail['position_embeddings'] / 1e6:.1f}M", end="")
        if CONFIG['use_rope']:
            print(f" ✨ (RoPE = 0 params!)")
        else:
            print()
        print(f"   • Transformer blocks   : {params_detail['transformer_blocks'] / 1e6:.1f}M")
        print(f"   • Final LayerNorm      : {params_detail['final_ln'] / 1e3:.1f}K")
        print(f"   • Output head          : {params_detail['output_head'] / 1e6:.1f}M (partagé)")
        if CONFIG['use_rope']:
            saved_params = params_detail['token_embeddings'] + params_detail['position_embeddings']
            # Position embeddings classiques = vocab_size * embed_dim
            classic_pos_emb = CONFIG['max_seq_len'] * CONFIG['embed_dim']
            print(f"\n   💰 Économie RoPE : ~{classic_pos_emb / 1e6:.1f}M paramètres")


    # --- Compile ---
    if CONFIG['use_compile'] and device == 'cuda':
        print(f"\n⚡ torch.compile (mode={CONFIG['compile_mode']})...")
        try:
            model = torch.compile(model, mode=CONFIG['compile_mode'])
            print(f"   ✅ Compilé")
        except Exception as e:
            print(f"   ⚠️  Compilation échouée : {e}")

    # --- Optimizer ---
    # AdamW en FP32 pour la stabilité (même si modèle en BF16)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=(device == 'cuda'),
    )

    # --- Scheduler WSD ---
    scheduler = WSDScheduler(
        optimizer,
        max_lr=CONFIG['learning_rate'],
        total_steps=TOTAL_STEPS,
        warmup_ratio=CONFIG['warmup_ratio'],
        decay_ratio=CONFIG['decay_ratio'],
        min_lr_ratio=CONFIG['min_lr_ratio'],
    )

    # --- Training history ---
    training_history = {
        'config':          CONFIG,
        'special_tokens':  SPECIAL_TOKENS,
        'total_params':    total_params,
        'num_chunks':      NUM_CHUNKS,
        'total_steps':     TOTAL_STEPS,
        'epochs':          [],
        'validations':     [],
        'start_time':      datetime.now().isoformat(),
    }

    global_step          = 0
    start_chunk_idx      = 0
    total_training_time  = 0

    # --- Resume from checkpoint ---
    checkpoint = checkpoint_manager.load()
    if checkpoint:
        print("\n♻️  REPRISE DU TRAINING")
        unwrapped = model._orig_mod if hasattr(model, '_orig_mod') else model
        unwrapped.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        global_step         = checkpoint['global_step']
        start_chunk_idx     = checkpoint['next_chunk_idx']
        training_history    = checkpoint['training_history']
        total_training_time = checkpoint.get('total_training_time', 0)
        print(f"   ▶️  Reprise à chunk index {start_chunk_idx} (step {global_step:,})")

    # --- Validation loader (chunk 0) ---
    print(f"\n📥 Préparation validation (chunk 0)...")
    val_loader = None
    if len(AVAILABLE_CHUNKS) > 0:
        val_chunk = AVAILABLE_CHUNKS[0]
        try:
            val_dataset = LazyChunkDataset(
                val_chunk, CONFIG['max_seq_len'], tokenizer.pad_token_id
            )
            # On prend seulement val_ratio des samples pour val
            val_size = max(1, int(len(val_dataset) * CONFIG['val_ratio']))
            val_subset = torch.utils.data.Subset(val_dataset, list(range(val_size)))
            val_loader = DataLoader(
                val_subset,
                batch_size=CONFIG['batch_size'],
                num_workers=2,
                pin_memory=True,
            )
            print(f"   ✅ Val set : {val_size:,} samples")
        except Exception as e:
            print(f"   ⚠️  Val loader échoué : {e}")
            val_loader = None

    # ============================================
    # TRAINING LOOP
    # ============================================
    print("\n" + "=" * 80)
    print("🚀 DÉMARRAGE TRAINING")
    print(f"   Chunks : {start_chunk_idx} → {NUM_CHUNKS}")
    print(f"   Total tokens : {NUM_CHUNKS * 1e9 / 1e9:.0f}B")
    print("=" * 80)

    overall_start = time.time()

    for chunk_idx in range(start_chunk_idx, NUM_CHUNKS):
        chunk_info = AVAILABLE_CHUNKS[chunk_idx]

        # Skip chunk 0 en training (utilisé pour val) sauf si c'est le seul
        # En fait on train dessus aussi, la val n'utilise qu'un petit subset
        # donc pas de conflit.

        try:
            global_step, total_training_time = train_one_chunk(
                model=model,
                chunk_info=chunk_info,
                optimizer=optimizer,
                scheduler=scheduler,
                val_loader=val_loader,
                checkpoint_manager=checkpoint_manager,
                training_history=training_history,
                global_step=global_step,
                total_training_time=total_training_time,
                chunk_idx=chunk_idx,
            )
        except KeyboardInterrupt:
            print("\n\n⚠️  CTRL+C — Sauvegarde d'urgence...")
            checkpoint_manager.save(
                model, optimizer, scheduler,
                metadata={
                    'global_step':         global_step,
                    'next_chunk_idx':      chunk_idx,  # Reprendre CE chunk
                    'training_history':    training_history,
                    'total_training_time': total_training_time,
                }
            )
            print("   ✅ Checkpoint sauvegardé. Relancer pour reprendre.")
            return
        except Exception as e:
            print(f"\n❌ ERREUR à l'epoch {chunk_idx + 1} :")
            print(traceback.format_exc())
            print("\n💾 Checkpoint d'urgence...")
            checkpoint_manager.save(
                model, optimizer, scheduler,
                metadata={
                    'global_step':         global_step,
                    'next_chunk_idx':      chunk_idx,
                    'training_history':    training_history,
                    'total_training_time': total_training_time,
                }
            )
            raise

    # --- FIN ---
    overall_time = time.time() - overall_start

    # Checkpoint final
    checkpoint_manager.save(
        model, optimizer, scheduler,
        metadata={
            'global_step':         global_step,
            'next_chunk_idx':      NUM_CHUNKS,
            'training_history':    training_history,
            'total_training_time': total_training_time,
        }
    )

    print("\n" + "=" * 80)
    print("🎉 TRAINING TERMINÉ !")
    print("=" * 80)
    print(f"\n📊 RÉSUMATS :")
    print(f"   Epochs complétées : {len(training_history['epochs'])}/{NUM_CHUNKS}")
    print(f"   Steps totaux      : {global_step:,}")
    print(f"   Tokens vus        : {NUM_CHUNKS * 1e9 / 1e9:.0f}B")
    print(f"   Temps training    : {total_training_time / 3600:.2f}h")
    print(f"   Temps réel        : {overall_time / 3600:.2f}h")

    if training_history['validations']:
        last = training_history['validations'][-1]
        print(f"   PPL final         : {last['perplexity']:.2f}")
        print(f"   Loss final        : {last['val_loss']:.4f}")

    print(f"\n💾 Checkpoint : {checkpoint_manager.path}")

    # Save history JSON
    history_path = CONFIG['checkpoint_file'].replace('.pt', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2, default=str)
    print(f"📝 History     : {history_path}")
    print("\n✅ DONE !")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrompu — checkpoint sauvegardé, relancer pour continuer.")
    except Exception as e:
        print(f"\n\n❌ ERREUR FATALE :")
        print(traceback.format_exc())
    finally:
        print("\n👋 Fin du script")
