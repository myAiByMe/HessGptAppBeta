#!/usr/bin/env python3
"""
Création du tokenizer.json pour HessGPT Mobile
GPT-2 tokenizer + 4 tokens ChatLM
"""

import json
import argparse
from transformers import GPT2Tokenizer

# Tokens spéciaux ChatLM (EXACTEMENT comme dans pretrain)
SPECIAL_TOKENS = {
    '<|system|>':    50257,
    '<|user|>':      50258,
    '<|assistant|>': 50259,
    '<|end|>':       50260,
}

def create_hessgpt_tokenizer(output_path):
    """
    Crée le tokenizer.json pour HessGPT Mobile
    """
    
    print("=" * 80)
    print("📝 CRÉATION TOKENIZER HESSGPT")
    print("=" * 80)
    
    # 1. Charger GPT-2 tokenizer
    print("\n📥 Téléchargement tokenizer GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # 2. Ajouter les tokens spéciaux
    print(f"\n✨ Ajout des tokens ChatLM...")
    tokenizer.add_special_tokens({
        'additional_special_tokens': list(SPECIAL_TOKENS.keys())
    })
    
    # 3. Construire le vocabulaire complet
    print(f"\n🏗️  Construction du vocabulaire...")
    vocab = tokenizer.get_vocab()
    
    print(f"   • Vocab GPT-2        : {50257:,} tokens")
    print(f"   • Tokens ChatLM      : {len(SPECIAL_TOKENS)} tokens")
    print(f"   • Total              : {len(vocab):,} tokens")
    
    # 4. Vérifier que les tokens spéciaux ont les bons IDs
    print(f"\n🔍 Vérification des IDs :")
    for token, expected_id in SPECIAL_TOKENS.items():
        actual_id = vocab.get(token)
        status = "✅" if actual_id == expected_id else "❌"
        print(f"   {status} {token:20s} → {actual_id} (attendu: {expected_id})")
    
    # 5. Récupérer les merges BPE
    print(f"\n📋 Extraction des merges BPE...")
    merges = []
    if hasattr(tokenizer, 'bpe_ranks'):
        merges = [f"{a} {b}" for (a, b) in tokenizer.bpe_ranks.keys()]
    elif hasattr(tokenizer, 'encoder') and hasattr(tokenizer, 'bpe_ranks'):
        # Alternative pour certaines versions de transformers
        bpe_file = tokenizer.vocab_files_names.get('merges_file', 'merges.txt')
        # Les merges sont stockés dans le modèle
        merges = []
    
    print(f"   ✅ {len(merges):,} merges BPE")
    
    # 6. Construire la structure JSON
    tokenizer_data = {
        "vocab": vocab,
        "merges": merges,
        "model_max_length": 512,  # Comme dans pretrain
        "added_tokens": [
            {"content": token, "id": token_id}
            for token, token_id in SPECIAL_TOKENS.items()
        ],
        "special_tokens": {
            "bos_token": tokenizer.bos_token,
            "eos_token": tokenizer.eos_token,
            "unk_token": tokenizer.unk_token,
            "pad_token": tokenizer.pad_token if tokenizer.pad_token else tokenizer.eos_token,
        },
        "model_type": "HessGPT",
        "version": "1.0",
    }
    
    # 7. Sauvegarder
    print(f"\n💾 Sauvegarde : {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
    
    # 8. Test
    print(f"\n🧪 Test du tokenizer...")
    test_texts = [
        "Bonjour, comment ça va ?",
        "<|user|>Hello<|end|>",
        "<|assistant|>Salut !<|end|>",
    ]
    
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"\n   📝 '{text}'")
        print(f"   → IDs : {encoded[:10]}{'...' if len(encoded) > 10 else ''}")
        print(f"   → Décodé : '{decoded}'")
    
    print("\n" + "=" * 80)
    print("✅ TOKENIZER CRÉÉ !")
    print("=" * 80)
    print(f"\n📦 Fichier  : {output_path}")
    print(f"📊 Tokens   : {len(vocab):,}")
    print(f"🎯 Prêt pour Android !")
    
    print(f"\n📱 Prochaine étape :")
    print(f"   cp {output_path} HessGptMobileApp/app/src/main/assets/")


def main():
    parser = argparse.ArgumentParser(
        description="Créer tokenizer.json pour HessGPT Mobile"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='tokenizer.json',
        help='Fichier de sortie'
    )
    
    args = parser.parse_args()
    
    create_hessgpt_tokenizer(args.output)


if __name__ == "__main__":
    main()