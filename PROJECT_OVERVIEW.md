# 📱 HessGPT Mobile - Vue d'ensemble du projet

## 🎯 Objectif

Application mobile Android native permettant d'exécuter votre modèle de langage HessGPT (0.5B paramètres) **100% offline** sur smartphone, sans connexion internet requise.

## 🏗️ Architecture Technique

### Stack Technologique

```
┌─────────────────────────────────────────┐
│          Interface Utilisateur          │
│   ┌─────────────────────────────────┐  │
│   │  ChatActivity (Kotlin)          │  │
│   │  - RecyclerView pour messages   │  │
│   │  - Material Design 3            │  │
│   │  - Streaming en temps réel      │  │
│   └─────────────────────────────────┘  │
├─────────────────────────────────────────┤
│        Couche de Gestion ML             │
│   ┌─────────────────────────────────┐  │
│   │  ModelManager (Kotlin)          │  │
│   │  - Chargement du modèle         │  │
│   │  - Inférence PyTorch Mobile     │  │
│   │  - Génération autoregressive    │  │
│   │  - Top-K/Top-P sampling         │  │
│   └─────────────────────────────────┘  │
│   ┌─────────────────────────────────┐  │
│   │  Tokenizer (Kotlin)             │  │
│   │  - Encode texte → IDs           │  │
│   │  - Decode IDs → texte           │  │
│   │  - Support BPE                  │  │
│   └─────────────────────────────────┘  │
├─────────────────────────────────────────┤
│         PyTorch Mobile Runtime          │
│   ┌─────────────────────────────────┐  │
│   │  TorchScript (.ptl)             │  │
│   │  - Modèle HessGPT optimisé      │  │
│   │  - Quantification INT8          │  │
│   │  - Architecture GPT + RoPE      │  │
│   └─────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### Composants Principaux

#### 1. **ModelManager.kt**
- Charge le modèle TorchScript depuis les assets
- Gère le cycle de vie du modèle (load/release)
- Implémente la génération token par token
- Applique les stratégies de sampling (temperature, top-k, top-p)

#### 2. **Tokenizer.kt**
- Encode le texte en IDs de tokens
- Décode les IDs en texte lisible
- Supporte BPE (Byte Pair Encoding)
- Gère les tokens spéciaux (EOS, BOS, PAD)

#### 3. **ChatActivity.kt**
- Interface utilisateur du chat
- Streaming des tokens en temps réel
- Gestion de l'historique des messages
- Affichage des performances

#### 4. **PerformanceMonitor.kt**
- Mesure la latence d'inférence
- Calcule le débit (tokens/seconde)
- Monitore le temps jusqu'au premier token

## 📊 Flux de Données

```
User Input
    ↓
[Tokenizer] → IDs: [123, 456, 789]
    ↓
[ModelManager] → Forward pass
    ↓
[PyTorch Mobile] → Logits: [50257 dimensions]
    ↓
[Sampling] → Next Token ID: 234
    ↓
[Tokenizer] → Decoded Token: "bonjour"
    ↓
UI Update (streaming)
```

## 🔧 Pipeline d'Export

### 1. Entraînement (votre côté)
```python
# Dans HessGpt_RoPE/
python PreTrain.py --config config.yaml
# → Génère: checkpoint.pt
```

### 2. Export vers TorchScript
```python
# Dans HessGptMobileApp/python/
python export_model.py --checkpoint checkpoint.pt
# → Génère: model.ptl
```

### 3. Quantification (optionnel)
```python
python quantize_model.py --input model.ptl
# → Génère: model_quantized.ptl (réduction ~75%)
```

### 4. Déploiement Android
```bash
# Copier dans assets/
cp model.ptl app/src/main/assets/
cp tokenizer.json app/src/main/assets/

# Build Android
./gradlew assembleDebug
```

## 💾 Gestion de la Mémoire

### Occupation Mémoire Estimée

Pour un modèle de **0.5B paramètres**:

| Composant | FP32 | INT8 (quantifié) |
|-----------|------|------------------|
| Modèle | ~2 GB | ~500 MB |
| KV-Cache (512 tokens) | ~100 MB | ~25 MB |
| Activations | ~50 MB | ~50 MB |
| **Total** | **~2.15 GB** | **~575 MB** |

### Optimisations Mémoire

1. **Quantification INT8**: Réduit les poids de FP32 → INT8
2. **Limitation de séquence**: max_seq_len = 512 au lieu de 2048
3. **Batch size = 1**: Pas de batching, un exemple à la fois
4. **Pas de gradients**: `requires_grad = False` partout

## ⚡ Performances Attendues

### Sur Snapdragon 720G (mid-range 2020)

| Métrique | Valeur |
|----------|--------|
| Latence 1er token | 800-1200 ms |
| Débit | 3-5 tokens/sec |
| Mémoire utilisée | ~1.5 GB RAM |
| Taille APK | ~550 MB |

### Sur Snapdragon 8 Gen 2 (flagship 2023)

| Métrique | Valeur |
|----------|--------|
| Latence 1er token | 200-400 ms |
| Débit | 12-18 tokens/sec |
| Mémoire utilisée | ~1.2 GB RAM |
| Taille APK | ~550 MB |

## 🔐 Sécurité et Confidentialité

### Avantages du On-Device ML

✅ **Confidentialité totale**: Aucune donnée n'est envoyée à un serveur
✅ **Pas de connexion requise**: Fonctionne en mode avion
✅ **Latence réduite**: Pas de roundtrip réseau
✅ **Gratuit**: Pas de coûts d'API

### Considérations

⚠️ **Taille de l'app**: ~500MB, nécessite stockage suffisant
⚠️ **Batterie**: L'inférence consomme de l'énergie
⚠️ **Performances variables**: Dépend du hardware de l'appareil

## 📈 Évolutions Futures

### Version 1.1 (Court terme)
- [ ] KV-Cache persistant pour accélérer la génération
- [ ] Support du mode sombre
- [ ] Sauvegarde des conversations (Room Database)
- [ ] Partage des messages

### Version 1.2 (Moyen terme)
- [ ] Fine-tuning on-device avec LoRA
- [ ] Support de plusieurs modèles
- [ ] Synthèse vocale des réponses
- [ ] Reconnaissance vocale pour l'input

### Version 2.0 (Long terme)
- [ ] Modèles multimodaux (texte + images)
- [ ] Génération d'images on-device
- [ ] Support iOS avec Core ML
- [ ] Apprentissage fédéré

## 🎓 Apprentissages Clés

### PyTorch Mobile
- Conversion TorchScript: `torch.jit.trace()` vs `torch.jit.script()`
- Optimisation: `optimize_for_inference()` critique
- Quantification: Dynamic quantization pour LLMs

### Android
- Assets: Copie au runtime dans le cache pour accès PyTorch
- Threads: Inférence sur thread séparé (Coroutines)
- Mémoire: Libération explicite avec `module.destroy()`

### Performance
- Top-K/Top-P sampling: Équilibre diversité/qualité
- Streaming: Callback pour affichage progressif
- Monitoring: Mesure précise avec System.currentTimeMillis()

## 📚 Ressources

### Documentation
- [PyTorch Mobile Docs](https://pytorch.org/mobile/home/)
- [Android ML Kit](https://developers.google.com/ml-kit)
- [TorchScript Guide](https://pytorch.org/docs/stable/jit.html)

### Papiers de Recherche
- **RoPE**: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- **Quantization**: "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"
- **On-Device LLMs**: "MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases"

### Repos Similaires
- [GPT-2 Android](https://github.com/huggingface/tflite-android-transformers)
- [BLOOM Mobile](https://github.com/ml-opensource/bloom-mobile)
- [LLaMA.cpp](https://github.com/ggerganov/llama.cpp) (inspiration C++)

## 🤝 Contributions

Ce projet est un template de base. Améliorations bienvenues:
- Optimisations de performance
- Support de plus de tokenizers
- UI/UX améliorée
- Tests unitaires
- CI/CD

## 📄 Licence

Adaptez selon votre projet HessGPT.

---

**Créé avec ❤️ pour faire tourner des LLMs partout, même offline!**
