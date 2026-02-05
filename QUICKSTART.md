# 🚀 Guide de Démarrage Rapide

Ce guide vous aidera à déployer votre modèle HessGPT sur Android en quelques étapes.

## 📋 Prérequis

### 1. Environnement de développement
- **Android Studio** Arctic Fox ou supérieur
- **JDK 11+**
- **Python 3.8+** avec PyTorch 2.0+

### 2. Vérifier votre installation Python
```bash
python --version  # Devrait afficher Python 3.8+
pip install torch torchvision  # Si pas déjà installé
```

## 🔨 Étapes d'Installation

### Étape 1: Préparer votre modèle

#### Option A: Utiliser un modèle de démo (pour tester)
```bash
cd python

# Créer un tokenizer de démo
python create_tokenizer.py --output ../app/src/main/assets/tokenizer.json

# Créer un modèle de démo (petit, pour tester l'app)
python export_model.py --checkpoint demo --output ../app/src/main/assets/model.ptl
```

#### Option B: Utiliser votre vrai modèle HessGPT
```bash
cd python

# 1. Exporter votre modèle entraîné
python export_model.py \
    --checkpoint /path/to/your/HessGPT/checkpoint.pt \
    --output ../app/src/main/assets/model.ptl \
    --max-seq-len 512

# 2. Quantifier pour optimiser (RECOMMANDÉ)
python quantize_model.py \
    --input ../app/src/main/assets/model.ptl \
    --output ../app/src/main/assets/model_quantized.ptl \
    --compare

# 3. Utiliser le modèle quantifié
mv ../app/src/main/assets/model_quantized.ptl ../app/src/main/assets/model.ptl

# 4. Créer le tokenizer
# Si vous avez un tokenizer HuggingFace:
python create_tokenizer.py \
    --from-hf /path/to/your/tokenizer \
    --output ../app/src/main/assets/tokenizer.json
```

### Étape 2: Vérifier que le modèle fonctionne
```bash
cd python

# Tester le modèle avant déploiement
python test_model.py --model ../app/src/main/assets/model.ptl --inspect

# Vous devriez voir:
# ✅ Tous les tests sont passés!
# ⚡ Latence moyenne: XXX ms
# 🚀 Débit: X.X tokens/sec
```

### Étape 3: Adapter le code à votre modèle

#### Modifier `export_model.py`

Ouvrez `python/export_model.py` et remplacez la classe `DemoGPT` par votre vraie architecture:

```python
# Importez votre modèle
sys.path.append("../HessGpt_RoPE")
from Core.model import HessGPT, ModelConfig

# Dans la fonction export_to_torchscript():
config = ModelConfig(
    vocab_size=50257,  # Votre taille de vocabulaire
    n_embd=768,        # Dimension d'embedding
    n_head=12,         # Nombre de têtes d'attention
    n_layer=12,        # Nombre de couches
    max_seq_len=max_seq_len,
    # Ajoutez vos paramètres RoPE ici
)

model = HessGPT(config)
model.load_state_dict(checkpoint['model_state_dict'])
```

### Étape 4: Build l'application Android

1. **Ouvrir le projet dans Android Studio**
   ```bash
   # Ouvrir Android Studio
   # File > Open > Sélectionner HessGptMobileApp/
   ```

2. **Synchroniser Gradle**
   - Android Studio va automatiquement télécharger les dépendances
   - Attendez que la synchronisation se termine

3. **Vérifier les assets**
   - Assurez-vous que `app/src/main/assets/` contient:
     - `model.ptl` (ou `model_quantized.ptl`)
     - `tokenizer.json`

4. **Build et Run**
   - Connectez un appareil Android (API 26+) ou lancez un émulateur
   - Cliquez sur le bouton "Run" (▶️) dans Android Studio
   - L'app va se compiler et s'installer sur votre appareil

### Étape 5: Tester l'application

1. **Premier lancement**
   - L'app va charger le modèle (peut prendre 5-30 secondes selon la taille)
   - Vous verrez "Modèle chargé ✓" quand c'est prêt

2. **Envoyer un message**
   - Tapez une question dans le champ de texte
   - Appuyez sur le bouton d'envoi
   - Le modèle va générer une réponse en streaming

3. **Surveiller les performances**
   - La barre de statut affiche le débit (tokens/sec)
   - Vérifiez les logs dans Logcat pour plus de détails

## 🎯 Optimisations Recommandées

### 1. Réduire la taille du modèle

Si votre modèle est trop gros (>200MB):

```bash
# Quantification dynamique INT8
python quantize_model.py --input model.ptl --output model_q8.ptl

# Si encore trop gros, réduisez la taille du modèle à l'entraînement:
# - Moins de couches (n_layer)
# - Dimension plus petite (n_embd)
# - Distillation de modèle
```

### 2. Améliorer la vitesse d'inférence

Dans `ModelManager.kt`, ajustez:
```kotlin
// Réduire max_seq_length
private val maxSeqLength = 256  // Au lieu de 512

// Réduire maxNewTokens dans generate()
maxNewTokens = 50  // Au lieu de 100
```

### 3. Optimiser la mémoire

Dans `app/build.gradle.kts`, ajoutez:
```kotlin
android {
    defaultConfig {
        // Limiter aux ABIs nécessaires
        ndk {
            abiFilters += listOf("arm64-v8a")  // Uniquement 64-bit
        }
    }
}
```

## ❓ Problèmes Courants

### "Le modèle ne charge pas"
- Vérifiez que `model.ptl` est dans `app/src/main/assets/`
- Vérifiez la taille: doit être <200MB pour la plupart des appareils
- Regardez les logs Logcat pour l'erreur exacte

### "OutOfMemoryError"
- Utilisez un modèle quantifié (INT8)
- Réduisez `maxSeqLength`
- Testez sur un appareil avec plus de RAM (6GB+)

### "Génération trop lente"
- Quantifiez le modèle
- Réduisez le nombre de tokens générés
- Testez sur un appareil plus récent (Snapdragon 7xx+)

### "Tokens incompréhensibles"
- Vérifiez que `tokenizer.json` correspond à votre modèle
- Le tokenizer de démo est très basique, utilisez votre vrai tokenizer

## 📚 Ressources

- [Documentation PyTorch Mobile](https://pytorch.org/mobile/)
- [Guide Android Studio](https://developer.android.com/studio)
- [Optimisation des modèles](https://pytorch.org/tutorials/recipes/mobile_interpreter.html)

## 🎓 Prochaines Étapes

1. **Améliorer le tokenizer**: Utilisez votre vrai tokenizer BPE
2. **Ajouter des prompts système**: Guides de conversation
3. **Sauvegarder l'historique**: Avec Room Database
4. **Partager les conversations**: Export en texte/JSON
5. **Mode sombre**: Thème personnalisable
6. **Voix**: Synthèse vocale pour les réponses

## 🆘 Support

Si vous rencontrez des problèmes:
1. Vérifiez les logs dans Logcat (Android Studio)
2. Testez le modèle avec `test_model.py` avant déploiement
3. Commencez avec un petit modèle de test
4. Augmentez progressivement la taille

Bonne chance ! 🚀
