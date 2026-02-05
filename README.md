# HessGPT Mobile - Application Android Offline

Application mobile Android native (Kotlin) utilisant PyTorch Mobile pour faire tourner votre modèle HessGPT (0.5B paramètres) en mode offline.

## 📁 Structure du Projet

```
HessGptMobileApp/
├── app/
│   ├── src/
│   │   ├── main/
│   │   │   ├── java/com/hessgpt/mobile/
│   │   │   │   ├── MainActivity.kt
│   │   │   │   ├── ui/
│   │   │   │   │   ├── ChatActivity.kt
│   │   │   │   │   ├── ChatAdapter.kt
│   │   │   │   │   └── MessageViewHolder.kt
│   │   │   │   ├── ml/
│   │   │   │   │   ├── ModelManager.kt
│   │   │   │   │   ├── Tokenizer.kt
│   │   │   │   │   └── InferenceEngine.kt
│   │   │   │   ├── data/
│   │   │   │   │   ├── Message.kt
│   │   │   │   │   └── ChatRepository.kt
│   │   │   │   └── utils/
│   │   │   │       ├── FileUtils.kt
│   │   │   │       └── PerformanceMonitor.kt
│   │   │   ├── res/
│   │   │   │   ├── layout/
│   │   │   │   ├── values/
│   │   │   │   └── drawable/
│   │   │   ├── assets/
│   │   │   │   ├── model.ptl          # Modèle PyTorch Mobile
│   │   │   │   └── tokenizer.json     # Tokenizer
│   │   │   └── AndroidManifest.xml
│   ├── build.gradle.kts
│   └── proguard-rules.pro
├── python/
│   ├── export_model.py                 # Script pour exporter le modèle
│   ├── quantize_model.py              # Script pour quantifier le modèle
│   └── test_model.py                  # Script de test
├── build.gradle.kts
├── settings.gradle.kts
└── README.md
```

## 🔧 Technologies Utilisées

- **Android**: API Level 26+ (Android 8.0+)
- **Language**: Kotlin
- **ML Framework**: PyTorch Mobile 1.13+
- **Architecture**: MVVM avec Coroutines
- **UI**: Material Design 3

## 📊 Spécifications du Modèle

- **Taille**: 0.5B paramètres
- **Architecture**: GPT avec RoPE (Rotary Position Embeddings)
- **Format**: TorchScript (.ptl)
- **Quantization**: INT8 pour optimiser la taille et la vitesse

## 🚀 Installation

### Prérequis

1. Android Studio Arctic Fox ou supérieur
2. JDK 11+
3. Python 3.8+ (pour l'export du modèle)
4. PyTorch 2.0+

### Étapes

1. **Exporter le modèle PyTorch vers TorchScript**:
```bash
cd python
python export_model.py --checkpoint /path/to/your/checkpoint.pt --output ../app/src/main/assets/model.ptl
```

2. **Quantifier le modèle (optionnel mais recommandé)**:
```bash
python quantize_model.py --input ../app/src/main/assets/model.ptl --output ../app/src/main/assets/model_quantized.ptl
```

3. **Ouvrir le projet dans Android Studio**
4. **Synchroniser Gradle**
5. **Build et Run**

## 📱 Fonctionnalités

- ✅ Inférence 100% offline
- ✅ Chat interactif en temps réel
- ✅ Streaming de tokens
- ✅ Historique des conversations
- ✅ Optimisé pour performance mobile
- ✅ Support multi-thread
- ✅ Monitoring des performances (tokens/sec, latence)

## 🎯 Optimisations

1. **Quantization INT8**: Réduit la taille du modèle de ~75%
2. **KV-Cache**: Optimise la génération de tokens successifs
3. **Thread Pool**: Utilise plusieurs threads pour l'inférence
4. **Memory Management**: Libération automatique de la mémoire

## 📈 Performances Attendues

Sur un appareil mid-range (Snapdragon 720G, 6GB RAM):
- **Latence première token**: 800-1200ms
- **Throughput**: 3-5 tokens/sec
- **Mémoire**: ~2GB RAM

## 🔐 Permissions

```xml
<uses-permission android:name="android.permission.INTERNET"/> <!-- Optionnel, pour télécharger le modèle -->
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE"/> <!-- Pour sauvegarder les conversations -->
```

## 📝 Licence

Votre licence ici

## 🤝 Contribution

Contributions bienvenues ! Veuillez soumettre un pull request.
