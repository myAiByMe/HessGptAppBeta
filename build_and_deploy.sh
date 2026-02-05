#!/bin/bash

# Script de build et déploiement automatisé pour HessGPT Mobile
# Usage: ./build_and_deploy.sh [checkpoint_path]

set -e  # Arrêter en cas d'erreur

echo "🚀 HessGPT Mobile - Build & Deploy"
echo "=================================="

# Couleurs pour la sortie
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
CHECKPOINT_PATH=${1:-""}
ASSETS_DIR="app/src/main/assets"
PYTHON_DIR="python"
MODEL_NAME="model.ptl"
TOKENIZER_NAME="tokenizer.json"

# Fonction d'affichage
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Vérifier Python
check_python() {
    log_info "Vérification de Python..."
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 n'est pas installé!"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    log_info "Python version: $PYTHON_VERSION"
    
    # Vérifier PyTorch
    if ! python3 -c "import torch" 2>/dev/null; then
        log_warn "PyTorch n'est pas installé. Installation en cours..."
        pip3 install torch torchvision
    fi
}

# Créer les dossiers nécessaires
setup_directories() {
    log_info "Création des dossiers..."
    mkdir -p "$ASSETS_DIR"
}

# Exporter le modèle
export_model() {
    log_info "Export du modèle..."
    
    cd "$PYTHON_DIR"
    
    if [ -z "$CHECKPOINT_PATH" ]; then
        log_warn "Pas de checkpoint fourni, création d'un modèle de démo..."
        python3 export_model.py \
            --checkpoint demo \
            --output "../$ASSETS_DIR/$MODEL_NAME"
    else
        if [ ! -f "$CHECKPOINT_PATH" ]; then
            log_error "Le checkpoint n'existe pas: $CHECKPOINT_PATH"
            exit 1
        fi
        
        log_info "Export depuis: $CHECKPOINT_PATH"
        python3 export_model.py \
            --checkpoint "$CHECKPOINT_PATH" \
            --output "../$ASSETS_DIR/$MODEL_NAME" \
            --max-seq-len 512
    fi
    
    cd ..
}

# Quantifier le modèle
quantize_model() {
    log_info "Quantification du modèle..."
    
    cd "$PYTHON_DIR"
    
    # Vérifier si le modèle existe
    if [ ! -f "../$ASSETS_DIR/$MODEL_NAME" ]; then
        log_error "Modèle non trouvé, impossible de quantifier"
        exit 1
    fi
    
    # Quantifier
    python3 quantize_model.py \
        --input "../$ASSETS_DIR/$MODEL_NAME" \
        --output "../$ASSETS_DIR/model_q8.ptl" \
        --compare
    
    # Remplacer par la version quantifiée
    mv "../$ASSETS_DIR/model_q8.ptl" "../$ASSETS_DIR/$MODEL_NAME"
    
    cd ..
}

# Créer le tokenizer
create_tokenizer() {
    log_info "Création du tokenizer..."
    
    cd "$PYTHON_DIR"
    
    # Utiliser le tokenizer HF si spécifié, sinon créer un démo
    if [ ! -z "$TOKENIZER_PATH" ]; then
        python3 create_tokenizer.py \
            --from-hf "$TOKENIZER_PATH" \
            --output "../$ASSETS_DIR/$TOKENIZER_NAME"
    else
        log_warn "Utilisation d'un tokenizer de démo (pour tests seulement)"
        python3 create_tokenizer.py \
            --output "../$ASSETS_DIR/$TOKENIZER_NAME" \
            --vocab-size 1000
    fi
    
    cd ..
}

# Tester le modèle
test_model() {
    log_info "Test du modèle..."
    
    cd "$PYTHON_DIR"
    
    python3 test_model.py \
        --model "../$ASSETS_DIR/$MODEL_NAME" \
        --inspect \
        --iterations 5
    
    cd ..
}

# Vérifier les assets
check_assets() {
    log_info "Vérification des assets..."
    
    if [ ! -f "$ASSETS_DIR/$MODEL_NAME" ]; then
        log_error "Modèle manquant: $ASSETS_DIR/$MODEL_NAME"
        exit 1
    fi
    
    if [ ! -f "$ASSETS_DIR/$TOKENIZER_NAME" ]; then
        log_error "Tokenizer manquant: $ASSETS_DIR/$TOKENIZER_NAME"
        exit 1
    fi
    
    # Afficher les tailles
    MODEL_SIZE=$(du -h "$ASSETS_DIR/$MODEL_NAME" | cut -f1)
    TOKEN_SIZE=$(du -h "$ASSETS_DIR/$TOKENIZER_NAME" | cut -f1)
    
    log_info "Taille du modèle: $MODEL_SIZE"
    log_info "Taille du tokenizer: $TOKEN_SIZE"
    
    # Avertir si le modèle est trop gros
    MODEL_SIZE_MB=$(du -m "$ASSETS_DIR/$MODEL_NAME" | cut -f1)
    if [ $MODEL_SIZE_MB -gt 200 ]; then
        log_warn "Le modèle est volumineux (${MODEL_SIZE_MB}MB). Envisagez la quantification."
    fi
}

# Build Android
build_android() {
    log_info "Build de l'application Android..."
    
    # Vérifier que Gradle est disponible
    if [ -f "./gradlew" ]; then
        chmod +x ./gradlew
        ./gradlew assembleDebug
    else
        log_error "Gradle wrapper non trouvé. Utilisez Android Studio pour le build."
        return 1
    fi
}

# Installer sur appareil
install_app() {
    log_info "Installation sur l'appareil..."
    
    # Vérifier ADB
    if ! command -v adb &> /dev/null; then
        log_warn "ADB non trouvé, installation manuelle requise"
        log_info "APK disponible dans: app/build/outputs/apk/debug/"
        return
    fi
    
    # Vérifier qu'un appareil est connecté
    DEVICES=$(adb devices | grep -v "List" | grep "device" | wc -l)
    if [ $DEVICES -eq 0 ]; then
        log_warn "Aucun appareil Android connecté"
        log_info "APK disponible dans: app/build/outputs/apk/debug/"
        return
    fi
    
    # Installer
    APK_PATH="app/build/outputs/apk/debug/app-debug.apk"
    if [ -f "$APK_PATH" ]; then
        adb install -r "$APK_PATH"
        log_info "✅ Application installée avec succès!"
    else
        log_error "APK non trouvé: $APK_PATH"
    fi
}

# Menu principal
main() {
    echo ""
    log_info "Début du processus de build..."
    echo ""
    
    # Étape 1: Vérifications
    check_python
    setup_directories
    
    # Étape 2: Préparer le modèle
    echo ""
    read -p "Voulez-vous exporter/créer le modèle? (o/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Oo]$ ]]; then
        export_model
        
        echo ""
        read -p "Voulez-vous quantifier le modèle? (O/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            quantize_model
        fi
    fi
    
    # Étape 3: Créer le tokenizer
    echo ""
    read -p "Voulez-vous créer/mettre à jour le tokenizer? (o/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Oo]$ ]]; then
        create_tokenizer
    fi
    
    # Étape 4: Tester
    echo ""
    read -p "Voulez-vous tester le modèle? (O/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        test_model
    fi
    
    # Étape 5: Vérifier les assets
    check_assets
    
    # Étape 6: Build Android
    echo ""
    read -p "Voulez-vous builder l'application Android? (O/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        build_android
        
        echo ""
        read -p "Voulez-vous installer sur un appareil? (o/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Oo]$ ]]; then
            install_app
        fi
    fi
    
    # Résumé
    echo ""
    echo "=================================="
    log_info "Build terminé!"
    echo "=================================="
    echo ""
    log_info "Prochaines étapes:"
    echo "  1. Ouvrez le projet dans Android Studio"
    echo "  2. Synchronisez Gradle"
    echo "  3. Connectez un appareil Android (API 26+)"
    echo "  4. Cliquez sur Run (▶️)"
    echo ""
    log_info "Assets créés:"
    echo "  📦 Modèle: $ASSETS_DIR/$MODEL_NAME"
    echo "  📝 Tokenizer: $ASSETS_DIR/$TOKENIZER_NAME"
    echo ""
}

# Lancer le script
main
