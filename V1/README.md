# 🔍 OCR Intelligent + Google Gemini AI

Outil Python de traitement de texte intelligent combinant **Tesseract OCR** et **Google Gemini AI**.

## ✨ Fonctionnalités

- **Extraction OCR** : Support multilingue (Français, Arabe, Anglais)
- **Amélioration d'image** : Débruitage et seuillage Otsu automatique
- **Correction IA** : Correction des erreurs OCR, orthographe et ponctuation via Google Gemini
- **Détection de type** : Identification automatique du type de document (Lettre, Facture, CV, Contrat, etc.)
- **Sélection de modèle** : Choix parmi les modèles Gemini disponibles
- **Interface moderne** : GUI PySide6 intuitive et professionnelle

## 🛠️ Prérequis

### 1. Tesseract OCR
Téléchargez et installez Tesseract :
- **Windows** : [Télécharger Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- Installez les packs de langues : `fra`, `ara`, `eng`

### 2. Clé API Google Gemini
1. Rendez-vous sur [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Créez une clé API gratuite
3. Créez un fichier `.env` à la racine du projet :
```env
GOOGLE_API_KEY=votre_cle_api_ici
```

## 📦 Installation

```bash
# Cloner ou télécharger le projet

# Installer les dépendances
pip install -r requirements.txt
```

## 🚀 Utilisation

```bash
python OCR_Gemini_AI.py
```

### Workflow :
1. **Charger une image** contenant du texte
2. **Configurer les options** OCR (amélioration d'image, langues)
3. **Sélectionner le modèle Gemini** souhaité
4. **Lancer l'extraction OCR**
5. **Confirmer le traitement IA** pour corriger et analyser le texte

## 📁 Structure du Projet

```
V1 - tesseract/
├── OCR_Gemini_AI.py       # Script principal avec IA Gemini
├── ImageProcessorFixed.py  # Version basique (OCR seul)
├── requirements.txt        # Dépendances Python
├── .env                    # Configuration API (à créer)
├── .env.example           # Template de configuration
└── README.md              # Documentation
```

## 🔧 Configuration

### Modèles Gemini disponibles :
| Modèle | Description |
|--------|-------------|
| `gemini-2.0-flash` | Très rapide, bon rapport qualité/vitesse |
| `gemini-1.5-flash` | Rapide et efficace |
| `gemini-1.5-flash-8b` | Version légère |
| `gemini-1.5-pro` | Plus puissant, meilleure qualité |

### Types de documents détectés :
- Lettre formelle / Courrier officiel
- Facture / Devis
- Contrat / Document juridique
- CV / Curriculum Vitae
- Article / Publication
- Rapport / Compte-rendu
- Formulaire administratif
- Document médical
- Document académique / Diplôme
- Correspondance personnelle
- Document commercial
- Et plus...

## ⚙️ Architecture du Code

Le script est organisé en classes modulaires :

- **`GeminiAPIManager`** : Gère la connexion à l'API Google, liste les modèles et traite les requêtes
- **`AIProcessingThread`** : Thread pour le traitement IA non-bloquant
- **`ModelSelectionDialog`** : Dialogue de sélection du modèle
- **`AIConfirmationDialog`** : Dialogue de confirmation avant traitement IA
- **`ImageProcessorInterface`** : Interface principale PySide6

## ⚠️ Gestion des erreurs

Le script gère automatiquement :
- ✅ Clé API manquante ou invalide
- ✅ Connexion API échouée
- ✅ Texte vide après OCR
- ✅ Chemins de fichiers avec caractères spéciaux (accents, arabe)
- ✅ Formats d'image non supportés
- ✅ Bibliothèque google-generativeai non installée

## 🔐 Sécurité

- La clé API est stockée dans un fichier `.env` (non versionné)
- Ne partagez jamais votre fichier `.env`
- Ajoutez `.env` à votre `.gitignore`

## 📝 Exemple de .env

```env
# Configuration de l'API Google Gemini
GOOGLE_API_KEY=YourApiKey
```

## 🐛 Dépannage

### "Bibliothèque non installée"
```bash
pip install google-generativeai python-dotenv
```

### "Clé API non configurée"
Créez un fichier `.env` avec votre clé API Google.

### "Erreur Tesseract"
Vérifiez que Tesseract est installé et que le chemin est correct dans le script.
