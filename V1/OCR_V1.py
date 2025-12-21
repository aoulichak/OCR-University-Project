"""
=============================================================================
 Outil de Traitement de Texte Intelligent : OCR (Tesseract) + Google Gemini
=============================================================================
 Auteur: A. Mohamed
 Version: 2.0
 Date: Décembre 2025
 
 Description:
 - Extraction de texte via Tesseract OCR (Français, Arabe, Anglais)
 - Correction intelligente par Google Gemini AI
 - Détection automatique du type de document
 - Interface graphique moderne avec PySide6
=============================================================================
"""

import sys
import os
from dotenv import load_dotenv

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QPlainTextEdit, QFileDialog, QSizePolicy,
    QGroupBox, QCheckBox, QMessageBox, QStatusBar, QComboBox,
    QProgressBar, QDialog, QDialogButtonBox, QRadioButton, QButtonGroup
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont

import cv2
import pytesseract
import numpy as np

# Import Google Generative AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ google-generativeai non installé. Installez avec: pip install google-generativeai")

# --- CONFIGURATION ---
# Charger les variables d'environnement depuis .env
load_dotenv()

# Configuration Tesseract (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# =============================================================================
#                           CLASSES UTILITAIRES
# =============================================================================

class GeminiAPIManager:
    """
    Gestionnaire de l'API Google Gemini.
    Gère la connexion, la liste des modèles et les requêtes.
    """
    
    # Liste des modèles Gemini recommandés
    RECOMMENDED_MODELS = [
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro",
    ]
    
    def __init__(self):
        self.api_key = None
        self.model = None
        self.model_name = None
        self.available_models = []
        
    def configure(self, api_key: str) -> bool:
        """Configure l'API avec la clé fournie."""
        try:
            self.api_key = api_key
            genai.configure(api_key=api_key)
            return True
        except Exception as e:
            print(f"Erreur de configuration API: {e}")
            return False
    
    def fetch_available_models(self) -> list:
        """Récupère la liste des modèles disponibles supportant generateContent."""
        try:
            models = []
            for model in genai.list_models():
                # Filtrer les modèles qui supportent la génération de contenu
                if 'generateContent' in model.supported_generation_methods:
                    model_name = model.name.replace('models/', '')
                    models.append(model_name)
            
            # Prioriser les modèles recommandés
            prioritized = []
            for rec in self.RECOMMENDED_MODELS:
                if rec in models:
                    prioritized.append(rec)
            
            # Ajouter les autres modèles
            for m in models:
                if m not in prioritized and 'gemini' in m.lower():
                    prioritized.append(m)
                    
            self.available_models = prioritized if prioritized else models
            return self.available_models
            
        except Exception as e:
            print(f"Erreur lors de la récupération des modèles: {e}")
            return self.RECOMMENDED_MODELS  # Fallback
    
    def select_model(self, model_name: str) -> bool:
        """Sélectionne et initialise un modèle."""
        try:
            self.model = genai.GenerativeModel(model_name)
            self.model_name = model_name
            return True
        except Exception as e:
            print(f"Erreur de sélection du modèle: {e}")
            return False
    
    def process_text(self, raw_text: str) -> dict:
        """
        Envoie le texte à Gemini pour correction et détection de type.
        Retourne un dictionnaire avec le texte corrigé et le type de document.
        """
        if not self.model:
            return {"error": "Aucun modèle sélectionné"}
        
        prompt = f"""Tu es un assistant expert en traitement de texte OCR. Analyse le texte suivant qui a été extrait par OCR et peut contenir des erreurs.

TEXTE BRUT EXTRAIT PAR OCR:
\"\"\"
{raw_text}
\"\"\"

INSTRUCTIONS:
1. **CORRECTION**: Corrige toutes les erreurs de lecture OCR (caractères mal reconnus, espaces incorrects), l'orthographe, la grammaire et la ponctuation. Reconstitue les mots coupés ou mal formés.

2. **DÉTECTION DE TYPE**: Identifie le type de document parmi les catégories suivantes:
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
   - Autre (préciser)

RÉPONDS EXACTEMENT AU FORMAT SUIVANT (respecte les balises):

<TYPE_DOCUMENT>
[Indique ici le type de document détecté]
</TYPE_DOCUMENT>

<TEXTE_CORRIGE>
[Insère ici le texte entièrement corrigé et reformaté proprement]
</TEXTE_CORRIGE>

<CONFIANCE>
[Indique ton niveau de confiance pour la détection: Élevé/Moyen/Faible]
</CONFIANCE>
"""
        
        try:
            response = self.model.generate_content(prompt)
            result_text = response.text
            
            # Parser la réponse
            return self._parse_response(result_text)
            
        except Exception as e:
            return {"error": f"Erreur API Gemini: {str(e)}"}
    
    def _parse_response(self, response_text: str) -> dict:
        """Parse la réponse structurée de Gemini."""
        result = {
            "type_document": "Non détecté",
            "texte_corrige": "",
            "confiance": "Non spécifié",
            "raw_response": response_text
        }
        
        try:
            # Extraire le type de document
            if "<TYPE_DOCUMENT>" in response_text and "</TYPE_DOCUMENT>" in response_text:
                start = response_text.find("<TYPE_DOCUMENT>") + len("<TYPE_DOCUMENT>")
                end = response_text.find("</TYPE_DOCUMENT>")
                result["type_document"] = response_text[start:end].strip()
            
            # Extraire le texte corrigé
            if "<TEXTE_CORRIGE>" in response_text and "</TEXTE_CORRIGE>" in response_text:
                start = response_text.find("<TEXTE_CORRIGE>") + len("<TEXTE_CORRIGE>")
                end = response_text.find("</TEXTE_CORRIGE>")
                result["texte_corrige"] = response_text[start:end].strip()
            
            # Extraire le niveau de confiance
            if "<CONFIANCE>" in response_text and "</CONFIANCE>" in response_text:
                start = response_text.find("<CONFIANCE>") + len("<CONFIANCE>")
                end = response_text.find("</CONFIANCE>")
                result["confiance"] = response_text[start:end].strip()
                
        except Exception as e:
            result["error"] = f"Erreur de parsing: {str(e)}"
            result["texte_corrige"] = response_text  # Fallback
            
        return result


class AIProcessingThread(QThread):
    """Thread pour le traitement IA en arrière-plan (non bloquant)."""
    finished = Signal(dict)
    error = Signal(str)
    
    def __init__(self, gemini_manager, text):
        super().__init__()
        self.gemini_manager = gemini_manager
        self.text = text
    
    def run(self):
        try:
            result = self.gemini_manager.process_text(self.text)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class ModelSelectionDialog(QDialog):
    """Dialogue pour sélectionner le modèle Gemini."""
    
    def __init__(self, models: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sélection du Modèle Google Gemini")
        self.setMinimumWidth(450)
        self.selected_model = None
        
        layout = QVBoxLayout(self)
        
        # Instructions
        info_label = QLabel(
            "🤖 Choisissez le modèle Google Gemini à utiliser pour la correction:\n\n"
            "• gemini-2.0-flash : Très rapide, bon rapport qualité/vitesse\n"
            "• gemini-1.5-flash : Rapide et efficace\n"
            "• gemini-1.5-pro : Plus puissant, meilleure qualité"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #2c3e50; padding: 10px; background-color: #ecf0f1; border-radius: 5px;")
        layout.addWidget(info_label)
        
        # ComboBox pour la sélection
        self.model_combo = QComboBox()
        self.model_combo.addItems(models)
        self.model_combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                font-size: 11pt;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.model_combo)
        
        # Boutons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def get_selected_model(self) -> str:
        return self.model_combo.currentText()


class AIConfirmationDialog(QDialog):
    """Dialogue pour confirmer le lancement du traitement IA."""
    
    def __init__(self, text_preview: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Traitement par Intelligence Artificielle")
        self.setMinimumSize(500, 350)
        
        layout = QVBoxLayout(self)
        
        # Message d'information
        info_label = QLabel(
            "✨ Le texte a été extrait avec succès par Tesseract OCR.\n\n"
            "Voulez-vous envoyer ce texte à Google Gemini pour:\n"
            "  • Corriger les erreurs OCR et d'orthographe\n"
            "  • Détecter automatiquement le type de document"
        )
        info_label.setStyleSheet("font-size: 10pt; color: #2c3e50;")
        layout.addWidget(info_label)
        
        # Aperçu du texte
        preview_group = QGroupBox("Aperçu du texte extrait:")
        preview_layout = QVBoxLayout(preview_group)
        
        preview_text = QPlainTextEdit()
        preview_text.setPlainText(text_preview[:500] + ("..." if len(text_preview) > 500 else ""))
        preview_text.setReadOnly(True)
        preview_text.setMaximumHeight(150)
        preview_text.setStyleSheet("background-color: #f8f9fa; color: #2c3e50;")
        preview_layout.addWidget(preview_text)
        
        layout.addWidget(preview_group)
        
        # Boutons
        button_layout = QHBoxLayout()
        
        self.btn_yes = QPushButton("🚀 Oui, lancer le traitement IA")
        self.btn_yes.setStyleSheet("""
            QPushButton {
                background-color: #27ae60; color: white; 
                padding: 10px 20px; border-radius: 5px; font-weight: bold;
            }
            QPushButton:hover { background-color: #219a52; }
        """)
        self.btn_yes.clicked.connect(self.accept)
        
        self.btn_no = QPushButton("Non, garder le texte brut")
        self.btn_no.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6; color: white;
                padding: 10px 20px; border-radius: 5px;
            }
            QPushButton:hover { background-color: #7f8c8d; }
        """)
        self.btn_no.clicked.connect(self.reject)
        
        button_layout.addWidget(self.btn_no)
        button_layout.addWidget(self.btn_yes)
        layout.addLayout(button_layout)


# =============================================================================
#                           INTERFACE PRINCIPALE
# =============================================================================

class ImageProcessorInterface(QMainWindow):
    """Interface principale de l'application OCR + IA."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔍 OCR Intelligent + Google Gemini AI")
        self.setGeometry(100, 100, 1000, 800)
        
        # Variables d'état
        self.current_file_path = None
        self.raw_extracted_text = ""
        self.gemini_manager = GeminiAPIManager() if GEMINI_AVAILABLE else None
        self.ai_thread = None
        
        # Initialiser l'API Gemini
        self._init_gemini_api()
        
        # Style général
        self.setStyleSheet(self._get_professional_stylesheet())
        
        # Construction de l'interface
        self._build_ui()
        
    def _init_gemini_api(self):
        """Initialise l'API Google Gemini avec la clé d'environnement."""
        if not GEMINI_AVAILABLE:
            return
            
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if api_key and api_key != "votre_cle_api_google_ici":
            if self.gemini_manager.configure(api_key):
                models = self.gemini_manager.fetch_available_models()
                if models:
                    self.gemini_manager.select_model(models[0])  # Modèle par défaut
        else:
            print("⚠️ Clé API Google non configurée. Créez un fichier .env avec GOOGLE_API_KEY=votre_clé")
    
    def _build_ui(self):
        """Construit l'interface utilisateur."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(25, 25, 25, 25)

        # === SECTION SUPÉRIEURE: Contrôles ===
        top_layout = QHBoxLayout()
        top_layout.setSpacing(20)

        # 1. Zone Fichier
        file_group = QGroupBox("1. Fichier Source")
        file_layout = QVBoxLayout(file_group)
        
        self.upload_btn = QPushButton("📂 Charger une Image")
        self.upload_btn.setObjectName("PrimaryButton")
        self.upload_btn.clicked.connect(self.upload_file)
        file_layout.addWidget(self.upload_btn)
        
        self.status_label = QLabel("Aucun fichier sélectionné")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
        file_layout.addWidget(self.status_label)
        
        top_layout.addWidget(file_group, 2)

        # 2. Zone Options OCR
        options_group = QGroupBox("2. Options OCR")
        options_layout = QVBoxLayout(options_group)
        
        self.option_enhance = QCheckBox("🖼️ Amélioration d'image (Bruit/Otsu)")
        self.option_enhance.setChecked(True)
        self.option_enhance.setToolTip("Optimise la qualité de l'image avant l'OCR")
        
        self.option_ocr = QCheckBox("📝 Extraction OCR (FR + AR + EN)")
        self.option_ocr.setChecked(True)
        self.option_ocr.setToolTip("Extrait le texte en français, arabe et anglais")
        
        options_layout.addWidget(self.option_enhance)
        options_layout.addWidget(self.option_ocr)
        
        top_layout.addWidget(options_group, 2)

        # 3. Zone Modèle IA
        ai_group = QGroupBox("3. Configuration IA (Google Gemini)")
        ai_layout = QVBoxLayout(ai_group)
        
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Modèle:"))
        
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(180)
        if self.gemini_manager and self.gemini_manager.available_models:
            self.model_combo.addItems(self.gemini_manager.available_models)
        else:
            self.model_combo.addItems(GeminiAPIManager.RECOMMENDED_MODELS)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        model_row.addWidget(self.model_combo)
        
        self.refresh_models_btn = QPushButton("🔄")
        self.refresh_models_btn.setToolTip("Actualiser la liste des modèles")
        self.refresh_models_btn.setFixedWidth(35)
        self.refresh_models_btn.clicked.connect(self._refresh_models)
        model_row.addWidget(self.refresh_models_btn)
        
        ai_layout.addLayout(model_row)
        
        # Status API
        self.api_status_label = QLabel()
        self._update_api_status()
        ai_layout.addWidget(self.api_status_label)
        
        top_layout.addWidget(ai_group, 3)
        
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self._create_separator())

        # === SECTION MILIEU: Actions ===
        action_layout = QHBoxLayout()
        action_layout.addStretch(1)
        
        self.start_btn = QPushButton("▶️  Démarrer l'Extraction OCR")
        self.start_btn.setObjectName("AccentButton")
        self.start_btn.setMinimumHeight(50)
        self.start_btn.setMinimumWidth(250)
        self.start_btn.clicked.connect(self.launch_processing)
        action_layout.addWidget(self.start_btn)
        
        action_layout.addStretch(1)
        main_layout.addLayout(action_layout)
        
        # Barre de progression
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        main_layout.addWidget(self.progress_bar)

        # === SECTION RÉSULTATS ===
        results_layout = QHBoxLayout()
        results_layout.setSpacing(15)
        
        # Zone Texte Brut OCR
        raw_group = QGroupBox("📄 Texte Brut (OCR)")
        raw_layout = QVBoxLayout(raw_group)
        self.raw_text_edit = QPlainTextEdit()
        self.raw_text_edit.setPlaceholderText("Le texte extrait par OCR apparaîtra ici...")
        self.raw_text_edit.setReadOnly(True)
        raw_layout.addWidget(self.raw_text_edit)
        
        copy_raw_btn = QPushButton("📋 Copier")
        copy_raw_btn.setObjectName("SecondaryButton")
        copy_raw_btn.clicked.connect(lambda: self._copy_to_clipboard(self.raw_text_edit.toPlainText()))
        raw_layout.addWidget(copy_raw_btn)
        
        results_layout.addWidget(raw_group)
        
        # Zone Texte Corrigé IA
        corrected_group = QGroupBox("✨ Texte Corrigé (IA)")
        corrected_layout = QVBoxLayout(corrected_group)
        
        # Label type de document
        self.doc_type_label = QLabel("Type de document: -")
        self.doc_type_label.setStyleSheet("""
            font-weight: bold; 
            color: #2980b9; 
            padding: 8px; 
            background-color: #ebf5fb;
            border-radius: 4px;
        """)
        corrected_layout.addWidget(self.doc_type_label)
        
        self.corrected_text_edit = QPlainTextEdit()
        self.corrected_text_edit.setPlaceholderText("Le texte corrigé par l'IA apparaîtra ici...")
        self.corrected_text_edit.setReadOnly(True)
        corrected_layout.addWidget(self.corrected_text_edit)
        
        copy_corrected_btn = QPushButton("📋 Copier")
        copy_corrected_btn.setObjectName("SecondaryButton")
        copy_corrected_btn.clicked.connect(lambda: self._copy_to_clipboard(self.corrected_text_edit.toPlainText()))
        corrected_layout.addWidget(copy_corrected_btn)
        
        results_layout.addWidget(corrected_group)
        
        main_layout.addLayout(results_layout)

        # === FOOTER ===
        footer_layout = QHBoxLayout()
        
        self.ai_process_btn = QPushButton("🤖 Lancer le Traitement IA")
        self.ai_process_btn.setObjectName("AIButton")
        self.ai_process_btn.setEnabled(False)
        self.ai_process_btn.clicked.connect(self._launch_ai_processing)
        footer_layout.addWidget(self.ai_process_btn)
        
        footer_layout.addStretch(1)
        
        copyright_label = QLabel("© 2025 A. Mohamed | Tous droits réservés.")
        copyright_label.setStyleSheet("color: #95a5a6; font-size: 9pt;")
        footer_layout.addWidget(copyright_label)
        
        main_layout.addLayout(footer_layout)

        # Barre de statut
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Prêt. Chargez une image pour commencer.")

    # =========================================================================
    #                           MÉTHODES LOGIQUES
    # =========================================================================

    def optimize_img(self, img_cv2):
        """Améliore l'image pour l'OCR (débruitage + seuillage Otsu)."""
        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        denoised = cv2.medianBlur(gray, 3)
        _, thresholded = cv2.threshold(
            denoised, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        return thresholded

    def extract_txt(self, processed_img) -> str:
        """Extrait le texte de l'image via Tesseract."""
        custom_config = r'--psm 6 -l fra+ara+eng'
        try:
            text = pytesseract.image_to_string(processed_img, config=custom_config)
            return text.strip()
        except Exception as e:
            return f"Erreur Tesseract: {str(e)}"

    # =========================================================================
    #                           HANDLERS D'ÉVÉNEMENTS
    # =========================================================================

    def upload_file(self):
        """Ouvre le dialogue de sélection de fichier."""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Sélectionner une Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)"
        )
        
        if file_name:
            self.current_file_path = file_name
            self.status_label.setText(f"📁 {os.path.basename(file_name)}")
            self.status_label.setStyleSheet("color: #27ae60; font-weight: bold;")
            self.status_bar.showMessage(f"Fichier chargé: {file_name}")
            
            # Réinitialiser les zones de texte
            self.raw_text_edit.clear()
            self.corrected_text_edit.clear()
            self.doc_type_label.setText("Type de document: -")
            self.ai_process_btn.setEnabled(False)
        else:
            self.status_bar.showMessage("Sélection annulée.")

    def launch_processing(self):
        """Lance l'extraction OCR."""
        if not self.current_file_path:
            QMessageBox.warning(self, "Erreur", "Veuillez d'abord charger une image.")
            return

        # Réinitialiser
        self.raw_text_edit.clear()
        self.corrected_text_edit.clear()
        self.doc_type_label.setText("Type de document: -")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Chargement de l'image...")
        QApplication.processEvents()

        # Charger l'image (support des caractères spéciaux dans le chemin)
        try:
            file_data = np.fromfile(self.current_file_path, dtype=np.uint8)
            img_source = cv2.imdecode(file_data, cv2.IMREAD_COLOR)
        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Erreur", f"Impossible de lire le fichier: {str(e)}")
            return

        if img_source is None:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Erreur", "Format d'image non reconnu ou fichier corrompu.")
            return

        working_image = img_source
        self.progress_bar.setValue(25)

        # Amélioration d'image
        if self.option_enhance.isChecked():
            self.progress_bar.setFormat("Amélioration de l'image...")
            QApplication.processEvents()
            working_image = self.optimize_img(working_image)
            self.progress_bar.setValue(50)

        # Extraction OCR
        if self.option_ocr.isChecked():
            self.progress_bar.setFormat("Extraction du texte (OCR)...")
            QApplication.processEvents()
            self.raw_extracted_text = self.extract_txt(working_image)
            self.progress_bar.setValue(100)
        
        self.progress_bar.setVisible(False)

        # Afficher le résultat
        if self.raw_extracted_text:
            self.raw_text_edit.setPlainText(self.raw_extracted_text)
            self.ai_process_btn.setEnabled(True)
            self.status_bar.showMessage("✅ Extraction OCR terminée. Vous pouvez lancer le traitement IA.")
            
            # Proposer le traitement IA
            self._prompt_ai_processing()
        else:
            self.raw_text_edit.setPlainText("(Aucun texte détecté)")
            self.ai_process_btn.setEnabled(False)
            self.status_bar.showMessage("⚠️ Aucun texte détecté dans l'image.")

    def _prompt_ai_processing(self):
        """Demande à l'utilisateur s'il souhaite lancer le traitement IA."""
        if not GEMINI_AVAILABLE or not self.gemini_manager:
            return
            
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "votre_cle_api_google_ici":
            QMessageBox.information(
                self, "Configuration requise",
                "Pour utiliser le traitement IA, configurez votre clé API Google:\n\n"
                "1. Créez un fichier .env à côté du script\n"
                "2. Ajoutez: GOOGLE_API_KEY=votre_clé_api\n\n"
                "Obtenez une clé sur: https://makersuite.google.com/app/apikey"
            )
            return
        
        # Dialogue de confirmation
        dialog = AIConfirmationDialog(self.raw_extracted_text, self)
        if dialog.exec() == QDialog.Accepted:
            self._launch_ai_processing()

    def _launch_ai_processing(self):
        """Lance le traitement par l'IA Google Gemini."""
        if not self.raw_extracted_text:
            QMessageBox.warning(self, "Erreur", "Aucun texte à traiter. Lancez d'abord l'extraction OCR.")
            return
        
        if not GEMINI_AVAILABLE:
            QMessageBox.critical(
                self, "Erreur",
                "La bibliothèque google-generativeai n'est pas installée.\n"
                "Installez-la avec: pip install google-generativeai"
            )
            return
        
        # Vérifier la configuration API
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "votre_cle_api_google_ici":
            QMessageBox.warning(
                self, "Configuration requise",
                "Clé API Google non configurée.\n\n"
                "Créez un fichier .env avec:\nGOOGLE_API_KEY=votre_clé"
            )
            return
        
        # Configurer si nécessaire
        if not self.gemini_manager.model:
            self.gemini_manager.configure(api_key)
            self.gemini_manager.select_model(self.model_combo.currentText())
        
        # Désactiver les contrôles pendant le traitement
        self.ai_process_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Mode indéterminé
        self.progress_bar.setFormat("Traitement IA en cours...")
        self.status_bar.showMessage(f"🤖 Envoi au modèle {self.model_combo.currentText()}...")
        QApplication.processEvents()
        
        # Lancer le thread de traitement
        self.ai_thread = AIProcessingThread(self.gemini_manager, self.raw_extracted_text)
        self.ai_thread.finished.connect(self._on_ai_finished)
        self.ai_thread.error.connect(self._on_ai_error)
        self.ai_thread.start()

    def _on_ai_finished(self, result: dict):
        """Callback quand le traitement IA est terminé."""
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        self.ai_process_btn.setEnabled(True)
        self.start_btn.setEnabled(True)
        
        if "error" in result:
            QMessageBox.warning(self, "Erreur IA", result["error"])
            self.status_bar.showMessage(f"❌ Erreur: {result['error']}")
            return
        
        # Afficher les résultats
        self.corrected_text_edit.setPlainText(result.get("texte_corrige", ""))
        
        doc_type = result.get("type_document", "Non détecté")
        confidence = result.get("confiance", "")
        self.doc_type_label.setText(f"📋 Type de document: {doc_type} (Confiance: {confidence})")
        
        self.status_bar.showMessage("✅ Traitement IA terminé avec succès!")

    def _on_ai_error(self, error_msg: str):
        """Callback en cas d'erreur du thread IA."""
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        self.ai_process_btn.setEnabled(True)
        self.start_btn.setEnabled(True)
        
        QMessageBox.critical(self, "Erreur", f"Erreur lors du traitement IA:\n{error_msg}")
        self.status_bar.showMessage(f"❌ Erreur IA: {error_msg}")

    def _on_model_changed(self, model_name: str):
        """Callback quand l'utilisateur change de modèle."""
        if self.gemini_manager and model_name:
            self.gemini_manager.select_model(model_name)
            self.status_bar.showMessage(f"Modèle sélectionné: {model_name}")

    def _refresh_models(self):
        """Actualise la liste des modèles disponibles."""
        if not GEMINI_AVAILABLE or not self.gemini_manager:
            return
            
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "votre_cle_api_google_ici":
            QMessageBox.warning(self, "Configuration", "Configurez d'abord votre clé API Google.")
            return
        
        self.gemini_manager.configure(api_key)
        models = self.gemini_manager.fetch_available_models()
        
        self.model_combo.clear()
        self.model_combo.addItems(models)
        self._update_api_status()
        self.status_bar.showMessage(f"✅ {len(models)} modèles disponibles.")

    def _update_api_status(self):
        """Met à jour l'affichage du statut de l'API."""
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if not GEMINI_AVAILABLE:
            self.api_status_label.setText("❌ Bibliothèque non installée")
            self.api_status_label.setStyleSheet("color: #e74c3c;")
        elif not api_key or api_key == "votre_cle_api_google_ici":
            self.api_status_label.setText("⚠️ Clé API non configurée")
            self.api_status_label.setStyleSheet("color: #f39c12;")
        else:
            self.api_status_label.setText("✅ API configurée")
            self.api_status_label.setStyleSheet("color: #27ae60;")

    def _copy_to_clipboard(self, text: str):
        """Copie le texte dans le presse-papiers."""
        if text:
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
            self.status_bar.showMessage("📋 Texte copié dans le presse-papiers!")
        else:
            self.status_bar.showMessage("⚠️ Rien à copier.")

    def _create_separator(self):
        """Crée un séparateur horizontal."""
        separator = QWidget()
        separator.setFixedHeight(1)
        separator.setStyleSheet("background-color: #bdc3c7;")
        return separator

    def _get_professional_stylesheet(self):
        """Retourne la feuille de style de l'application."""
        return """
            QMainWindow { background-color: #ffffff; }
            
            QWidget { 
                background-color: #ffffff;
                font-family: 'Segoe UI', 'Arial', sans-serif;
                color: #2c3e50;
            }
            
            QGroupBox {
                font-weight: bold;
                color: #2c3e50;
                border: 1px solid #dcdfe4;
                border-radius: 8px;
                margin-top: 12px;
                padding: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                background-color: #ffffff;
                color: #2c3e50;
            }
            
            QCheckBox {
                color: #2c3e50;
                spacing: 8px;
                font-size: 10pt;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            
            QComboBox {
                padding: 6px 10px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: #ffffff;
                color: #2c3e50;
            }
            QComboBox:hover { border-color: #3498db; }
            QComboBox::drop-down {
                border: none;
                width: 25px;
            }
            
            QPlainTextEdit {
                border: 1px solid #bdc3c7;
                border-radius: 6px;
                padding: 10px;
                background-color: #f8f9fa;
                color: #2c3e50;
                font-size: 10pt;
                font-family: 'Consolas', 'Courier New', monospace;
            }
            
            QProgressBar {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                text-align: center;
                background-color: #ecf0f1;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 3px;
            }
            
            QPushButton#PrimaryButton {
                background-color: #34495e;
                color: white;
                border-radius: 6px;
                padding: 10px 15px;
                border: none;
                font-weight: bold;
            }
            QPushButton#PrimaryButton:hover { background-color: #2c3e50; }
            
            QPushButton#AccentButton {
                background-color: #3498db;
                color: white;
                border-radius: 8px;
                font-weight: bold;
                font-size: 12pt;
                border: none;
                padding: 12px 25px;
            }
            QPushButton#AccentButton:hover { background-color: #2980b9; }
            QPushButton#AccentButton:disabled { background-color: #bdc3c7; }
            
            QPushButton#SecondaryButton {
                background-color: #ecf0f1;
                color: #2c3e50;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 6px 15px;
            }
            QPushButton#SecondaryButton:hover { background-color: #d5dbdb; }
            
            QPushButton#AIButton {
                background-color: #9b59b6;
                color: white;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
                border: none;
            }
            QPushButton#AIButton:hover { background-color: #8e44ad; }
            QPushButton#AIButton:disabled { background-color: #bdc3c7; }
            
            QStatusBar {
                background-color: #f0f0f0;
                color: #2c3e50;
                border-top: 1px solid #dcdfe4;
            }
            
            QLabel {
                color: #2c3e50;
            }
        """


# =============================================================================
#                               POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    # Afficher les infos de configuration au démarrage
    print("=" * 60)
    print("  OCR Intelligent + Google Gemini AI")
    print("=" * 60)
    print(f"  • Tesseract: {pytesseract.pytesseract.tesseract_cmd}")
    print(f"  • Gemini API: {'Disponible' if GEMINI_AVAILABLE else 'Non installé'}")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key and api_key != "votre_cle_api_google_ici":
        print("  • Clé API: Configurée ✓")
    else:
        print("  • Clé API: Non configurée (créez un fichier .env)")
    print("=" * 60)
    
    app = QApplication(sys.argv)
    window = ImageProcessorInterface()
    window.show()
    sys.exit(app.exec())
