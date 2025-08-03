import sys
import subprocess
import threading
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStatusBar, QProgressBar, QSplashScreen
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QSize
from PyQt6.QtGui import QPixmap, QKeyEvent, QIcon, QFont, QFontDatabase
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("racing_game.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
RESOURCES_DIR = "misc"
DATA_FILE = "data.csv"
MODEL_FILE = "mlp_behavior_clone.pth"
BG_IMAGE = os.path.join(RESOURCES_DIR, "bg.png")
BUTTON_WIDTH = 220
BUTTON_HEIGHT = 60


class TrainingSignals(QObject):
    """Class to handle signals from training thread to UI"""
    epoch_update = pyqtSignal(str)
    training_complete = pyqtSignal(str, bool)
    progress_update = pyqtSignal(int)


class StyleSheet:
    """Centralized stylesheet definitions"""
    
    @staticmethod
    def main_window():
        return (
            "QMainWindow { background: qlineargradient(x1:0,y1:0,x2:1,y2:1,"
            "stop:0 #1e1e2e, stop:1 #2e2e4e); }"
        )
    
    @staticmethod
    def button(primary=True):
        base_color = "#3b82f6" if primary else "#4b5563"
        hover_color = "#60a5fa" if primary else "#6b7280"
        pressed_color = "#2563eb" if primary else "#374151"
        
        return (
            f"QPushButton {{ background-color: {base_color}; color: white;"
            " font-size: 16px; font-weight: bold; border-radius: 10px;"
            " padding: 10px; }"
            f"QPushButton:hover {{ background-color: {hover_color}; }}"
            f"QPushButton:pressed {{ background-color: {pressed_color}; }}"
            "QPushButton:disabled { background-color: #6b7280; color: #d1d5db; }"
        )
    
    @staticmethod
    def title_label():
        return (
            "QLabel { font-size: 42px; font-weight: bold; color: #ffffff; }"
        )
    
    @staticmethod
    def status_label():
        return "QLabel { font-size: 14px; color: #ffffff; }"
    
    @staticmethod
    def status_bar():
        return "color: #ffffff; background: #2e2e4e; font-size: 13px;"
    
    @staticmethod
    def progress_bar():
        return (
            "QProgressBar {"
            "   border: 1px solid #2e2e4e;"
            "   border-radius: 5px;"
            "   text-align: center;"
            "   color: white;"
            "   background-color: #1e1e2e;"
            "}"
            "QProgressBar::chunk {"
            "   background-color: #3b82f6;"
            "   border-radius: 5px;"
            "}"
        )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize attributes
        self.bg_label = None
        self.original_pixmap = None
        self.training_signals = TrainingSignals()
        
        # Connect signals
        self.training_signals.epoch_update.connect(self.update_epoch_info)
        self.training_signals.training_complete.connect(self.update_status)
        self.training_signals.progress_update.connect(self.update_progress)

        self.setWindowTitle("Racing Game Menu")
        self.setStyleSheet(StyleSheet.main_window())

        # Set window icon
        try:
            icon_path = os.path.join(RESOURCES_DIR, "icon.png")
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
        except Exception as e:
            logger.warning(f"Could not load window icon: {e}")

        # Load custom fonts if available
        try:
            font_path = os.path.join(RESOURCES_DIR, "racing_font.ttf")
            if os.path.exists(font_path):
                font_id = QFontDatabase.addApplicationFont(font_path)
                if font_id != -1:
                    font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
                    title_font = QFont(font_family, 42)
                else:
                    title_font = QFont("Arial Black", 42)
            else:
                title_font = QFont("Arial Black", 42)
        except Exception as e:
            logger.warning(f"Could not load custom font: {e}")
            title_font = QFont("Arial Black", 42)

        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout (vertical)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.setSpacing(25)
        main_layout.setContentsMargins(30, 40, 30, 40)

        # Load background image
        self._load_background_image()

        # Title label
        title = QLabel("RACING GAME")
        title.setFont(title_font)
        title.setStyleSheet(StyleSheet.title_label())
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("Ultimate Edition")
        subtitle.setStyleSheet("QLabel { font-size: 24px; color: #60a5fa; font-style: italic; }")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(subtitle)
        
        # Add some spacing
        main_layout.addSpacing(20)

        # Button container with horizontal layout for better organization
        button_container = QWidget()
        button_layout = QVBoxLayout(button_container)
        button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        button_layout.setSpacing(15)
        main_layout.addWidget(button_container)

        # Buttons
        self.classic_button = self.create_button("🏎️ Play Classic Game", self.play_classic_game, primary=True)
        button_layout.addWidget(self.classic_button)

        self.ai_button = self.create_button("🤖 Play AI Game", self.play_ai_game, primary=True)
        button_layout.addWidget(self.ai_button)

        self.train_button = self.create_button("🧠 Train AI Model", self.start_training, primary=True)
        button_layout.addWidget(self.train_button)
        
        # Add settings button
        self.settings_button = self.create_button("⚙️ Settings", self.show_settings, primary=False)
        button_layout.addWidget(self.settings_button)

        # Add horizontal layout for exit button to be separated
        exit_layout = QHBoxLayout()
        exit_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.exit_button = self.create_button("🚪 Exit", self.close, primary=False)
        exit_layout.addWidget(self.exit_button)
        button_layout.addLayout(exit_layout)

        # Training information section
        training_info = QWidget()
        training_layout = QVBoxLayout(training_info)
        
        # Epoch info label
        self.epoch_label = QLabel("")
        self.epoch_label.setStyleSheet(StyleSheet.status_label())
        self.epoch_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        training_layout.addWidget(self.epoch_label)
        
        # Progress bar for training
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(StyleSheet.progress_bar())
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(10)
        self.progress_bar.setTextVisible(False)
        training_layout.addWidget(self.progress_bar)
        
        main_layout.addWidget(training_info)
        training_info.hide()
        self.training_info_widget = training_info

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        self.status_bar.setStyleSheet(StyleSheet.status_bar())

        # Add version info
        version_label = QLabel("v1.2.0")
        version_label.setStyleSheet("color: #9ca3af; font-size: 12px;")
        self.status_bar.addWidget(version_label)

        # Training indicator
        self.training_active = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_training_indicator)

        # Create a more attractive splash screen if resources allow
        try:
            splash_pixmap = QPixmap(os.path.join(RESOURCES_DIR, "splash.png"))
            if not splash_pixmap.isNull():
                splash = QSplashScreen(splash_pixmap)
                splash.show()
                # Process application events to display splash immediately
                QApplication.processEvents()
                splash.finish(self)
            else:
                logger.warning("Splash screen image not found or invalid format.")
        except Exception as e:
            logger.warning(f"Could not load splash screen: {e}")

        # Show window after setup
        self.show()
        # Default to maximized instead of full screen for better usability
        self.showMaximized()

    def _load_background_image(self):
        """Load and set the background image"""
        try:
            pixmap = QPixmap(BG_IMAGE)
            if not pixmap.isNull():
                self.original_pixmap = pixmap
                self.bg_label = QLabel(self)
                scaled = self.original_pixmap.scaled(
                    self.size(),
                    Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.bg_label.setPixmap(scaled)
                self.bg_label.setGeometry(0, 0, self.width(), self.height())
                self.bg_label.lower()
            else:
                raise FileNotFoundError(f"Image not found or invalid format: {BG_IMAGE}")
        except Exception as e:
            logger.warning(f"Could not load background image: {e}")
            if self.bg_label:
                self.bg_label.hide()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.bg_label and self.original_pixmap:
            scaled = self.original_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation
            )
            self.bg_label.setPixmap(scaled)
            self.bg_label.setGeometry(0, 0, self.width(), self.height())

    def keyPressEvent(self, event: QKeyEvent):
        # Escape key toggles between maximized and normal
        if event.key() == Qt.Key.Key_Escape:
            if self.isFullScreen():
                self.showMaximized()
            else:
                self.close()
        # F11 for fullscreen toggle
        elif event.key() == Qt.Key.Key_F11:
            if self.isFullScreen():
                self.showMaximized()
            else:
                self.showFullScreen()

    def create_button(self, text, callback, primary=True):
        """Create a styled button with the given text and callback"""
        button = QPushButton(text)
        button.setFixedSize(BUTTON_WIDTH, BUTTON_HEIGHT)
        button.setStyleSheet(StyleSheet.button(primary))
        button.clicked.connect(callback)
        return button

    def play_classic_game(self):
        """Launch the classic game mode"""
        try:
            # Check if the game file exists
            game_file = 'FINAL_GAME.py'
            if not os.path.exists(game_file):
                self.show_error_message(f"Game file '{game_file}' not found")
                return
                
            subprocess.Popen([sys.executable, game_file])
            self.status_bar.showMessage("Started Classic Game")
            logger.info("Launched Classic Game")
        except Exception as e:
            error_msg = f"Error launching Classic Game: {e}"
            self.status_bar.showMessage(error_msg)
            logger.error(error_msg, exc_info=True)
            self.show_error_message(error_msg)

    def play_ai_game(self):
        """Launch the AI game mode"""
        try:
            # Check if the game file exists
            game_file = 'AI_Game.py'
            if not os.path.exists(game_file):
                self.show_error_message(f"Game file '{game_file}' not found")
                return
                
            # Check if model exists
            if not os.path.exists(MODEL_FILE):
                # Offer to train the model
                self.status_bar.showMessage("AI model not found. Starting training...")
                QTimer.singleShot(1500, self.start_training)
                return
                
            subprocess.Popen([sys.executable, game_file])
            self.status_bar.showMessage("Started AI Game")
            logger.info("Launched AI Game")
        except Exception as e:
            error_msg = f"Error launching AI Game: {e}"
            self.status_bar.showMessage(error_msg)
            logger.error(error_msg, exc_info=True)
            self.show_error_message(error_msg)

    def start_training(self):
        """Start the AI model training process"""
        if self.training_active:
            return
            
        # Check if data file exists
        if not os.path.exists(DATA_FILE):
            self.show_error_message(f"Training data file '{DATA_FILE}' not found")
            return
            
        self.training_active = True
        self.train_button.setEnabled(False)
        self.training_info_widget.show()
        self.progress_bar.setValue(0)
        self.status_bar.showMessage("Training in progress")
        self.timer.start(500)
        
        # Start training in a separate thread to avoid blocking UI
        threading.Thread(target=self.run_training, daemon=True).start()
        logger.info("Started model training")

    def run_training(self):
        """Run the training process with UI updates"""
        try:
            import pandas as pd
            import numpy as np
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from sklearn.model_selection import train_test_split

            # 1. Load and preprocess data
            self.training_signals.epoch_update.emit("Loading data...")
            data = pd.read_csv(DATA_FILE)
            
            # Features: x, z, heading
            X = data[['x', 'z', 'heading']].values.astype(np.float32)
            
            # Actions: W, A, S, D (one-hot/multi-hot)
            y = data[['w', 'a', 's', 'd']].values.astype(np.float32)
            
            # Split into train and test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Convert to torch tensors
            torch_X_train = torch.tensor(X_train)
            torch_y_train = torch.tensor(y_train)
            torch_X_test = torch.tensor(X_test)
            torch_y_test = torch.tensor(y_test)
            
            # 2. Define the model
            class MLP(nn.Module):
                def __init__(self, input_dim=3, output_dim=4):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(input_dim, 32),
                        nn.ReLU(),
                        nn.Linear(32, 32),
                        nn.ReLU(),
                        nn.Linear(32, output_dim)
                    )
                def forward(self, x):
                    return self.net(x)

            model = MLP()
            
            # 3. Training setup
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            batch_size = 64
            epochs = 100  # Reduced from 5000 for UI responsiveness
            
            # 4. Training loop
            for epoch in range(epochs):
                permutation = torch.randperm(torch_X_train.size()[0])
                epoch_loss = 0
                
                for i in range(0, torch_X_train.size()[0], batch_size):
                    indices = permutation[i:i+batch_size]
                    batch_x, batch_y = torch_X_train[indices], torch_y_train[indices]

                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    
                    # Update progress
                    progress = int((i + batch_size) / torch_X_train.size()[0] * 100)
                    self.training_signals.progress_update.emit(progress)
                
                # Update epoch info
                self.training_signals.epoch_update.emit(
                    f"Epoch {epoch+1}/{epochs}\n"
                    f"Loss: {epoch_loss:.4f}"
                )
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
                
            # 5. Evaluation
            with torch.no_grad():
                outputs = model(torch_X_test)
                predictions = torch.sigmoid(outputs) > 0.5
                accuracy = (predictions == torch_y_test.bool()).float().mean()
                logger.info(f"Test accuracy: {accuracy.item():.4f}")
                
            # 6. Save the model
            torch.save(model.state_dict(), MODEL_FILE)
            logger.info(f"Model saved to {MODEL_FILE}")
            
            self.training_signals.training_complete.emit(
                f"Training complete! Accuracy: {accuracy.item():.4f}", True
            )
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            self.training_signals.training_complete.emit(f"Training failed: {str(e)}", False)

    def show_settings(self):
        """Show settings dialog (placeholder for future implementation)"""
        self.status_bar.showMessage("Settings feature coming soon!", 3000)

    def show_error_message(self, message: str):
        """Display error message in status bar with red styling"""
        self.status_bar.setStyleSheet("color: #ff5555; background: #2e2e4e; font-size: 13px;")
        self.status_bar.showMessage(message, 5000)
        QTimer.singleShot(5000, lambda: self.status_bar.setStyleSheet(StyleSheet.status_bar()))

    def update_training_indicator(self):
        """Update the training indicator animation in status bar"""
        if not self.training_active:
            self.timer.stop()
            return
        dots = '.' * ((self.status_bar.currentMessage().count('.') % 4) + 1)
        self.status_bar.showMessage(f"Training{dots}")

    def update_epoch_info(self, text):
        """Update the epoch information label during training"""
        self.epoch_label.setText(text)
        
    def update_progress(self, value):
        """Update the progress bar during training"""
        self.progress_bar.setValue(value)
        
    def update_status(self, message, success=True):
        """Update status after training completion"""
        self.training_active = False
        self.train_button.setEnabled(True)
        self.timer.stop()
        
        # Don't hide training info immediately so user can see final results
        QTimer.singleShot(5000, lambda: self.training_info_widget.hide())
        
        self.status_bar.showMessage(message)
        if success:
            self.status_bar.setStyleSheet("color: #55ff55; background: #2e2e4e; font-size: 13px;")
        else:
            self.status_bar.setStyleSheet("color: #ff5555; background: #2e2e4e; font-size: 13px;")
        QTimer.singleShot(5000, lambda: self.status_bar.setStyleSheet(StyleSheet.status_bar()))
        logger.info(f"Training completed: {message}")


def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
        
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
        
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas")
        
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
        
    try:
        import sklearn
    except ImportError:
        missing_deps.append("scikit-learn")
        
    if missing_deps:
        error_msg = "Missing dependencies: " + ", ".join(missing_deps)
        logger.error(error_msg)
        print(error_msg)
        print("Please install them using: pip install " + " ".join(missing_deps))
        return False
        
    return True


# Main Application
if __name__ == '__main__':
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
        
    # Create required directories if they don't exist
    os.makedirs(RESOURCES_DIR, exist_ok=True)
    
    # Create application
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Create and show the main window
    window = MainWindow()
    
    sys.exit(app.exec())