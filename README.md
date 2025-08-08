# Autonomous Racing Game with Deep Learning

## Overview

This project implements a 3D racing game where players can drive a car manually or let an AI agent navigate the track autonomously using a deep learning model. The game includes a graphical user interface (GUI) for selecting game modes, training the AI model, and monitoring training progress. The AI learns to mimic human driving behavior by training on player-collected data, using a Multi-Layer Perceptron (MLP) neural network.

The project consists of three main components:
- **Classic Game Mode** (`FINAL_GAME.py`): A manual driving mode where players control the car using WASD keys.
- **AI Game Mode** (`AI_Game.py`): An autonomous mode where the trained MLP model controls the car.
- **Training Script** (`train.py`): Trains the MLP model on collected driving data.
- **GUI Menu** (`Main.py`): A PyQt6-based interface for launching game modes and initiating training.

## Features

- **Game Environment**: A 3D racing track with a car model, collision detection, and a finish line, built using the Ursina game engine.
- **Camera Modes**: Multiple perspectives (Follow, Free with Car, Top-Down, Free, and AI-specific view with minimap and path prediction).
- **Sound Effects**: Engine idle, acceleration, and braking sounds for immersive gameplay.
- **Data Collection**: Captures car position, heading, and player inputs (WASD) during gameplay, saved to `data.csv`.
- **Neural Network**: An MLP trained with PyTorch to predict driving actions based on car state (x, z, heading).
- **GUI**: A PyQt6 interface for easy navigation, with real-time training progress updates via a progress bar and status messages.

## Installation

### Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure Resources**:
   - Place the `misc/` folder (containing sound files and assets) and `models/` folder (containing 3D models) in the project root.
   - Ensure `coords.json` is present for track coordinates.

5. **Run the Application**:
   ```bash
   python Main.py
   ```

## Usage

1. **Launch the GUI**:
   - Run `Main.py` to open the main menu.
   - Use the buttons to select:
     - **Play Classic Game**: Start the manual driving mode.
     - **Play AI Game**: Run the AI-controlled mode (requires a trained model).
     - **Train AI Model**: Train the MLP model using data from `data.csv`.
     - **Settings**: Placeholder for future settings (not implemented).
     - **Exit**: Close the application.

2. **Game Controls**:
   - **WASD**: Move the car (forward, left, backward, right).
   - **Space**: Toggle AI mode (in `AI_Game.py`).
   - **V**: Cycle through camera modes.
   - **R**: Restart the game after a win or collision.
   - **Arrow Keys**: Adjust camera position in Free with Car or Free modes.
   - **Scroll Wheel**: Zoom in/out in Top-Down mode.
   - **Q/E**: Move camera up/down in Free with Car or Free modes.

3. **Data Collection**:
   - In Classic or AI mode, driving data is collected when the `W` key is pressed.
   - Data is saved to `data.csv` upon reaching the finish line.

4. **Training**:
   - Click "Train AI Model" in the GUI to start training.
   - Monitor progress via the epoch information and progress bar.
   - The trained model is saved as `mlp_behavior_clone.pth`.

## Project Structure

```
your-repo-name/
├── Main.py               # GUI menu for launching game modes and training
├── FINAL_GAME.py         # Classic game mode (manual driving)
├── AI_Game.py            # AI-controlled game mode
├── train.py              # Script for training the MLP model
├── requirements.txt      # Python dependencies
├── data.csv              # Collected driving data (generated during gameplay)
├── mlp_behavior_clone.pth # Trained model weights (generated after training)
├── coords.json           # Track coordinates
├── misc/                 # Sound files and GUI assets
│   ├── bg.png
│   ├── icon.png
│   ├── splash.png
│   ├── racing_font.ttf
│   ├── engine_idle.wav
│   ├── engine_accelerate.wav
│   ├── brake_screech.wav
├── models/               # 3D models for the game
│   ├── map_flat.glb
│   ├── car2.glb
│   ├── wheels/
│   │   ├── Front_Wheel.glb
│   │   ├── Back_Wheel.glb
├── racing_game.log       # Log file for debugging
```

## Dependencies

See `requirements.txt` for a complete list. Key dependencies include:
- `ursina`: Game engine for 3D rendering and physics.
- `torch`: PyTorch for neural network training and inference.
- `pandas`: Data manipulation for training data.
- `numpy`: Numerical operations.
- `scikit-learn`: Data splitting for training.
- `PyQt6`: GUI framework.

## Training the AI Model

- **Data**: The model trains on `data.csv`, which contains car state (x, z, heading) and player actions (W, A, S, D).
- **Model**: A Multi-Layer Perceptron (MLP) with 3 input nodes, two hidden layers (32 nodes each with ReLU), and 4 output nodes.
- **Training**: Uses Adam optimizer and BCEWithLogitsLoss for multi-label classification. Training runs for 100 epochs (GUI) or 5000 epochs (standalone `train.py`).
- **Output**: Saves the trained model as `mlp_behavior_clone.pth`.

## AI Game Mode

- The AI mode loads `mlp_behavior_clone.pth` to control the car.
- Inputs: Car's x, z position, and heading.
- Outputs: Predicted actions (W, A, S, D) based on probabilities (threshold > 0.5).
- Features: Displays predicted actions and probabilities, shows a predicted path, and includes a minimap.

## Troubleshooting

- **Missing Dependencies**: Ensure all dependencies are installed (`pip install -r requirements.txt`).
- **Model Not Found**: Run training to generate `mlp_behavior_clone.pth` or ensure it’s in the project root.
- **Sound Issues**: Verify that `misc/` contains the required `.wav` files.
- **Track Issues**: Ensure `coords.json` is correctly formatted and present.
- **GUI Errors**: Check `racing_game.log` for detailed error messages.

## Contributing

Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit changes (`git commit -m 'Add YourFeature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.


## Acknowledgments

- Built with [Ursina Engine](https://www.ursinaengine.org/).
- Neural network powered by [PyTorch](https://pytorch.org/).
- GUI developed with [PyQt6](https://www.riverbankcomputing.com/software/pyqt/).
