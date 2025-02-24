# High-Speed Rail Data with Federated Learning

## Overview
This project implements **Federated Learning** using high-speed rail data. The dataset utilized for this project is **V22**. The main objective is to enhance data privacy while training machine learning models across multiple decentralized clients without sharing raw data.

## Features
- Federated Learning implementation tailored for high-speed rail data.
- Utilizes the **V22 dataset** for real-world applications.
- Supports decentralized model training.
- Optimized for reducing packet loss during handovers in high-speed rail networks.

## Dataset
**V22 Dataset** is used for training and evaluation. It contains various network-related metrics collected from high-speed rail environments. The dataset is preprocessed before being fed into the federated learning framework.

## Installation
To set up the environment, follow these steps:

```bash
# Clone the repository
git clone https://github.com/yourusername/yourrepository.git
cd yourrepository

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Usage
Run the main federated learning script:

```bash
python train.py
```

Modify configurations in `config.yaml` to adjust training parameters such as learning rate, number of rounds, and client settings.

## Project Structure
```
├── data/                   # Folder containing dataset
├── models/                 # Pretrained and trained models
├── src/                    # Source code
│   ├── train.py            # Main federated learning training script
│   ├── federated_utils.py  # Helper functions for federated learning
│   ├── preprocessing.py    # Data preprocessing scripts
│   ├── config.yaml         # Configuration settings
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
```

## Configuration
Edit `config.yaml` to customize the learning process:
```yaml
learning_rate: 0.01
num_rounds: 50
num_clients: 10
batch_size: 32
```

## Results
The trained model achieves improved handover prediction in high-speed rail networks, reducing packet loss and enhancing connectivity.

## Future Improvements
- Integrate additional datasets for better generalization.
- Optimize communication efficiency in federated learning.
- Implement differential privacy techniques for enhanced security.

## Contributions
Feel free to submit issues and pull requests! Contributions are welcome.

## License
This project is licensed under the MIT License. See `LICENSE` for details.

