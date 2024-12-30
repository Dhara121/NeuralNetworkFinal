# XOR Neural Network Project

## Overview
This project demonstrates a modular approach to building a neural network that solves the XOR problem. The network features 2 input nodes, a hidden layer, and 1 output node, showcasing how minimal architectures can address non-linear challenges.

---

## Key Components
- **Config Module:** Centralized configuration for hyperparameters, activation functions, loss functions, and file paths.
- **Datasets Module:** Contains the XOR dataset, simplifying data access and organization.
- **Preprocessing Module:** Handles data transformation and reshaping for training and evaluation.
- **Trained Models Module:** Manages saving and loading of trained models for reusability.
- **Pipeline Module:** Combines all components for streamlined training and evaluation.
- **Prediction Module:** Enables easy inference using trained models.
- **Dockerfile:** Provides a containerized environment for consistent and scalable deployments.

---

## Why Modular Design?
- **Flexibility:** Easily update or replace components without affecting the rest of the system.
- **Scalability:** Effortlessly extend functionality as the project grows.
- **Collaboration:** Enables simultaneous development on independent modules.
- **Ease of Maintenance:** Simplifies debugging and future enhancements.

---

## Setup
1. Install dependencies from `requirements.txt`, including Pandas, NumPy, Scikit-learn, Flask, and more.
2. Use the provided Dockerfile to create a containerized environment for seamless deployment.

---

## Future Enhancements
- Support for more complex non-linear problems.
- Enhanced visualization for training and predictions.
- Deployment via Flask for real-time inference.

---

## License
This project is licensed under the MIT License.

---

## Acknowledgments
Thanks to the open-source community for their contributions and inspiration!

