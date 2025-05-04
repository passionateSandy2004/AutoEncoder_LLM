# Autoencoder-based Language Model (AE-LLM)

An innovative language model that combines autoencoder architecture with transition networks for question-answering tasks. This model has been trained and fine-tuned on Wikipedia data, demonstrating promising results in generating accurate responses to questions from its training domain.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch
- CUDA-capable GPU (recommended)
- Basic dependencies (see requirements.txt)

### Quick Inference
1. Clone this repository
2. Navigate to the project directory
3. Run the inference script:
```bash
python QuickInference.py
```

The model will load the pre-trained weights from `saved_model2FineTune/` and be ready to answer questions.

## 🧠 Model Architecture

### Core Components

1. **Autoencoder (Seq2SeqAE)**
   - Encoder: Converts input sequences into latent representations
   - Decoder: Reconstructs sequences from latent representations
   - Latent Space: Captures semantic meaning of input sequences

2. **Transition Network (Thinking Net)**
   - Sparse Mixture of Experts (MoE): Handles complex transformations
   - Key-Value Memory: Stores and retrieves relevant information
   - Gated MLP Blocks: Processes and refines representations
   - Hyper Adapter: Adapts latent space to vocabulary space

### Training Process

1. **Initial Training**
   - Autoencoder learns to encode and decode sequences
   - Transition network learns to map between question and answer spaces
   - Training typically requires 3000+ epochs

2. **Fine-tuning**
   - Models are fine-tuned on specific domains
   - Current model fine-tuned to:
     - Autoencoder loss: ~0.14
     - Transition network loss: <0.01
   - Note: Catastrophic forgetting is a known issue

## 📊 Performance

- Trained on first 500 Wikipedia QA pairs
- Training time: ~18 hours on RTX 3050 (4GB)
- Demonstrates high accuracy for in-domain questions
- Current limitations:
  - Limited to training domain knowledge
  - Requires careful fine-tuning to prevent forgetting

## 🛠️ Usage

### Training
```bash
# Train autoencoder
python main.py --train-ae --data data.txt --save-dir saved_models --ae-epochs 3000

# Train transition network
python main.py --train-trans --data data.txt --save-dir saved_models --trans-epochs 3000
```

### Fine-tuning
```bash
python main.py --finetune --pretrained-dir saved_models --data new_data.txt --save-dir fine_tuned_models --ae-epochs 500 --trans-epochs 500
```

### Inference
```python
from QuickInference import answer_question

response = answer_question("Your question here")
print(response)
```

## 🔮 Future Development

1. **Adaptation Learning**
   - Implementation of LoRA (Low-Rank Adaptation with Lite Routers)
   - Enables efficient fine-tuning without catastrophic forgetting
   - Preserves original model knowledge while learning new domains

2. **Reinforcement Learning Integration**
   - Planned integration with RL for improved response quality
   - Will enable learning from user feedback
   - Expected to enhance model's adaptability

3. **Performance Improvements**
   - Support for larger training datasets
   - Optimized for high-end GPUs
   - Enhanced memory management

## 📚 Technical Details

### Model Specifications
- Vocabulary Size: ~1800 tokens
- Embedding Dimension: 64
- Hidden Dimension: 32
- Latent Dimension: 124
- Batch Size: 16

### Training Parameters
- Learning Rate (AE): 5e-4
- Learning Rate (Transition): 1e-4
- Teacher Forcing: 0.5
- Gradient Clipping: 1.0

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Wikipedia dataset from Hugging Face
- PyTorch community for the deep learning framework
- Contributors and maintainers of this project

## 📞 Contact

For questions or suggestions, please open an issue in the repository. 
