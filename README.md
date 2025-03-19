# **Energy-Efficient Multi-Label Classification of arXiv Papers Using BERT Variants**

---

## **Overview**

This project focuses on **large-scale multi-label classification** of arXiv papers using **BERT**, **DistilBERT**, and **TinyBERT** models. The goal is to evaluate the **energy efficiency**, **accuracy**, and **inference performance** of these models, with a focus on **green software principles** and **sustainability**. The project includes:

- Training and fine-tuning of BERT, DistilBERT, and TinyBERT models.
- Quantization of models to reduce memory footprint and energy consumption.
- Inference testing on CPU using both **PyTorch** and **ONNX Runtime**.
- Comprehensive profiling of energy consumption and performance metrics.

The results demonstrate that **TinyBERT-Quantized** is the most energy-efficient model, offering significant **carbon emission reductions** with minimal accuracy trade-offs.

---

## **File Structure**

```
.
├── emissions/
│   └── emissions.csv                  # Energy consumption and carbon emissions data
├── visualisations/
│   ├── inference/
│   │   ├── batch_size_impact.png      # Impact of batch size on inference time
│   │   ├── inference_time_comparison.png # Comparison of inference times
│   │   └── memory_usage_comparison.png # Memory usage comparison
│   ├── quantized_training/
│   │   ├── accuracy_vs_emissions.png  # Accuracy vs emissions for quantized models
│   │   ├── loss_curves.png            # Training loss curves for quantized models
│   │   ├── model_efficiency.png       # Model efficiency comparison
│   │   └── performance_comparison.png # Performance metrics for quantized models
│   └── training/
│       ├── accuracy_vs_emissions.png  # Accuracy vs emissions for regular models
│       ├── loss_curves.png            # Training loss curves for regular models
│       ├── model_efficiency.png       # Model efficiency comparison
│       └── performance_metrics_scatter.png # Scatter plot of performance metrics
├── arxiv_data.csv                     # Dataset of arXiv papers
├── bert_models_benchmark.json         # Benchmark results for all models
├── inference.ipynb                    # Notebook for inference testing
├── large_scale_multi_text_classification.ipynb # Notebook for training regular models
├── LICENSE                            # Project license
├── quantized_large_scale_multi_text_classification.ipynb # Notebook for training quantized models
├── README.md                          # This file
└── requirements.txt                   # Python dependencies
```

---

## **Getting Started**

### **1. Prerequisites**

- Python 3.8 or higher.
- A GPU-enabled environment for training (e.g., Google Colab or a local machine with CUDA support).
- CPU for inference testing.

### **2. Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/sakshikumar19/GSoC-Green-Software-Evaluation-Task.git
   cd GSoC-Green-Software-Evaluation-Task
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Reproducing the Results**

### **1. Training the Models**

- **Regular Models**: Use the `large_scale_multi_text_classification.ipynb` notebook to train BERT, DistilBERT, and TinyBERT.
- **Quantized Models**: Use the `quantized_large_scale_multi_text_classification.ipynb` notebook to train quantized versions of the models.

**Note**: Training takes approximately **10-20 mins per model** on a GPU (there are 3 models which each having two variants - quantized and not quantized). The notebooks will generate checkpoints for the fine-tuned weights, which are not included in the repository due to size constraints. You can download the pre-trained weights from the following links:

- **ONNX Checkpoints**: [Download Here](https://drive.google.com/drive/folders/1BA51-cXEi9nnUFE_VmVuyO90FqjgyARu?usp=drive_link)
- **PyTorch Checkpoints**: [Download Here](https://drive.google.com/drive/folders/1l9BlzXiLB4lMWqvTAsS2PyaGDo_8iuig?usp=drive_link)

Place the downloaded checkpoints in the following folders:

- **ONNX Checkpoints**: `onnx_checkpoints/`
- **PyTorch Checkpoints**: `pt_checkpoints/`

### **2. Running Inference**

- Use the `inference.ipynb` notebook to perform inference on the trained models.
- The notebook supports both **PyTorch** and **ONNX Runtime** for inference.
- Ensure the checkpoints are placed in the correct folders (`onnx_checkpoints/` and `pt_checkpoints/`).

---

## **Key Results**

### **1. Training Results**

| Model                | Accuracy | F1 Score | Carbon Emissions (kg CO₂eq) | Efficiency (Accuracy/Emissions) |
| -------------------- | -------- | -------- | --------------------------- | ------------------------------- |
| BERT                 | 0.9941   | 0.7349   | 0.012601                    | 78.89                           |
| DistilBERT           | 0.9942   | 0.7379   | 0.007015                    | 141.73                          |
| TinyBERT             | 0.9891   | 0.5756   | 0.001393                    | 710.12                          |
| BERT-Quantized       | 0.9894   | 0.5846   | 0.009463                    | 104.55                          |
| DistilBERT-Quantized | 0.9894   | 0.5846   | 0.005485                    | 180.37                          |
| TinyBERT-Quantized   | 0.9943   | 0.7453   | 0.001938                    | 513.08                          |

**Recommendation**:

- **Most Accurate Model**: DistilBERT (non-quantized).
- **Most Efficient Model**: TinyBERT-Quantized.
- By choosing **TinyBERT-Quantized**, you can achieve an **80.14% reduction in emissions** with only a **0.51% drop in accuracy**.

### **2. Inference Results**

- **Average Inference Time**:
  - TinyBERT: **0.26 seconds**
  - DistilBERT: **6.35 seconds**
  - BERT: **12.60 seconds**
- **Quantization Impact**:
  - Quantization slightly increases inference time but reduces memory usage.
- **PyTorch vs ONNX Runtime**:
  - ONNX Runtime is slower than PyTorch but offers cross-platform compatibility.
- **Batch Size Impact**:
  - Larger batch sizes improve inference efficiency (e.g., **1.19x speedup** for TinyBERT with batch size 16).

---

## **Scientific Principles and Industry Patterns**

This project aligns with several **green software principles** and **industry best practices**:

1. **Energy Efficiency**: By using smaller, quantized models like **TinyBERT**, we reduce energy consumption during both training and inference.
2. **Carbon Awareness**: The use of **CodeCarbon** to profile emissions helps raise awareness of the environmental impact of AI.
3. **Hardware Efficiency**: Quantization reduces memory usage, making the models more efficient on hardware.
4. **Reproducibility**: The code is well-documented and modular, ensuring reproducibility and ease of use.

**C++ Inference**:
While this project uses Python-based ONNX Runtime for inference, **C++ inference** would have been even more efficient due to lower overhead. The ONNX weights generated in this project can be directly used for C++ inference, making it a viable option for future work.

---

## **Visualizations**

The project includes several visualizations to help interpret the results:

- **Training**: Loss curves, accuracy vs emissions, and model efficiency plots.
- **Inference**: Inference time comparison, memory usage comparison, and batch size impact plots.

These visualizations are stored in the `visualisations/` folder.

---

## **Conclusion**

This project demonstrates that **energy-efficient AI** is achievable without significant sacrifices in accuracy. By leveraging smaller, quantized models like **TinyBERT**, organizations like **CERN** can reduce the environmental impact of their AI workflows while maintaining high performance. The insights gained from this project can be applied to other tasks, such as **particle physics data analysis** or **scientific paper classification**, further contributing to sustainable research practices.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
