# CQuence Documentation

**Advanced Gene Sequence Analysis Tool**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
   - [Prerequisites](#prerequisites)
   - [Windows Installation](#windows-installation)
   - [Linux Installation](#linux-installation)
   - [GPU Support](#gpu-support)
4. [Getting Started](#getting-started)
   - [Interactive Mode](#interactive-mode)
   - [Command Line Interface](#command-line-interface)
5. [Basic Usage](#basic-usage)
   - [Training a Model](#training-a-model)
   - [Analyzing Sequences](#analyzing-sequences)
   - [API Server](#api-server)
6. [Advanced Usage](#advanced-usage)
   - [Customizing Model Architecture](#customizing-model-architecture)
   - [Optimizing for Large Datasets](#optimizing-for-large-datasets)
   - [Performance Tuning](#performance-tuning)
   - [Custom Sequence Augmentation](#custom-sequence-augmentation)
7. [API Reference](#api-reference)
   - [RESTful API](#restful-api)
   - [Python API](#python-api)
8. [Use Cases](#use-cases)
   - [Basic Research](#basic-research)
   - [Clinical Applications](#clinical-applications)
   - [Large-Scale Analysis](#large-scale-analysis)
9. [Troubleshooting](#troubleshooting)
10. [Contributing](#contributing)
11. [License](#license)

---

## Introduction

CQuence is a high-performance gene sequence analysis framework designed for researchers and bioinformaticians. It leverages a hybrid transformer-CNN architecture optimized for both CPU and GPU execution to provide accurate and efficient analysis of DNA sequences.

The tool includes a versatile interface with interactive CLI, command-line arguments, and a RESTful API, making it suitable for a wide range of applications from simple sequence classification to complex large-scale genomic analyses.

---

## Features

- **Hybrid Transformer-CNN Architecture**: Combines the benefits of transformers for capturing long-range dependencies and CNNs for efficient feature extraction
- **Multiple Interfaces**: Interactive CLI, command-line interface, and RESTful API
- **GPU Acceleration**: Optimized for hardware acceleration with CUDA-compatible GPUs
- **Mixed Precision Training**: Support for mixed-precision training to improve performance
- **Sequence Data Handling**: Efficient processing of FASTA files and sequence data
- **Advanced Training Options**: Early stopping, learning rate scheduling, model checkpointing
- **Data Augmentation**: Built-in sequence augmentation techniques for improved model generalization
- **Visualization**: Rich output formatting with progress tracking

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional but recommended for large datasets)

### Windows Installation

1. **Install Python 3.8+**

   Download and install from [python.org](https://www.python.org/downloads/)

2. **Install PyTorch with CUDA support (if applicable)**

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

   For CPU-only:
   ```bash
   pip install torch torchvision torchaudio
   ```

3. **Install CQuence and dependencies**

   ```bash
   # Clone the repository
   git clone https://github.com/example/cquence.git
   cd cquence

   # Install dependencies
   pip install -r requirements.txt
   ```

   Alternatively, create a requirements.txt file with the following dependencies:
   ```
   numpy
   torch
   typer
   rich
   fastapi
   uvicorn
   pydantic
   ```

### Linux Installation

1. **Install Python and required packages**

   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install python3 python3-pip python3-venv

   # Create a virtual environment (recommended)
   python3 -m venv cquence-env
   source cquence-env/bin/activate
   ```

2. **Install PyTorch with CUDA support (if applicable)**

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

   For CPU-only:
   ```bash
   pip install torch torchvision torchaudio
   ```

3. **Install CQuence and dependencies**

   ```bash
   # Clone the repository
   git clone https://github.com/example/cquence.git
   cd cquence

   # Install dependencies
   pip install -r requirements.txt
   ```

### GPU Support

For GPU acceleration:

1. **Install NVIDIA drivers**
   - Windows: Download and install from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
   - Linux: `sudo apt install nvidia-driver-XXX` (replace XXX with the appropriate version)

2. **Install CUDA Toolkit**
   - Download and install CUDA Toolkit from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)

3. **Verify installation**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA device count: {torch.cuda.device_count()}")
   print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
   ```

---

## Getting Started

CQuence can be used in three main modes:

### Interactive Mode

Start the interactive CLI by running the script without arguments:

```bash
python cquence.py
```

This launches a user-friendly interface with menus for training, analyzing sequences, and configuring the tool.

### Command Line Interface

Use direct commands for automation and batch processing:

```bash
# Get help
python cquence.py --help

# Train a model
python cquence.py train --data-path ./data --epochs 20

# Analyze a sequence
python cquence.py analyze --sequence "ATGCGTACGATCGACTGACTGAC"

# Start API server
python cquence.py serve --port 8080
```

---

## Basic Usage

### Training a Model

1. **Prepare your data**

   Organize your sequence data in FASTA format with the following directory structure:
   ```
   data/
   ├── positive/
   │   ├── sample1.fa
   │   ├── sample2.fa
   │   └── ...
   └── negative/
       ├── sample1.fa
       ├── sample2.fa
       └── ...
   ```

2. **Train using the interactive CLI**

   ```bash
   python cquence.py
   ```
   
   Then select "Train Model" from the menu and follow the prompts.

3. **Train using command line**

   ```bash
   python cquence.py train --data-path ./data --model-type hybrid --epochs 20 --batch-size 64 --learning-rate 3e-4
   ```

### Analyzing Sequences

1. **Analyze a sequence using the interactive CLI**

   ```bash
   python cquence.py
   ```
   
   Select "Analyze Sequence" from the menu and follow the prompts to input a sequence or load from a file.

2. **Analyze using command line**

   ```bash
   python cquence.py analyze --sequence "ATGCGTACGATCGACTGACTGACATGC"
   ```

   Alternatively, analyze a sequence from a file:
   ```bash
   python cquence.py analyze --sequence "$(cat sequence.fa)"
   ```

### API Server

1. **Start the API server**

   ```bash
   python cquence.py serve --port 8000 --host 0.0.0.0
   ```

2. **Send requests to the API**

   ```bash
   curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"sequence":"ATGCGTACGATCGACTGACTGAC"}'
   ```

---

## Advanced Usage

### Customizing Model Architecture

Modify the model architecture by adjusting parameters:

```bash
python cquence.py train --data-path ./data --model-type hybrid --input-dim 256 --hidden-dim 512 --num-layers 6 --num-heads 8 --dropout 0.2
```

For more extensive customization, you can modify the code directly:

```python
# Import the module
from cquence import GeneTransformer, GeneEmbedding, Trainer

# Create a custom model configuration
custom_model = GeneTransformer(
    input_dim=192,
    hidden_dim=384,
    num_layers=5,
    num_heads=6,
    dropout=0.15
)

# Train with custom configuration
trainer = Trainer(
    model=custom_model,
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=2e-4,
    weight_decay=2e-5,
    precision="mixed"
)

# Train with custom parameters
trainer.train(
    epochs=30,
    device=torch.device("cuda"),
    early_stop_patience=7,
    save_best=True
)
```

### Optimizing for Large Datasets

For processing large-scale genomic data:

1. **Enable caching to speed up data loading**

   ```bash
   python cquence.py train --data-path ./large_data --cache-dir ./cache
   ```

2. **Adjust batch size and workers for your hardware**

   ```bash
   python cquence.py train --batch-size 128 --num-workers 8
   ```

3. **Use mixed precision training for GPU acceleration**

   ```bash
   python cquence.py train --precision mixed
   ```

4. **Process data in chunks for massive datasets**

   ```python
   # Process a very large dataset in chunks
   import os
   from pathlib import Path

   data_dir = Path("./massive_data")
   chunk_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
   
   for i, chunk_dir in enumerate(chunk_dirs):
       print(f"Processing chunk {i+1}/{len(chunk_dirs)}")
       os.system(f"python cquence.py train --data-path {chunk_dir} --save-dir ./models/chunk_{i} --epochs 5")
   
   # Merge models or use ensemble methods for final predictions
   ```

### Performance Tuning

Fine-tune the system for optimal performance:

1. **Profile memory usage**

   ```python
   import torch
   from cquence import GeneTransformer

   # Start with smaller dimensions for large sequences
   model = GeneTransformer(
       input_dim=64,
       hidden_dim=128,
       num_layers=2
   )
   
   # Test with different sequence lengths
   test_input = torch.randint(0, 5, (1, 2048))
   torch.cuda.reset_peak_memory_stats()
   _ = model(test_input.cuda())
   print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e6:.2f} MB")
   ```

2. **Adjust model size based on available hardware**

   For limited GPU memory:
   ```bash
   python cquence.py train --input-dim 64 --hidden-dim 128
   ```

   For high-end GPUs:
   ```bash
   python cquence.py train --input-dim 256 --hidden-dim 512
   ```

3. **Use gradient accumulation for larger effective batch sizes**

   Modify the Trainer class to implement gradient accumulation:
   ```python
   def _train_epoch(self, device: torch.device, accumulation_steps: int = 4) -> float:
       self.model.train()
       total_loss = 0.0
       
       for i, (inputs, labels) in enumerate(self.train_loader):
           inputs, labels = inputs.to(device), labels.to(device)
           
           # Forward pass with mixed precision if enabled
           with autocast(enabled=(self.precision == "mixed")):
               outputs = self.model(inputs)
               loss = self.criterion(outputs, labels) / accumulation_steps
           
           # Backward pass with gradient scaling if mixed precision
           if self.precision == "mixed":
               self.scaler.scale(loss).backward()
           else:
               loss.backward()
           
           if (i + 1) % accumulation_steps == 0 or (i + 1) == len(self.train_loader):
               if self.precision == "mixed":
                   self.scaler.step(self.optimizer)
                   self.scaler.update()
               else:
                   self.optimizer.step()
               
               self.optimizer.zero_grad()
           
           total_loss += loss.item() * accumulation_steps
       
       return total_loss / len(self.train_loader)
   ```

### Custom Sequence Augmentation

Implement custom sequence augmentation for improved model generalization:

```python
from cquence import SequenceAugmenter, SequenceNormalizer

class AdvancedAugmenter(SequenceAugmenter):
    @staticmethod
    def motif_insertion(sequence: str, motifs: List[str] = None) -> str:
        """Insert common genetic motifs at random positions"""
        if motifs is None:
            motifs = ["TATA", "CAAT", "GAGA", "TTGACA"]
        
        chars = list(sequence)
        if len(chars) > 20:  # Only insert if sequence is long enough
            insert_pos = random.randint(5, len(chars) - 10)
            motif = random.choice(motifs)
            chars[insert_pos:insert_pos] = list(motif)
        return ''.join(chars)
    
    @staticmethod
    def apply_augmentations(sequence: str, prob: float = 0.5) -> str:
        """Apply multiple augmentations with a probability"""
        augmenters = [
            SequenceAugmenter.random_mutation,
            SequenceAugmenter.reverse_complement,
            SequenceAugmenter.random_insert,
            AdvancedAugmenter.motif_insertion
        ]
        
        # Apply each augmentation with probability
        for augmenter in augmenters:
            if random.random() < prob:
                sequence = augmenter(sequence)
        
        return sequence

# Use in training
for seq in sequences:
    augmented_seq = AdvancedAugmenter.apply_augmentations(seq)
    # Process and train with augmented sequence
```

---

## API Reference

### RESTful API

The CQuence API provides endpoints for sequence analysis:

#### Endpoints

1. **GET /** - Check API status

   **Response:**
   ```json
   {
     "message": "Welcome to CQuence API", 
     "status": "operational"
   }
   ```

2. **POST /analyze** - Analyze a DNA sequence

   **Request:**
   ```json
   {
     "sequence": "ATGCGTACGATCGA"
   }
   ```

   **Response:**
   ```json
   {
     "prediction": 1,
     "confidence": 95.23,
     "processed_length": 14,
     "features": {
       "gc_content": 50.0,
       "a_content": 28.57,
       "t_content": 21.43
     }
   }
   ```

#### API Client Examples

**Python:**
```python
import requests

url = "http://localhost:8000/analyze"
payload = {"sequence": "ATGCGTACGATCGACTGACTGAC"}
response = requests.post(url, json=payload)
print(response.json())
```

**JavaScript:**
```javascript
fetch('http://localhost:8000/analyze', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    sequence: 'ATGCGTACGATCGACTGACTGAC'
  }),
})
.then(response => response.json())
.then(data => console.log(data));
```

### Python API

Import and use CQuence components directly in your Python code:

```python
from cquence import (
    GeneTransformer, 
    GeneDataset, 
    Trainer, 
    SequenceNormalizer
)
import torch
from torch.utils.data import DataLoader

# Initialize model
model = GeneTransformer(
    input_dim=128,
    hidden_dim=256,
    num_layers=4,
    dropout=0.1
)

# Load custom dataset
dataset = GeneDataset(
    data_dir="./my_data",
    max_seq_length=1024,
    cache_dir="./cache"
)

# Split data
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4
)

# Initialize trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=3e-4,
    precision="mixed" if torch.cuda.is_available() else "float32",
    save_dir="./my_models"
)

# Train model
training_stats = trainer.train(
    epochs=20,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    early_stop_patience=5,
    save_best=True
)

# Analyze a sequence
def analyze_sequence(seq_str):
    model.eval()
    normalized_seq = SequenceNormalizer.normalize(seq_str)
    encoded_seq = SequenceNormalizer.encode(normalized_seq)
    
    # Pad or truncate
    if len(encoded_seq) > 1024:
        encoded_seq = encoded_seq[:1024]
    else:
        encoded_seq = encoded_seq + [4] * (1024 - len(encoded_seq))
    
    seq_tensor = torch.tensor([encoded_seq], dtype=torch.long).to(device)
    
    with torch.no_grad():
        probs = model.predict_proba(seq_tensor)
    
    prediction = probs.argmax(dim=1).item()
    confidence = probs[0, prediction].item() * 100
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "sequence_length": len(normalized_seq)
    }

result = analyze_sequence("ATGCAGTACGTACGT")
print(result)
```

---

## Use Cases

### Basic Research

1. **Gene Classification**

   Classify genes based on their sequence properties:
   
   ```bash
   # Train a model to classify gene types
   python cquence.py train --data-path ./gene_types_data
   
   # Analyze unknown genes
   python cquence.py analyze --sequence "$(cat unknown_gene.fa)"
   ```

2. **Promoter Prediction**

   Identify potential promoter regions in DNA sequences:
   
   ```bash
   # Organize data with promoters as positive samples
   # Train a model
   python cquence.py train --data-path ./promoter_data
   
   # Batch analyze potential promoters
   for file in ./sequences/*.fa; do
     echo "Analyzing $file"
     python cquence.py analyze --sequence "$(cat $file)" --model-path ./models/promoter_model.pt
   done > promoter_predictions.txt
   ```

### Clinical Applications

1. **Pathogen Detection**

   Train models to identify pathogenic sequences:

   ```bash
   # Train with data from known pathogens
   python cquence.py train --data-path ./pathogen_data --model-type transformer --epochs 30
   
   # Deploy API for clinical testing
   python cquence.py serve --port 8080 --model-path ./models/pathogen_model.pt
   ```

2. **Mutation Analysis**

   Analyze genetic mutations and their potential effects:

   ```python
   from cquence import SequenceNormalizer, GeneTransformer, Trainer
   import torch
   
   # Load reference gene
   with open("reference_gene.fa", "r") as f:
     reference = f.read().replace(">.*\n", "").replace("\n", "")
   
   # Load model
   model = GeneTransformer()
   model, _ = Trainer.load_model(model, "mutation_model.pt")
   model.eval()
   
   # Generate mutations
   mutations = []
   for i in range(len(reference)):
     for base in "ATCG":
       if reference[i] != base:
         mutated = reference[:i] + base + reference[i+1:]
         mutations.append((i, reference[i], base, mutated))
   
   # Analyze mutations
   results = []
   for pos, orig, mut, seq in mutations:
     # Encode and predict
     encoded = SequenceNormalizer.encode(seq)
     tensor = torch.tensor([encoded], dtype=torch.long)
     with torch.no_grad():
       prob = model.predict_proba(tensor)
     
     pred = prob.argmax(dim=1).item()
     conf = prob[0, pred].item() * 100
     results.append((pos, f"{orig}>{mut}", pred, conf))
   
   # Sort by impact (confidence change)
   results.sort(key=lambda x: x[3], reverse=True)
   ```

### Large-Scale Analysis

1. **Genome-Wide Analysis**

   Process full genomes efficiently:

   ```python
   from cquence import SequenceNormalizer, GeneTransformer, Trainer
   import torch
   
   # Load model
   model = GeneTransformer(
       input_dim=128,
       hidden_dim=256,
       num_layers=4
   )
   model, _ = Trainer.load_model(model, "best_model.pt")
   model.eval()
   
   # Function to analyze in sliding window
   def analyze_genome(genome_path, window_size=1024, step_size=512):
       # Read genome
       with open(genome_path, "r") as f:
           # Skip header
           line = f.readline()
           while line.startswith(">"):
               line = f.readline()
           
           genome = ""
           while line:
               genome += line.strip()
               line = f.readline()
       
       results = []
       
       # Sliding window analysis
       for i in range(0, len(genome) - window_size, step_size):
           window = genome[i:i+window_size]
           normalized = SequenceNormalizer.normalize(window)
           encoded = SequenceNormalizer.encode(normalized)
           
           # Skip windows with too many unknown bases
           if encoded.count(4) > window_size * 0.1:
               continue
           
           tensor = torch.tensor([encoded], dtype=torch.long).to("cuda")
           
           with torch.no_grad():
               probs = model.predict_proba(tensor)
           
           pred = probs.argmax(dim=1).item()
           conf = probs[0, pred].item() * 100
           
           if conf > 90:  # Only keep high-confidence predictions
               results.append((i, i+window_size, pred, conf))
       
       return results
   
   # Analyze multiple genomes
   import os
   genome_dir = "./genomes"
   
   for genome_file in os.listdir(genome_dir):
       if genome_file.endswith(".fa"):
           print(f"Analyzing {genome_file}...")
           path = os.path.join(genome_dir, genome_file)
           results = analyze_genome(path)
           
           # Save results
           output_file = f"results_{genome_file.replace('.fa', '.txt')}"
           with open(output_file, "w") as f:
               f.write("Start\tEnd\tPrediction\tConfidence\n")
               for start, end, pred, conf in results:
                   f.write(f"{start}\t{end}\t{pred}\t{conf:.2f}\n")
   ```

2. **Distributed Processing**

   For extremely large datasets, implement distributed processing:

   ```python
   # Master script for distributed processing
   import subprocess
   import os
   from pathlib import Path
   
   # Split a large FASTA file into chunks
   def split_fasta(fasta_path, output_dir, chunk_size=1000):
       os.makedirs(output_dir, exist_ok=True)
       
       sequence_count = 0
       file_count = 0
       current_file = None
       
       with open(fasta_path, 'r') as f:
           for line in f:
               if line.startswith('>'):
                   if sequence_count % chunk_size == 0:
                       if current_file:
                           current_file.close()
                       file_count += 1
                       current_file = open(f"{output_dir}/chunk_{file_count}.fa", 'w')
                   sequence_count += 1
               
               current_file.write(line)
       
       if current_file:
           current_file.close()
       
       return file_count
   
   # Distribute processing
   def process_distributed(input_fasta, num_workers=4):
       # Split the input file
       chunks_dir = Path("./chunks")
       num_chunks = split_fasta(input_fasta, chunks_dir)
       
       # Prepare worker commands
       worker_processes = []
       for i in range(min(num_workers, num_chunks)):
           chunk_range = range(i+1, num_chunks+1, num_workers)
           chunk_list = ",".join(f"chunk_{j}.fa" for j in chunk_range)
           
           cmd = f"python worker.py --chunks {chunk_list} --chunks-dir {chunks_dir} --output-dir ./results"
           print(f"Starting worker with command: {cmd}")
           
           proc = subprocess.Popen(cmd, shell=True)
           worker_processes.append(proc)
       
       # Wait for all workers to complete
       for proc in worker_processes:
           proc.wait()
       
       # Merge results
       merge_results("./results")
   
   # Worker script (worker.py)
   """
   import argparse
   import os
   from pathlib import Path
   from cquence import GeneTransformer, SequenceNormalizer, Trainer
   import torch
   
   def worker_main():
       parser = argparse.ArgumentParser()
       parser.add_argument("--chunks", type=str, required=True)
       parser.add_argument("--chunks-dir", type=str, required=True)
       parser.add_argument("--output-dir", type=str, required=True)
       args = parser.parse_args()
       
       chunk_files = args.chunks.split(",")
       chunks_dir = Path(args.chunks_dir)
       output_dir = Path(args.output_dir)
       os.makedirs(output_dir, exist_ok=True)
       
       # Load model
       model = GeneTransformer()
       model, _ = Trainer.load_model(model, "best_model.pt")
       model.eval()
       
       # Process each chunk
       for chunk_file in chunk_files:
           chunk_path = chunks_dir / chunk_file
           output_path = output_dir / f"result_{chunk_file.replace('.fa', '.txt')}"
           
           with open(chunk_path, 'r') as f, open(output_path, 'w') as out:
               out.write("Sequence_ID\tPrediction\tConfidence\n")
               
               seq_id = ""
               sequence = ""
               
               for line in f:
                   if line.startswith('>'):
                       if sequence:
                           # Process previous sequence
                           normalized = SequenceNormalizer.normalize(sequence)
                           encoded = SequenceNormalizer.encode(normalized)
                           
                           # Truncate or pad
                           if len(encoded) > 1024:
                               encoded = encoded[:1024]
                           else:
                               encoded = encoded + [4] * (1024 - len(encoded))
                           
                           tensor = torch.tensor([encoded], dtype=torch.long)
                           with torch.no_grad():
                               probs = model.predict_proba(tensor)
                           
                           pred = probs.argmax(dim=1).item()
                           conf = probs[0, pred].item() * 100
                           
                           out.write(f"{seq_id}\t{pred}\t{conf:.2f}\n")
                       
                       seq_id = line.strip()[1:]
                       sequence = ""
                   else:
                       sequence += line.strip()
               
               # Process last sequence
               if sequence:
                   normalized = SequenceNormalizer.normalize(sequence)
                   encoded = SequenceNormalizer.encode(normalized)
                   
                   if len(encoded) > 1024:
                       encoded = encoded[:1024]
                   else:
                       encoded = encoded + [4] * (1024 - len(encoded))
                   
                   tensor = torch.tensor([encoded], dtype=torch.long)
                   with torch.no_grad():
                       probs = model.predict_proba(tensor)
                   
                   pred = probs.argmax(dim=1).item()
                   conf = probs[0, pred].item() * 100
                   
                   out.write(f"{seq_id}\t{pred}\t{conf:.2f}\n")
       
   if __name__ == "__main__":
       worker_main()
   """
   
   # Merge results from workers
   def merge_results(results_dir):
       results_dir = Path(results_dir)
       all_results = []
       
       for result_file in results_dir.glob("result_*.txt"):
           with open(result_file, 'r') as f:
               # Skip header
               next(f)
               for line in f:
                   all_results.append(line.strip().split('\t'))
       
       # Sort by confidence
       all_results.sort(key=lambda x: float(x[2]), reverse=True)
       
       # Write merged results
       with open("final_results.txt", 'w') as f:
           f.write("Sequence_ID\tPrediction\tConfidence\n")
           for result in all_results:
               f.write('\t'.join(result) + '\n')
   
   # Run distributed processing
   process_distributed("massive_genome.fa", num_workers=8)
   ```

---

## Troubleshooting

### Common Issues and Solutions

1. **CUDA Out of Memory**

   **Problem:** `RuntimeError: CUDA out of memory`
   
   **Solutions:**
   - Reduce batch size: `--batch-size 16`
   - Reduce model dimensions: `--input-dim 64 --hidden-dim 128`
   - Use gradient accumulation (see Advanced Usage section)
   - Process sequences in smaller chunks

2. **Slow Training**

   **Problem:** Training takes too long
   
   **Solutions:**
   - Enable mixed precision: `--precision mixed`
   - Increase batch size (if memory allows): `--batch-size 64`
   - Use GPU acceleration
   - Enable caching: `--cache-dir ./cache`
   - Adjust worker count: `--num-workers 4`

3. **Poor Model Performance**

   **Problem:** Low accuracy or high loss
   
   **Solutions:**
   - Increase training epochs: `--epochs 30`
   - Use data augmentation (see Custom Sequence Augmentation section)
   - Try different model architectures: `--model-type transformer`
   - Adjust learning rate: `--learning-rate 1e-4`
   - Check data quality
