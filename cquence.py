# cquence.py
"""
CQuence: Advanced Gene Sequence Analysis Tool
---------------------------------------------
A high-performance gene sequence analysis framework with hybrid transformer-CNN
architecture optimized for both CPU and GPU execution.
"""

import os
import sys
import platform
import logging
import json
import random
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Union, Tuple, Any
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import typer
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
from rich.logging import RichHandler
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import multiprocessing as mp

# ---------- Configuration and Setup ----------
class DNABase(str, Enum):
    """Enumeration of DNA bases for type safety"""
    A = "A"
    T = "T"
    C = "C"
    G = "G"
    N = "N"  # For unknown bases

@dataclass
class CQuenceConfig:
    """Configuration class for CQuence"""
    model_type: str = "transformer"  # transformer, cnn, hybrid
    input_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.1
    learning_rate: float = 3e-4
    batch_size: int = 32
    max_seq_length: int = 1024
    num_workers: int = 0  # Will be set automatically based on system
    precision: str = "mixed"  # float32, float16, mixed
    save_dir: str = "./models"
    log_dir: str = "./logs"
    data_dir: str = "./data"
    cache_dir: str = "./cache"
    device: str = ""  # Will be set automatically

# System detection and optimization
OS_TYPE = platform.system()
CPU_COUNT = mp.cpu_count()
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")

# Initialize console with rich styling
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("cquence")

# Global configuration
CONFIG = CQuenceConfig()
CONFIG.device = str(DEVICE)
CONFIG.num_workers = min(4, CPU_COUNT - 1) if CPU_COUNT > 1 else 0

# Create necessary directories
os.makedirs(CONFIG.save_dir, exist_ok=True)
os.makedirs(CONFIG.log_dir, exist_ok=True)
os.makedirs(CONFIG.data_dir, exist_ok=True)
os.makedirs(CONFIG.cache_dir, exist_ok=True)

console.print(Panel(f"""
[bold blue]CQuence Gene Analysis Tool[/bold blue]
[green]System:[/green] {OS_TYPE}
[green]CPU Cores:[/green] {CPU_COUNT}
[green]Device:[/green] {DEVICE}
[green]Workers:[/green] {CONFIG.num_workers}
""", title="System Information"))

# ---------- Core Models ----------
class GeneEmbedding(nn.Module):
    """Enhanced embedding layer for DNA sequences"""
    def __init__(self, input_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        # 5 for A, T, C, G, and N (unknown)
        self.embedding = nn.Embedding(5, input_dim)
        self.position_encoding = self._create_position_encoding(2048, input_dim)
        self.dropout = nn.Dropout(dropout)
    
    def _create_position_encoding(self, max_length: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal position encoding"""
        position = torch.arange(0, max_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        
        pos_encoding = torch.zeros(max_length, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding.unsqueeze(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        embeddings = self.embedding(x)
        
        # Add positional encoding
        position_encoding = self.position_encoding[:, :seq_len, :].to(x.device)
        embeddings = embeddings + position_encoding
        
        return self.dropout(embeddings)

class ResidualBlock(nn.Module):
    """Residual block for the CNN part of the model"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class GeneTransformer(nn.Module):
    """Advanced Hybrid Transformer-CNN for gene sequence analysis"""
    def __init__(
        self, 
        input_dim: int = 128, 
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_classes: int = 2
    ):
        super().__init__()
        self.embedding = GeneEmbedding(input_dim, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # CNN feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.AdaptiveMaxPool1d(1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights for better convergence"""
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                nn.init.kaiming_normal_(p, mode='fan_out', nonlinearity='relu')
            elif 'bias' in name:
                nn.init.constant_(p, 0)
    
    @autocast()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len]
        
        # Get embeddings with positional encoding
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # Apply transformer to capture long-range dependencies
        x = self.transformer(x)  # [batch_size, seq_len, embedding_dim]
        
        # CNN processes sequence features
        x = x.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len]
        x = self.conv_layers(x).squeeze(-1)  # [batch_size, hidden_dim]
        
        # Classification
        return self.classifier(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probabilities instead of logits"""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

class AttentionPooling(nn.Module):
    """Attention pooling layer for sequence data"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, hidden_dim]
        attention_weights = F.softmax(self.attention(x).squeeze(-1), dim=1)
        return torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)

# ---------- Data Handling ----------
class SequenceNormalizer:
    """Utility for normalizing and encoding DNA sequences"""
    @staticmethod
    def normalize(sequence: str) -> str:
        """Normalize a DNA sequence by converting to uppercase and handling invalid bases"""
        sequence = sequence.upper()
        valid_bases = {'A', 'T', 'C', 'G'}
        return ''.join(base if base in valid_bases else 'N' for base in sequence)
    
    @staticmethod
    def encode(sequence: str) -> List[int]:
        """Encode a DNA sequence to numeric values"""
        mapping = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}
        return [mapping.get(base, 4) for base in sequence]
    
    @staticmethod
    def decode(encoded: List[int]) -> str:
        """Decode numeric values back to a DNA sequence"""
        mapping = {0: 'A', 1: 'T', 2: 'C', 3: 'G', 4: 'N'}
        return ''.join(mapping.get(idx, 'N') for idx in encoded)

class GeneDataset(Dataset):
    """Dataset for loading and processing DNA sequences"""
    def __init__(
        self, 
        data_dir: str,
        max_seq_length: int = 1024,
        transform=None,
        cache_dir: Optional[str] = None
    ):
        self.data_dir = Path(data_dir)
        self.max_seq_length = max_seq_length
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        self.sequences = []
        self.labels = []
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load sequence data from files or cache"""
        if self.cache_dir and (self.cache_dir / "dataset_cache.pt").exists():
            logger.info("Loading dataset from cache...")
            cache = torch.load(self.cache_dir / "dataset_cache.pt")
            self.sequences = cache["sequences"]
            self.labels = cache["labels"]
            return
        
        logger.info("Loading dataset from files...")
        
        # Load positive samples (class 1)
        pos_dir = self.data_dir / "positive"
        if pos_dir.exists():
            for file_path in pos_dir.glob("*.fa"):
                self._process_fasta_file(file_path, 1)
        
        # Load negative samples (class 0)
        neg_dir = self.data_dir / "negative"
        if neg_dir.exists():
            for file_path in neg_dir.glob("*.fa"):
                self._process_fasta_file(file_path, 0)
        
        # If no data loaded, create dummy data for testing
        if not self.sequences:
            logger.warning("No sequence data found. Creating dummy data for testing.")
            self._create_dummy_data()
        
        # Cache the dataset if cache directory is provided
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            torch.save({
                "sequences": self.sequences,
                "labels": self.labels
            }, self.cache_dir / "dataset_cache.pt")
    
    def _process_fasta_file(self, file_path: Path, label: int):
        """Process a FASTA file and extract sequences"""
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        current_seq = ""
        for line in lines:
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence if exists
                if current_seq:
                    normalized_seq = SequenceNormalizer.normalize(current_seq)
                    self.sequences.append(normalized_seq)
                    self.labels.append(label)
                # Start new sequence
                current_seq = ""
            else:
                current_seq += line
        
        # Add last sequence
        if current_seq:
            normalized_seq = SequenceNormalizer.normalize(current_seq)
            self.sequences.append(normalized_seq)
            self.labels.append(label)
    
    def _create_dummy_data(self, num_samples: int = 100):
        """Create dummy data for testing"""
        bases = ['A', 'T', 'C', 'G']
        for _ in range(num_samples):
            seq_length = random.randint(200, 800)
            sequence = ''.join(random.choice(bases) for _ in range(seq_length))
            self.sequences.append(sequence)
            self.labels.append(random.randint(0, 1))
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Convert sequence to tensor
        encoded_seq = SequenceNormalizer.encode(sequence)
        
        # Pad or truncate sequence to max_seq_length
        if len(encoded_seq) > self.max_seq_length:
            encoded_seq = encoded_seq[:self.max_seq_length]
        else:
            encoded_seq = encoded_seq + [4] * (self.max_seq_length - len(encoded_seq))
        
        seq_tensor = torch.tensor(encoded_seq, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            seq_tensor = self.transform(seq_tensor)
        
        return seq_tensor, label_tensor

class SequenceAugmenter:
    """Data augmentation for DNA sequences"""
    @staticmethod
    def random_mutation(sequence: str, mutation_rate: float = 0.05) -> str:
        """Randomly mutate bases in the sequence"""
        bases = ['A', 'T', 'C', 'G']
        chars = list(sequence)
        for i in range(len(chars)):
            if random.random() < mutation_rate and chars[i] in bases:
                # Replace with a different base
                chars[i] = random.choice([b for b in bases if b != chars[i]])
        return ''.join(chars)
    
    @staticmethod
    def reverse_complement(sequence: str) -> str:
        """Get the reverse complement of a DNA sequence"""
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
        return ''.join(complement.get(base, 'N') for base in reversed(sequence))
    
    @staticmethod
    def random_insert(sequence: str, insert_rate: float = 0.03) -> str:
        """Randomly insert bases into the sequence"""
        bases = ['A', 'T', 'C', 'G']
        chars = list(sequence)
        i = 0
        while i < len(chars):
            if random.random() < insert_rate:
                chars.insert(i, random.choice(bases))
                i += 2  # Skip the newly inserted base
            else:
                i += 1
        return ''.join(chars)

# ---------- Training Utilities ----------
class Trainer:
    """Model trainer with advanced features"""
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        precision: str = "mixed",
        save_dir: str = "./models"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.precision = precision
        self.save_dir = Path(save_dir)
        
        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=10,  # Will be updated in train()
            eta_min=learning_rate / 10
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Gradient scaler for mixed precision
        self.scaler = GradScaler(enabled=(precision == "mixed"))
        
        # Metrics tracking
        self.train_loss_history = []
        self.val_loss_history = []
        self.accuracy_history = []
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
    
    def train(
        self,
        epochs: int,
        device: torch.device,
        early_stop_patience: int = 5,
        save_best: bool = True
    ):
        """Train the model"""
        self.model.to(device)
        
        # Update scheduler T_max
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=self.learning_rate / 10
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("Loss: {task.fields[loss]:.4f}"),
            console=console
        ) as progress:
            task = progress.add_task("[bold green]Training...", total=epochs, loss=0.0)
            
            for epoch in range(epochs):
                # Train for one epoch
                train_loss = self._train_epoch(device)
                self.train_loss_history.append(train_loss)
                
                # Validate if validation data is provided
                if self.val_loader:
                    val_loss, accuracy = self._validate(device)
                    self.val_loss_history.append(val_loss)
                    self.accuracy_history.append(accuracy)
                    
                    progress.update(task, advance=1, loss=train_loss, 
                                   description=f"[bold green]Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {accuracy:.2f}%")
                    
                    # Early stopping check
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        
                        # Save best model
                        if save_best:
                            self.save_model("best_model.pt")
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stop_patience:
                            logger.info(f"Early stopping triggered after {epoch+1} epochs")
                            break
                else:
                    progress.update(task, advance=1, loss=train_loss,
                                   description=f"[bold green]Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")
                
                # Update learning rate
                self.scheduler.step()
        
        # Save final model
        self.save_model("final_model.pt")
        
        return {
            "train_loss": self.train_loss_history,
            "val_loss": self.val_loss_history,
            "accuracy": self.accuracy_history
        }
    
    def _train_epoch(self, device: torch.device) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision if enabled
            with autocast(enabled=(self.precision == "mixed")):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
            
            # Backward pass with gradient scaling if mixed precision
            if self.precision == "mixed":
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def _validate(self, device: torch.device) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                with autocast(enabled=(self.precision == "mixed")):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        
        return val_loss, accuracy
    
    def save_model(self, filename: str):
        """Save the model and training state"""
        save_path = self.save_dir / filename
        
        state_dict = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "train_loss_history": self.train_loss_history,
            "val_loss_history": self.val_loss_history,
            "accuracy_history": self.accuracy_history
        }
        
        torch.save(state_dict, save_path)
        logger.info(f"Model saved to {save_path}")
    
    @staticmethod
    def load_model(
        model: nn.Module,
        filename: str,
        save_dir: str = "./models",
        map_location: Optional[str] = None
    ) -> Tuple[nn.Module, Dict]:
        """Load a model from a saved state"""
        load_path = Path(save_dir) / filename
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model file {load_path} not found")
        
        state_dict = torch.load(load_path, map_location=map_location)
        model.load_state_dict(state_dict["model_state"])
        
        return model, state_dict

# ---------- API Components ----------
app = FastAPI(
    title="CQuence API",
    description="Gene Sequence Analysis API",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SequenceRequest(BaseModel):
    sequence: str = Field(..., description="DNA sequence to analyze")
    
    @validator('sequence')
    def validate_sequence(cls, v):
        valid_chars = {'A', 'T', 'C', 'G', 'N', 'a', 't', 'c', 'g', 'n'}
        if not all(c in valid_chars for c in v):
            raise ValueError("Sequence must contain only A, T, C, G, or N characters")
        return v.upper()

class AnalysisResponse(BaseModel):
    prediction: int
    confidence: float
    processed_length: int
    features: Dict[str, float]

@app.get("/")
async def root():
    return {"message": "Welcome to CQuence API", "status": "operational"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_sequence(request: SequenceRequest):
    """Analyze a DNA sequence"""
    try:
        # Load model
        model_path = Path(CONFIG.save_dir) / "best_model.pt"
        if not model_path.exists():
            model_path = Path(CONFIG.save_dir) / "final_model.pt"
            if not model_path.exists():
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="No trained model available. Please train a model first."
                )
        
        # Initialize model
        model = GeneTransformer(
            input_dim=CONFIG.input_dim,
            hidden_dim=CONFIG.hidden_dim,
            num_layers=CONFIG.num_layers,
            dropout=CONFIG.dropout
        )
        
        # Load model weights
        model, _ = Trainer.load_model(
            model, 
            model_path.name, 
            save_dir=CONFIG.save_dir,
            map_location=CONFIG.device
        )
        model.to(DEVICE)
        model.eval()
        
        # Process sequence
        sequence = request.sequence
        normalized_seq = SequenceNormalizer.normalize(sequence)
        encoded_seq = SequenceNormalizer.encode(normalized_seq)
        
        # Truncate or pad sequence
        if len(encoded_seq) > CONFIG.max_seq_length:
            encoded_seq = encoded_seq[:CONFIG.max_seq_length]
        else:
            encoded_seq = encoded_seq + [4] * (CONFIG.max_seq_length - len(encoded_seq))
        
        # Convert to tensor
        seq_tensor = torch.tensor([encoded_seq], dtype=torch.long).to(DEVICE)
        
        # Make prediction with autocast for mixed precision
        with torch.no_grad(), autocast(enabled=(CONFIG.precision == "mixed")):
            probs = model.predict_proba(seq_tensor)
        
        # Get prediction and confidence
        prediction = probs.argmax(dim=1).item()
        confidence = probs[0, prediction].item() * 100
        
        # Extract some basic sequence features
        gc_content = (normalized_seq.count('G') + normalized_seq.count('C')) / len(normalized_seq) * 100
        a_content = normalized_seq.count('A') / len(normalized_seq) * 100
        t_content = normalized_seq.count('T') / len(normalized_seq) * 100
        
        return AnalysisResponse(
            prediction=prediction,
            confidence=confidence,
            processed_length=len(normalized_seq),
            features={
                "gc_content": gc_content,
                "a_content": a_content,
                "t_content": t_content
            }
        )
    
    except Exception as e:
        logger.error(f"Error analyzing sequence: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

# ---------- CLI Interface ----------
cli = typer.Typer(help="CQuence: Gene Sequence Analysis Tool")

@cli.command()
def train(
    data_path: str = typer.Option(
        CONFIG.data_dir, 
        help="Path to the data directory containing sequence files"
    ),
    model_type: str = typer.Option(
        "hybrid", 
        help="Model type (transformer, cnn, hybrid)"
    ),
    epochs: int = typer.Option(
        10, 
        help="Number of training epochs"
    ),
    batch_size: int = typer.Option(
        32, 
        help="Training batch size"
    ),
    use_gpu: bool = typer.Option(
        USE_GPU,
        help="Use GPU for training if available"
    ),
    learning_rate: float = typer.Option(
        3e-4,
        help="Learning rate for optimizer"
    ),
    save_dir: str = typer.Option(
        CONFIG.save_dir,
        help="Directory to save model checkpoints"
    ),
    cache_dir: str = typer.Option(
        CONFIG.cache_dir,
        help="Directory to cache processed data"
    ),
    max_seq_length: int = typer.Option(
        1024,
        help="Maximum sequence length to use"
    )
):
    """Train the gene sequencing model"""
    console.print(Panel("[bold green]Starting model training...[/bold green]"))
    
    # Update config
    CONFIG.model_type = model_type
    CONFIG.batch_size = batch_size
    CONFIG.learning_rate = learning_rate
    CONFIG.save_dir = save_dir
    CONFIG.cache_dir = cache_dir
    CONFIG.max_seq_length = max_seq_length
    
    # Check GPU availability
    if use_gpu and not torch.cuda.is_available():
        console.print("[bold yellow]Warning: GPU requested but not available! Using CPU instead.[/bold yellow]")
        use_gpu = False
    
    device = torch.device("cuda" if use_gpu else "cpu")
    console.print(f"[bold blue]Using device:[/bold blue] {device}")
    
    # Load dataset
    console.print("[bold blue]Loading dataset...[/bold blue]")
    dataset = GeneDataset(
        data_dir=data_path,
        max_seq_length=max_seq_length,
        cache_dir=cache_dir
    )
    
    # Split dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=CONFIG.num_workers,
        pin_memory=use_gpu
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=CONFIG.num_workers,
        pin_memory=use_gpu
    )
    
    console.print(f"[bold blue]Dataset loaded:[/bold blue] {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    
    # Initialize model based on model_type
    if model_type == "transformer":
        model = GeneTransformer(
            input_dim=CONFIG.input_dim,
            hidden_dim=CONFIG.hidden_dim,
            num_layers=CONFIG.num_layers,
            dropout=CONFIG.dropout
        )
    elif model_type == "cnn":
        # Implement a CNN-only model if needed
        model = GeneTransformer(
            input_dim=CONFIG.input_dim,
            hidden_dim=CONFIG.hidden_dim,
            num_layers=2,  # Reduced transformer layers
            dropout=CONFIG.dropout
        )
    else:  # hybrid (default)
        model = GeneTransformer(
            input_dim=CONFIG.input_dim,
            hidden_dim=CONFIG.hidden_dim,
            num_layers=CONFIG.num_layers,
            dropout=CONFIG.dropout
        )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        precision="mixed" if use_gpu else "float32",
        save_dir=save_dir
    )
    
    # Train model
    console.print("[bold green]Training model...[/bold green]")
    training_stats = trainer.train(
        epochs=epochs,
        device=device,
        early_stop_patience=5,
        save_best=True
    )
    
    console.print("[bold green]Training complete![/bold green]")
    
    # Display training stats
    table = Table(title="Training Statistics")
    table.add_column("Epoch", justify="right", style="cyan")
    table.add_column("Train Loss", justify="right", style="green")
    table.add_column("Val Loss", justify="right", style="red")
    table.add_column("Accuracy (%)", justify="right", style="yellow")
    
    for i, (train_loss, val_loss, acc) in enumerate(zip(
        training_stats["train_loss"],
        training_stats["val_loss"],
        training_stats["accuracy"]
    )):
        table.add_row(
            str(i+1),
            f"{train_loss:.4f}",
            f"{val_loss:.4f}",
            f"{acc:.2f}"
        )
    
    console.print(table)
    
    return training_stats

@cli.command()
def serve(
    port: int = typer.Option(8000, help="Port to serve API on"),
    host: str = typer.Option("0.0.0.0", help="Host to serve API on"),
    model_path: str = typer.Option(
        None, 
        help="Path to model file (default: best_model.pt in save_dir)"
    ),
    save_dir: str = typer.Option(
        CONFIG.save_dir,
        help="Directory containing model checkpoints"
    )
):
    """Start the API server"""
    console.print(Panel("[bold green]Starting CQuence API server...[/bold green]"))
    
    # Check if model exists
    if model_path:
        model_file = Path(model_path)
    else:
        model_file = Path(save_dir) / "best_model.pt"
        if not model_file.exists():
            model_file = Path(save_dir) / "final_model.pt"
    
    if not model_file.exists():
        console.print(f"[bold red]Model file not found: {model_file}[/bold red]")
        console.print("[yellow]Please train a model first or specify a valid model path.[/yellow]")
        return
    
    console.print(f"[bold blue]Using model:[/bold blue] {model_file}")
    console.print(f"[bold blue]API will be available at:[/bold blue] http://{host}:{port}")
    
    # Start uvicorn
    uvicorn.run("cquence:app", host=host, port=port, reload=False)

@cli.command()
def analyze(
    sequence: str = typer.Option(..., help="DNA sequence to analyze"),
    model_path: str = typer.Option(
        None, 
        help="Path to model file (default: best_model.pt in save_dir)"
    ),
    save_dir: str = typer.Option(
        CONFIG.save_dir,
        help="Directory containing model checkpoints"
    )
):
    """Analyze a DNA sequence from the command line"""
    console.print(Panel("[bold green]Analyzing DNA sequence...[/bold green]"))
    
    # Check if model exists
    if model_path:
        model_file = Path(model_path)
    else:
        model_file = Path(save_dir) / "best_model.pt"
        if not model_file.exists():
            model_file = Path(save_dir) / "final_model.pt"
    
    if not model_file.exists():
        console.print(f"[bold red]Model file not found: {model_file}[/bold red]")
        console.print("[yellow]Please train a model first or specify a valid model path.[/yellow]")
        return
    
    # Initialize model
    model = GeneTransformer(
        input_dim=CONFIG.input_dim,
        hidden_dim=CONFIG.hidden_dim,
        num_layers=CONFIG.num_layers,
        dropout=CONFIG.dropout
    )
    
    # Load model weights
    model, _ = Trainer.load_model(
        model, 
        model_file.name if model_path is None else model_file,
        save_dir=save_dir,
        map_location=CONFIG.device
    )
    model.to(DEVICE)
    model.eval()
    
    # Process sequence
    normalized_seq = SequenceNormalizer.normalize(sequence)
    encoded_seq = SequenceNormalizer.encode(normalized_seq)
    
    # Truncate or pad sequence
    if len(encoded_seq) > CONFIG.max_seq_length:
        encoded_seq = encoded_seq[:CONFIG.max_seq_length]
    else:
        encoded_seq = encoded_seq + [4] * (CONFIG.max_seq_length - len(encoded_seq))
    
    # Convert to tensor
    seq_tensor = torch.tensor([encoded_seq], dtype=torch.long).to(DEVICE)
    
    # Make prediction
    with torch.no_grad(), autocast(enabled=(CONFIG.precision == "mixed")):
        probs = model.predict_proba(seq_tensor)
    
    # Get prediction and confidence
    prediction = probs.argmax(dim=1).item()
    confidence = probs[0, prediction].item() * 100
    
    # Extract some basic sequence features
    gc_content = (normalized_seq.count('G') + normalized_seq.count('C')) / len(normalized_seq) * 100
    
    # Display results
    console.print(f"[bold blue]Original sequence length:[/bold blue] {len(sequence)}")
    console.print(f"[bold blue]Processed sequence length:[/bold blue] {len(normalized_seq)}")
    console.print(f"[bold blue]GC content:[/bold blue] {gc_content:.2f}%")
    console.print(f"[bold green]Prediction:[/bold green] {'Positive' if prediction == 1 else 'Negative'}")
    console.print(f"[bold green]Confidence:[/bold green] {confidence:.2f}%")
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "gc_content": gc_content
    }

# ---------- Interactive CLI ----------
class InteractiveCLI:
    """Interactive command-line interface with rich formatting"""
    def __init__(self):
        self.console = Console()
        self.current_model = None
        self.model_loaded = False
    
    def start(self):
        """Start the interactive CLI"""
        self.console.print(Panel.fit(
            "[bold blue]CQuence Interactive CLI[/bold blue]\n"
            "[cyan]Advanced Gene Sequence Analysis Tool[/cyan]",
            border_style="green"
        ))
        
        self.show_system_info()
        self.main_menu()
    
    def show_system_info(self):
        """Show system information"""
        system_info = Table.grid(padding=1)
        system_info.add_column(style="green")
        system_info.add_column(style="yellow")
        
        system_info.add_row("System", f"{OS_TYPE}")
        system_info.add_row("Python", f"{sys.version.split()[0]}")
        system_info.add_row("PyTorch", f"{torch.__version__}")
        system_info.add_row("CPU Cores", f"{CPU_COUNT}")
        system_info.add_row("GPU Available", f"{USE_GPU}")
        system_info.add_row("Device", f"{DEVICE}")
        
        self.console.print(Panel(system_info, title="System Information", border_style="blue"))
    
    def main_menu(self):
        """Display the main menu"""
        while True:
            self.console.print("\n[bold cyan]Main Menu:[/bold cyan]")
            menu_items = [
                ("1", "Train Model", "Train a new gene sequence analysis model"),
                ("2", "Analyze Sequence", "Analyze a DNA sequence"),
                ("3", "Start API Server", "Start the REST API server"),
                ("4", "Manage Models", "View and manage trained models"),
                ("5", "Settings", "Configure CQuence settings"),
                ("6", "Exit", "Exit CQuence")
            ]
            
            menu_table = Table(show_header=False, box=None)
            menu_table.add_column(style="bold green", width=3)
            menu_table.add_column(style="bold yellow")
            menu_table.add_column(style="cyan")
            
            for item in menu_items:
                menu_table.add_row(*item)
            
            self.console.print(menu_table)
            
            choice = Prompt.ask("Enter your choice", choices=["1", "2", "3", "4", "5", "6"], default="1")
            
            if choice == "1":
                self.train_menu()
            elif choice == "2":
                self.analyze_menu()
            elif choice == "3":
                self.serve_menu()
            elif choice == "4":
                self.model_management_menu()
            elif choice == "5":
                self.settings_menu()
            elif choice == "6":
                self.console.print("[bold green]Thank you for using CQuence![/bold green]")
                break
    
    def train_menu(self):
        """Training menu"""
        self.console.print(Panel("[bold blue]Model Training[/bold blue]"))
        
        # Get training parameters
        data_path = Prompt.ask("Data directory", default=CONFIG.data_dir)
        model_type = Prompt.ask(
            "Model type", 
            choices=["transformer", "cnn", "hybrid"], 
            default="hybrid"
        )
        epochs = int(Prompt.ask("Number of epochs", default="10"))
        batch_size = int(Prompt.ask("Batch size", default=str(CONFIG.batch_size)))
        use_gpu = Confirm.ask("Use GPU (if available)", default=USE_GPU)
        
        # Confirm training
        self.console.print("\n[bold yellow]Training Configuration:[/bold yellow]")
        config_table = Table(show_header=False, box=None)
        config_table.add_column(style="green")
        config_table.add_column(style="cyan")
        
        config_table.add_row("Data Directory", data_path)
        config_table.add_row("Model Type", model_type)
        config_table.add_row("Epochs", str(epochs))
        config_table.add_row("Batch Size", str(batch_size))
        config_table.add_row("Use GPU", str(use_gpu))
        
        self.console.print(config_table)
        
        if Confirm.ask("Start training with these settings?", default=True):
            # Call the train function
            train(
                data_path=data_path,
                model_type=model_type,
                epochs=epochs,
                batch_size=batch_size,
                use_gpu=use_gpu
            )
    
    def analyze_menu(self):
        """Sequence analysis menu"""
        self.console.print(Panel("[bold blue]Sequence Analysis[/bold blue]"))
        
        # Check if model is loaded
        if not self._ensure_model_loaded():
            return
        
        # Get sequence input method
        input_method = Prompt.ask(
            "Input method",
            choices=["text", "file"],
            default="text"
        )
        
        sequence = ""
        if input_method == "text":
            sequence = Prompt.ask("Enter DNA sequence")
        else:
            file_path = Prompt.ask("Enter path to sequence file")
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    # Remove FASTA header if present
                    if content.startswith('>'):
                        sequence = ''.join(content.split('\n')[1:])
                    else:
                        sequence = content.replace('\n', '')
            except Exception as e:
                self.console.print(f"[bold red]Error reading file:[/bold red] {str(e)}")
                return
        
        # Analyze sequence
        if sequence:
            analyze(
                sequence=sequence,
                model_path=self.current_model if self.current_model else None
            )
    
    def serve_menu(self):
        """API server menu"""
        self.console.print(Panel("[bold blue]API Server[/bold blue]"))
        
        # Get server parameters
        host = Prompt.ask("Host", default="0.0.0.0")
        port = int(Prompt.ask("Port", default="8000"))
        model_path = self.current_model if self.model_loaded else None
        
        if model_path is None:
            model_dir = Path(CONFIG.save_dir)
            models = list(model_dir.glob("*.pt"))
            
            if not models:
                self.console.print("[bold yellow]No trained models found. Please train a model first.[/bold yellow]")
                return
            
            self.console.print("[bold yellow]Available models:[/bold yellow]")
            for i, model in enumerate(models):
                self.console.print(f"[green]{i+1}.[/green] {model.name}")
            
            model_idx = int(Prompt.ask("Select model", default="1")) - 1
            model_path = str(models[model_idx])
        
        # Start server
        self.console.print(f"[bold green]Starting server at http://{host}:{port}[/bold green]")
        serve(
            host=host,
            port=port,
            model_path=model_path
        )
    
    def model_management_menu(self):
        """Model management menu"""
        self.console.print(Panel("[bold blue]Model Management[/bold blue]"))
        
        model_dir = Path(CONFIG.save_dir)
        models = list(model_dir.glob("*.pt"))
        
        if not models:
            self.console.print("[bold yellow]No trained models found.[/bold yellow]")
            return
        
        self.console.print("[bold yellow]Available models:[/bold yellow]")
        
        models_table = Table(show_header=True)
        models_table.add_column("#", style="green")
        models_table.add_column("Model Name", style="cyan")
        models_table.add_column("Size", style="yellow")
        models_table.add_column("Created", style="blue")
        
        for i, model in enumerate(models):
            stats = model.stat()
            size_mb = stats.st_size / (1024 * 1024)
            created = stats.st_ctime
            
            models_table.add_row(
                str(i+1),
                model.name,
                f"{size_mb:.2f} MB",
                f"{created}"
            )
        
        self.console.print(models_table)
        
        # Model actions
        self.console.print("\n[bold cyan]Actions:[/bold cyan]")
        actions_table = Table(show_header=False, box=None)
        actions_table.add_column(style="bold green", width=3)
        actions_table.add_column(style="bold yellow")
        
        actions_table.add_row("1", "Load Model")
        actions_table.add_row("2", "Delete Model")
        actions_table.add_row("3", "Model Details")
        actions_table.add_row("4", "Back to Main Menu")
        
        self.console.print(actions_table)
        
        choice = Prompt.ask("Enter your choice", choices=["1", "2", "3", "4"], default="1")
        
        if choice == "1":
            model_idx = int(Prompt.ask("Select model to load", default="1")) - 1
            self.current_model = str(models[model_idx])
            self.model_loaded = True
            self.console.print(f"[bold green]Model loaded:[/bold green] {models[model_idx].name}")
        
        elif choice == "2":
            model_idx = int(Prompt.ask("Select model to delete", default="1")) - 1
            if Confirm.ask(f"Are you sure you want to delete {models[model_idx].name}?", default=False):
                models[model_idx].unlink()
                self.console.print(f"[bold red]Model deleted:[/bold red] {models[model_idx].name}")
                
                # Unload if this was the current model
                if self.current_model == str(models[model_idx]):
                    self.current_model = None
                    self.model_loaded = False
        
        elif choice == "3":
            model_idx = int(Prompt.ask("Select model for details", default="1")) - 1
            # Load model metadata and display details
            model_file = models[model_idx]
            try:
                state_dict = torch.load(model_file, map_location="cpu")
                details = Table(show_header=False)
                details.add_column(style="green")
                details.add_column(style="cyan")
                
                details.add_row("Model Name", model_file.name)
                details.add_row("Size", f"{model_file.stat().st_size / (1024 * 1024):.2f} MB")
                
                if "train_loss_history" in state_dict:
                    details.add_row("Training Loss (final)", f"{state_dict['train_loss_history'][-1]:.4f}")
                
                if "val_loss_history" in state_dict:
                    details.add_row("Validation Loss (final)", f"{state_dict['val_loss_history'][-1]:.4f}")
                
                if "accuracy_history" in state_dict:
                    details.add_row("Accuracy (final)", f"{state_dict['accuracy_history'][-1]:.2f}%")
                
                self.console.print(details)
            except Exception as e:
                self.console.print(f"[bold red]Error loading model details:[/bold red] {str(e)}")
    
    def settings_menu(self):
        """Settings menu"""
        self.console.print(Panel("[bold blue]Settings[/bold blue]"))
        
        settings_table = Table(show_header=False, box=None)
        settings_table.add_column(style="bold green", width=3)
        settings_table.add_column(style="bold yellow")
        settings_table.add_column(style="cyan")
        
        settings_table.add_row("1", "Data Directory", CONFIG.data_dir)
        settings_table.add_row("2", "Save Directory", CONFIG.save_dir)
        settings_table.add_row("3", "Cache Directory", CONFIG.cache_dir)
        settings_table.add_row("4", "Max Sequence Length", str(CONFIG.max_seq_length))
        settings_table.add_row("5", "Batch Size", str(CONFIG.batch_size))
        settings_table.add_row("6", "Back to Main Menu", "")
        
        self.console.print(settings_table)
        
        choice = Prompt.ask("Enter setting to change", choices=["1", "2", "3", "4", "5", "6"], default="6")
        
        if choice == "1":
            CONFIG.data_dir = Prompt.ask("Enter new data directory", default=CONFIG.data_dir)
        elif choice == "2":
            CONFIG.save_dir = Prompt.ask("Enter new save directory", default=CONFIG.save_dir)
        elif choice == "3":
            CONFIG.cache_dir = Prompt.ask("Enter new cache directory", default=CONFIG.cache_dir)
        elif choice == "4":
            CONFIG.max_seq_length = int(Prompt.ask("Enter new max sequence length", default=str(CONFIG.max_seq_length)))
        elif choice == "5":
            CONFIG.batch_size = int(Prompt.ask("Enter new batch size", default=str(CONFIG.batch_size)))
    
    def _ensure_model_loaded(self) -> bool:
        """Ensure a model is loaded, load one if not"""
        if self.model_loaded:
            return True
        
        # Look for available models
        model_dir = Path(CONFIG.save_dir)
        models = list(model_dir.glob("*.pt"))
        
        if not models:
            self.console.print("[bold yellow]No trained models found. Please train a model first.[/bold yellow]")
            return False
        
        # If only one model, load it automatically
        if len(models) == 1:
            self.current_model = str(models[0])
            self.model_loaded = True
            self.console.print(f"[bold green]Model automatically loaded:[/bold green] {models[0].name}")
            return True
        
        # Otherwise, ask user to select a model
        self.console.print("[bold yellow]Available models:[/bold yellow]")
        for i, model in enumerate(models):
            self.console.print(f"[green]{i+1}.[/green] {model.name}")
        
        model_idx = int(Prompt.ask("Select model", default="1")) - 1
        self.current_model = str(models[model_idx])
        self.model_loaded = True
        self.console.print(f"[bold green]Model loaded:[/bold green] {models[model_idx].name}")
        return True

# ---------- Main Execution ----------
if __name__ == "__main__":
    # Windows compatibility for multiprocessing
    if OS_TYPE == "Windows":
        torch.multiprocessing.freeze_support()
    
    if len(sys.argv) > 1:
        # CLI mode with arguments
        cli()
    else:
        # Interactive mode
        cli_app = InteractiveCLI()
        cli_app.start()
