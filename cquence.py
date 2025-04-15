# geneai_integrated.py
import os
import sys
import platform
from pathlib import Path
from typing import Optional, List
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import typer
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from torch.utils.data import Dataset, DataLoader

# ---------- Configuration ----------
OS_TYPE = platform.system()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {OS_TYPE} with device: {DEVICE}")

# ---------- Core Models ----------
class GeneTransformer(nn.Module):
    """Hybrid Transformer-CNN for gene sequence analysis"""
    def __init__(self, input_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(4, input_dim)  # DNA bases: A,T,C,G
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=4),
            num_layers=4
        )
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.classifier = nn.Linear(hidden_dim, 2)

    @autocast()
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x).squeeze()
        return self.classifier(x)

class TextGAN(nn.Module):
    """GAN for text/speech interaction"""
    def __init__(self, latent_dim=128):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512)
        )
        self.discriminator = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def generate(self, z):
        return self.generator(z)

# ---------- Data Handling ----------
class GeneDataset(Dataset):
    def __init__(self, data_dir: str):
        self.sequences = self._load_sequences(data_dir)
        
    def _load_sequences(self, data_dir):
        # Simplified example - implement actual DNA sequence loading
        return ["ATCG", "GCTA", "TAGC"]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return self._dna_to_tensor(seq), 0  # Dummy label

    def _dna_to_tensor(self, seq):
        mapping = {'A':0, 'T':1, 'C':2, 'G':3}
        return torch.tensor([mapping.get(s, 0) for s in seq], dtype=torch.long)

# ---------- Training Utilities ----------
def create_optimizer(model: nn.Module, lr: float = 0.001):
    return torch.optim.Adam(model.parameters(), lr=lr)

def train_model(
    model: nn.Module,
    data_loader: DataLoader,
    epochs: int = 10,
    use_gpu: bool = True
):
    model.to(DEVICE)
    optimizer = create_optimizer(model)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")

# ---------- API Components ----------
app = FastAPI()

class SequenceRequest(BaseModel):
    sequence: str

@app.post("/analyze")
async def analyze_sequence(request: SequenceRequest):
    model = GeneTransformer().to(DEVICE)
    inputs = GeneDataset._dna_to_tensor(request.sequence).unsqueeze(0)
    with torch.no_grad():
        output = model(inputs.to(DEVICE))
    return {"prediction": output.argmax().item()}

# ---------- CLI Interface ----------
cli = typer.Typer()

@cli.command()
def train(
    data_path: str = typer.Option(...),
    epochs: int = typer.Option(10),
    use_gpu: bool = typer.Option(False)
):
    """Train the gene sequencing model"""
    dataset = GeneDataset(data_path)
    loader = DataLoader(dataset, batch_size=2)
    model = GeneTransformer()
    
    if use_gpu and not torch.cuda.is_available():
        print("Warning: GPU requested but not available!")
    
    train_model(model, loader, epochs, use_gpu)
    print("Training complete!")

@cli.command()
def serve(
    port: int = typer.Option(8000),
    host: str = typer.Option("0.0.0.0")
):
    """Start the API server"""
    uvicorn.run("geneai_integrated:app", host=host, port=port, reload=True)

# ---------- Main Execution ----------
if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Available commands:")
        print("  train   - Train the model")
        print("  serve   - Start the API server")
        sys.exit()
    
    # Windows compatibility for multiprocessing
    if OS_TYPE == "Windows":
        torch.multiprocessing.freeze_support()
    
    cli()
