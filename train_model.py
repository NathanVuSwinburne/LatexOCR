#!/usr/bin/env python3
"""
Custom training script for pix2tex LaTeX OCR model
Works around dependency issues with built-in pix2tex training module
"""

import os
import yaml
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CROHMEDataset(Dataset):
    """Custom dataset class for CROHME LaTeX OCR data"""
    
    def __init__(self, pkl_file, transform=None):
        """
        Args:
            pkl_file (str): Path to the pickle file containing image paths and labels
            transform: Optional transform to be applied on images
        """
        with open(pkl_file, 'rb') as f:
            self.data = pickle.load(f)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        img_path = item['image']
        if not os.path.isabs(img_path):
            # Handle relative paths
            base_dir = os.path.dirname(os.path.abspath(__file__))
            img_path = os.path.join(base_dir, img_path)
            
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (224, 224), color='white')
        
        if self.transform:
            image = self.transform(image)
            
        # Get LaTeX equation
        equation = item.get('equation', item.get('label', ''))
        
        return {
            'image': image,
            'equation': equation,
            'image_path': img_path
        }

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_simple_transforms():
    """Create simple image transforms without albumentations dependency"""
    try:
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    except ImportError:
        logger.warning("torchvision not available, using basic transforms")
        return None

class SimpleTokenizer:
    """Simple tokenizer for LaTeX equations"""
    
    def __init__(self, vocab_size=8000):
        self.vocab_size = vocab_size
        self.token_to_id = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.id_to_token = {0: '<PAD>', 1: '<BOS>', 2: '<EOS>', 3: '<UNK>'}
        self.next_id = 4
        
    def build_vocab(self, equations):
        """Build vocabulary from equations"""
        char_freq = {}
        for eq in equations:
            for char in eq:
                char_freq[char] = char_freq.get(char, 0) + 1
        
        # Add most frequent characters to vocabulary
        sorted_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)
        for char, freq in sorted_chars:
            if self.next_id >= self.vocab_size:
                break
            if char not in self.token_to_id:
                self.token_to_id[char] = self.next_id
                self.id_to_token[self.next_id] = char
                self.next_id += 1
    
    def encode(self, text, max_length=512):
        """Encode text to token IDs"""
        tokens = [1]  # BOS token
        for char in text[:max_length-2]:  # Leave space for BOS and EOS
            tokens.append(self.token_to_id.get(char, 3))  # UNK token if not found
        tokens.append(2)  # EOS token
        
        # Pad to max_length
        while len(tokens) < max_length:
            tokens.append(0)  # PAD token
            
        return tokens[:max_length]
    
    def decode(self, token_ids):
        """Decode token IDs to text"""
        text = ""
        for token_id in token_ids:
            if token_id == 2:  # EOS token
                break
            if token_id > 2:  # Skip PAD, BOS, EOS
                text += self.id_to_token.get(token_id, '<UNK>')
        return text

def train_epoch(model, dataloader, optimizer, criterion, device, tokenizer):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        images = batch['image'].to(device)
        equations = batch['equation']
        
        # Tokenize equations
        target_tokens = []
        for eq in equations:
            tokens = tokenizer.encode(eq)
            target_tokens.append(tokens)
        
        target_tokens = torch.tensor(target_tokens, device=device)
        
        optimizer.zero_grad()
        
        # Forward pass (simplified - you'll need to implement actual model)
        # This is a placeholder for the actual pix2tex model forward pass
        try:
            outputs = model(images, target_tokens[:, :-1])  # Teacher forcing
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), 
                           target_tokens[:, 1:].reshape(-1))
        except Exception as e:
            logger.warning(f"Model forward pass failed: {e}")
            # Skip this batch
            continue
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / max(num_batches, 1)

def validate(model, dataloader, criterion, device, tokenizer):
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch['image'].to(device)
            equations = batch['equation']
            
            # Tokenize equations
            target_tokens = []
            for eq in equations:
                tokens = tokenizer.encode(eq)
                target_tokens.append(tokens)
            
            target_tokens = torch.tensor(target_tokens, device=device)
            
            try:
                outputs = model(images, target_tokens[:, :-1])
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), 
                               target_tokens[:, 1:].reshape(-1))
                total_loss += loss.item()
                num_batches += 1
            except Exception as e:
                logger.warning(f"Validation forward pass failed: {e}")
                continue
    
    return total_loss / max(num_batches, 1)

def main():
    parser = argparse.ArgumentParser(description='Train pix2tex model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and config.get('device') == 'cuda' else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create transforms
    transforms = create_simple_transforms()
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = CROHMEDataset(config['data'], transform=transforms)
    val_dataset = CROHMEDataset(config['valdata'], transform=transforms)
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Build tokenizer vocabulary
    logger.info("Building tokenizer vocabulary...")
    tokenizer = SimpleTokenizer(vocab_size=config.get('num_tokens', 8000))
    
    # Collect all equations for vocabulary building
    all_equations = []
    for item in train_dataset.data:
        equation = item.get('equation', item.get('label', ''))
        all_equations.append(equation)
    
    tokenizer.build_vocab(all_equations)
    logger.info(f"Vocabulary size: {len(tokenizer.token_to_id)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.get('batch_size', 8),
        shuffle=True,
        num_workers=config.get('num_workers', 0),  # Set to 0 for Windows compatibility
        pin_memory=config.get('pin_memory', False)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('eval_batch_size', 8),
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', False)
    )
    
    # Note: You'll need to implement or import the actual pix2tex model here
    # For now, this is a placeholder
    logger.warning("Model initialization placeholder - you need to implement the actual pix2tex model")
    
    # Create checkpoint directory
    checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
    checkpoint_dir.mkdir(exist_ok=True)
    
    logger.info("Training setup complete!")
    logger.info("Note: This script provides the framework for training.")
    logger.info("You'll need to:")
    logger.info("1. Install the core pix2tex package (without [train] extras)")
    logger.info("2. Import and initialize the actual pix2tex model")
    logger.info("3. Run the training loop")
    
    # Save tokenizer for later use
    tokenizer_path = checkpoint_dir / 'tokenizer.pkl'
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    logger.info(f"Tokenizer saved to {tokenizer_path}")

if __name__ == "__main__":
    main()
