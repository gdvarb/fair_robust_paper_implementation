import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import nltk
from nltk.corpus import wordnet, stopwords

nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

#seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class NewsGroupDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long),
            'text': text
        }

def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym != word and synonym not in synonyms:
                synonyms.append(synonym)
    return synonyms

def generate_adversarial_examples(texts, perturb_percent=0.15):
    adv_texts = []
    stop_words = set(stopwords.words('english'))
    
    for text in texts:
        tokens = nltk.word_tokenize(text)
        num_to_perturb = max(1, int(len(tokens) * perturb_percent))
        
        valid_indices = [i for i, token in enumerate(tokens) 
                       if len(token) > 3 and token.isalpha() and token.lower() not in stop_words]
        
        if not valid_indices:
            adv_texts.append(text)
            continue
            
        indices_to_perturb = random.sample(valid_indices, min(num_to_perturb, len(valid_indices)))
        adv_tokens = tokens.copy()
        
        for idx in indices_to_perturb:
            synonyms = get_synonyms(tokens[idx])
            if synonyms:
                adv_tokens[idx] = random.choice(synonyms)
        
        adv_texts.append(' '.join(adv_tokens))
    
    return adv_texts

def evaluate(model, data_loader, num_classes, perturb_percent=0.15, device='cpu'):
    model.eval()
    
    clean_correct = torch.zeros(num_classes)
    adv_correct = torch.zeros(num_classes)
    class_count = torch.zeros(num_classes)
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            texts = batch['text']
            
            # Evaluate on clean examples
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            clean_result = (torch.max(outputs.logits, 1)[1] == labels)
            
            # Generate adversarial examples
            adv_texts = generate_adversarial_examples(texts, perturb_percent)
            
            # Tokenize adversarial examples
            tokenizer = model.config._name_or_path
            if isinstance(tokenizer, str):
                tokenizer = DistilBertTokenizer.from_pretrained(tokenizer)
                
            adv_encodings = tokenizer(
                adv_texts,
                truncation=True,
                padding='max_length',
                max_length=input_ids.size(1),
                return_tensors='pt'
            )
            
            # Evaluate on adversarial examples
            adv_outputs = model(
                input_ids=adv_encodings['input_ids'].to(device),
                attention_mask=adv_encodings['attention_mask'].to(device)
            )
            adv_result = (torch.max(adv_outputs.logits, 1)[1] == labels)
            
            # Update class-specific metrics
            for c in range(num_classes):
                idx = (labels == c)
                if idx.sum() > 0:
                    clean_correct[c] += clean_result[idx].sum().item()
                    adv_correct[c] += adv_result[idx].sum().item()
                    class_count[c] += idx.sum().item()
    
    # Calculate error rates
    class_clean_error = 1.0 - clean_correct / class_count
    class_adv_error = 1.0 - adv_correct / class_count
    class_bndy_error = clean_correct / class_count - adv_correct / class_count
    
    # Average errors
    total_clean_error = 1.0 - clean_correct.sum() / class_count.sum()
    total_adv_error = 1.0 - adv_correct.sum() / class_count.sum()
    total_bndy_error = (clean_correct.sum() - adv_correct.sum()) / class_count.sum()
    
    # Worst-class errors
    worst_class_clean_error = torch.max(class_clean_error).item()
    worst_class_adv_error = torch.max(class_adv_error).item()
    worst_class_bndy_error = torch.max(class_bndy_error).item()
    
    print(f"Average Clean Error: {total_clean_error:.4f}, Adversarial Error: {total_adv_error:.4f}, Boundary Error: {total_bndy_error:.4f}")
    print(f"Worst-class Clean Error: {worst_class_clean_error:.4f}, Adversarial Error: {worst_class_adv_error:.4f}, Boundary Error: {worst_class_bndy_error:.4f}")
    
    return class_clean_error, class_adv_error, class_bndy_error, total_clean_error, total_adv_error, total_bndy_error, worst_class_clean_error, worst_class_adv_error, worst_class_bndy_error

def frl_train(model, tokenizer, train_loader, valid_loader, optimizer, scheduler, epoch, num_classes,
              perturb_percent, device, delta0, delta1, rate1, rate2, lmbda, beta, lim):
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Evaluate validation set for lambda updates
    print(f"\nEvaluating validation set (pre-training for epoch {epoch})")
    class_clean_error, class_adv_error, class_bndy_error, total_clean_error, total_adv_error, total_bndy_error, *_ = \
        evaluate(model, valid_loader, num_classes, perturb_percent, device)
    
    # Split lambda into parts for standard error and boundary error
    lmbda_clean = lmbda[:num_classes]
    lmbda_adv = lmbda[num_classes:2*num_classes]
    
    # Update lambdas
    print("\nUpdating lambda values based on fairness constraints:")
    for i in range(num_classes):
        clean_vio = class_clean_error[i] - total_clean_error - delta0[i]
        lmbda_clean[i] = torch.clamp(lmbda_clean[i] + rate1 * clean_vio, min=0, max=lim)
        
        adv_vio = class_adv_error[i] - total_adv_error - delta1[i]
        lmbda_adv[i] = torch.clamp(lmbda_adv[i] + rate2 * adv_vio, min=0, max=lim)
        
        print(f"Class {i}: lambda_clean={lmbda_clean[i]:.4f}, lambda_adv={lmbda_adv[i]:.4f}")
    
    # Training loop
    print("\nBeginning training")
    for batch in tqdm(train_loader, desc=f"Training (Epoch {epoch})"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        texts = batch['text']
        
        # forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        # Generate and process adversarial examples
        adv_texts = generate_adversarial_examples(texts, perturb_percent)
        adv_encodings = tokenizer(
            adv_texts,
            truncation=True,
            padding='max_length',
            max_length=input_ids.size(1),
            return_tensors='pt'
        )
        
        adv_outputs = model(
            input_ids=adv_encodings['input_ids'].to(device),
            attention_mask=adv_encodings['attention_mask'].to(device), 
            labels=labels
        )
        
        # Calculate per-sample losses
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        clean_sample_losses = loss_fct(outputs.logits, labels)
        adv_sample_losses = loss_fct(adv_outputs.logits, labels)
        
        # fairness weights
        batch_loss = 0.0
        for i, label in enumerate(labels):
            class_idx = label.item()
            batch_loss += (1.0 + lmbda_clean[class_idx]) * clean_sample_losses[i] + \
                         beta * (1.0 + lmbda_adv[class_idx]) * adv_sample_losses[i]
        
        batch_loss = batch_loss / len(labels)
        
        # Backward 
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += batch_loss.item()
        num_batches += 1
    
    # Update lambda values
    lmbda = torch.cat([lmbda_clean, lmbda_adv])
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    print(f"Epoch {epoch} - Average loss: {avg_loss:.4f}")
    
    return lmbda

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    set_seed(42)
    
    # Hyperparameters
    num_epochs = 5
    batch_size = 16
    learning_rate = 2e-5
    max_length = 128
    
    # FRL hyperparameters
    beta = 1.0
    bound0 = 0.05  # Fairness bound for clean error
    bound1 = 0.05  # Fairness bound for adversarial error
    rate1 = 0.01   # Update rate for clean fairness
    rate2 = 0.05   # Update rate for adversarial fairness
    lim = 0.5      # Limit for lambda values
    perturb_percent = 0.15
    
    # Load dataset
    categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
    num_classes = len(categories)
    
    print("Loading 20 Newsgroups dataset...")
    newsgroups = fetch_20newsgroups(
        subset='all', 
        categories=categories,
        remove=('headers', 'footers', 'quotes')
    )
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        newsgroups.data, newsgroups.target, test_size=0.3, random_state=42
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    # Create datasets and loaders
    train_dataset = NewsGroupDataset(X_train, y_train, tokenizer, max_length)
    valid_dataset = NewsGroupDataset(X_valid, y_valid, tokenizer, max_length)
    test_dataset = NewsGroupDataset(X_test, y_test, tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=num_classes
    ).to(device)
    
    # optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    # Initialize constraints and multipliers
    delta0 = bound0 * torch.ones(num_classes)
    delta1 = bound1 * torch.ones(num_classes)
    lmbda = torch.zeros(2 * num_classes).to(device)
    
    # store results
    results_test = []
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        print(f"\n=== Epoch {epoch}/{num_epochs} ===")
        
        lmbda = frl_train(
            model, tokenizer, train_loader, valid_loader, optimizer, scheduler, epoch, num_classes,
            perturb_percent, device, delta0, delta1, rate1, rate2, lmbda, beta, lim
        )
        
        # Evaluate
        print("\nEvaluating on test set:")
        class_clean_error, class_adv_error, class_bndy_error, total_clean_error, total_adv_error, total_bndy_error, \
        worst_class_clean_error, worst_class_adv_error, worst_class_bndy_error = evaluate(
            model, test_loader, num_classes, perturb_percent, device
        )
        
        results_test.append(np.concatenate((
            np.array([total_clean_error, total_adv_error, total_bndy_error]),
            np.array([worst_class_clean_error, worst_class_adv_error, worst_class_bndy_error]),
            class_clean_error.cpu().numpy(), 
            class_adv_error.cpu().numpy(),
            class_bndy_error.cpu().numpy()
        )))
       
        # Decay learning rate if needed
        if epoch % 3 == 0:
            rate1 = rate1 / 2
    
    # Save model and results
    model.save_pretrained('frl_distilbert_model')
    print('Final model saved successfully.')
    
    np.savetxt('results_frl_distilbert_test.txt', np.array(results_test))
    
    # Final evaluation
    print("\nFinal evaluation:")
    evaluate(model, test_loader, num_classes, perturb_percent, device)

if __name__ == '__main__':
    main()
