import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
eval_iters = 200
n_layer=6
n_head=6
dropout=0.2 # used to randomly drop some neurons during training to not make the model depend on only one neuron basically used to prevent overfitting
n_embd=384  # the vector which gives meaning to each token 
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']: # for each split
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# this is a self attetion block
class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key=nn.Linear(n_embd,head_size,bias=False)  #What information does it have
        self.query=nn.Linear(n_embd,head_size,bias=False) #What is the token looking for
        self.value=nn.Linear(n_embd,head_size,bias=False) #What should the token share
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        B,T,C=x.shape
        k=self.key(x)
        q=self.query(x)
        #compute attention scores(affinities)
        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)
        wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        wei=F.softmax(wei,dim=-1)
        wei=self.dropout(wei)
        #weighted aggregation (the mean of the previous tokens and the present token)
        v=self.value(x)
        out=wei @ v
        return out

class MultiHeadAttention(nn.Module):
    
    def __init__(self,num_head,head_size):
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size) for _ in range(num_head)])
        self.proj=nn.Linear(head_size * num_head,n_embd)  # this step is used for optimisation
        self.dropout=nn.Dropout(dropout)  # used to randomly drop some neurons during training to not make the model depend on only one neuron

    def forward(self,x):  # x is the addition of the pos_emb and tok_emb 
        out=torch.cat([h(x) for h in self.heads],dim=-1)
        out=self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):

    def __init__(self,n_embd):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_embd,4*n_embd), # this is used to refine the own values of n_embd and give rise to a stronger learned network of n_embd
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd),  # the * 4 is used as a optimization technique to allow the model to learn more complex representations it was given imn the official documentation of the attention is all you need documentation
            nn.Dropout(dropout) # used to randomly drop some neurons during training to not make the model depend on only one neuron
        )
    def forward(self,x):
        return self.net(x)

class Block(nn.Module):  # this is used to run the multi head attention and feed forward network again to gain further understanding of the tokens by the model and improve the performance
     
    def __init__(self,n_embd,n_head):
         super().__init__()
         head_size=n_embd//n_head
         self.sa=MultiHeadAttention(n_head,head_size)
         self.ffwd=FeedForward(n_embd)
         self.ln1=nn.LayerNorm(n_embd)
         self.ln2=nn.LayerNorm(n_embd) # mean and variance are taken to normalise it so we take n_embd and based on it we normalise the layers and the data

    def forward(self,x):
        x= x+ self.sa(self.ln1(x))
        x= x + self.ffwd(self.ln2(x))
        return x
    

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)   # converts the vocab_size each token to a vector space of length n_embd to help the model know the importance of that specific word in the data
        self.position_embedding_table=nn.Embedding(block_size,n_embd)
            # changes the vector back to the vocab_size token to display the token
        # self.blocks=nn.Sequential(
        #     Block(n_embd,n_head=4),
        #     Block(n_embd,n_head=4),
        #     Block(n_embd,n_head=4),
        #     nn.LayerNorm(n_embd)
        # ) # the below line is same as this but prettier
        self.blocks=nn.Sequential(*[Block(n_embd,n_head=n_head) for _ in range(n_layer)])
        self.lm_head=nn.Linear(n_embd,vocab_size)
        # self.sa_head=MultiHeadAttention(4,n_embd//4)  # 4 is the no of multiheads that is looking at the tokens  
        # self.ffwd=FeedForward(n_embd)
        # we initialised these 2 above lines in Block class so we dont need it here
        self.lnf=nn.LayerNorm(n_embd) # this is the finalNorm and it is used to normalise the output of the model until now and make the data clean

    def forward(self, idx, targets=None):
        B,T=idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb=self.token_embedding_table(idx)  # the tokens that are embedded into n_embd space vector (B,T,n_embd)
        pos_emb=self.position_embedding_table(torch.arange(T,device=device))
        x=tok_emb+pos_emb
        #x=self.sa_head(x)  # this is the multi head attention block and passing the value to it
        #x=self.ffwd(x)  # after finding the attention of the token in the data and how it corresponds and the position of the token to other tokens we then use feed forward to further improve the model by learning from the patterns fo the multi head attention block and further improve the weight and quality of the value
        # the above lines are replaced by the line below due to the Block class running these even better
        x=self.blocks(x)
        # after the above x runs it returns messy data because of the continous computation of the blocks and layers so we use the below line to normalise it and make it clean
        x=self.lnf(x)
        logits = self.lm_head(x) # (B,T,C)   #it converts the vector into tokens (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # cropping the content of idx to match the size of batch_size
            # when generating text, the model works on batches in parallel for each batch, at every step, it looks only at the last block_size tokens using this
            idx_crop=idx[:,-block_size:] # this crop the length of block_size from the last and pass them to the forward layer and then find thier position embedding and token_embedding of those characters it 
            # get the predictions
            logits, loss = self(idx_crop)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))