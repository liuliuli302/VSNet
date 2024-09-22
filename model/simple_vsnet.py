import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, Tuple


class SimpleVSCofig:
    def __init__(
        self,
        input_size=1024,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        **kwargs
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout


class SimpleVSEmbeddings(nn.Module):
    
    def __init__(self, config: SimpleVSCofig):
        super().__init__()
        self.config = config
        self.fc = nn.Linear(config.input_size, config.hidden_size)
        
    def forward(self, input: torch.Tensor):
        input = self.fc(input)
        input = nn.functional.gelu(input, approximate="tanh")
        
        return input


class SimpleVSMLP(nn.Module):
    
    def __init__(self, config: SimpleVSCofig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)
        
        return hidden_states


class SimpleVSAttention(nn.Module):
    
    def __init__(self, config: SimpleVSCofig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_weights = (torch.matmul(query_states, key_states.transpose(2,3))* self.scale)
        
        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Wrong Attention weights shape!"
            )

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        
        attn_output = attn_output.transpose(1,2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        return attn_output


class SimpleVSEncoderLayer(nn.Module):
    
    def __init__(self, config: SimpleVSCofig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SimpleVSAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SimpleVSMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = hidden_states + residual
        
        residual = hidden_states
        
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class SimpleVSEncoder(nn.Module):

    def __init__(self, config: SimpleVSCofig):
        super().__init__()
        self.cofig = config
        self.layers = nn.ModuleList(
            [SimpleVSEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
    
    def forward(
        self,
        inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
            
        return hidden_states


class SimpleVSClassifier(nn.Module):
    
    def __init__(self, config: SimpleVSCofig):
        super().__init__()
        self.sum = nn.Linear(config.hidden_size, 1)
    
    def forward(self, input):
        input = self.sum(input)
        
        return input
        

class SimpleVSNet(nn.Module):
    
    def __init__(self, config: SimpleVSCofig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        
        self.embeddings = SimpleVSEmbeddings(config)
        self.encoder = SimpleVSEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.sum_layer = SimpleVSClassifier(config)
                
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(input)
        
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)
        
        last_hidden_state = self.post_layernorm(last_hidden_state)
        
        logits = self.sum_layer(last_hidden_state)
        
        return logits


class SimpleVSModule(pl.LightningModule):
    
    def __init__(self, config: SimpleVSCofig):
        self.net = SimpleVSNet(config)
        
    def forward(self, input):
        return self.net(input)


if __name__ == "__main__":
    config = SimpleVSCofig(
        input_size=1024,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
    )
    
    net = SimpleVSNet(config)

    random_input = torch.randn(4, 500, 1024)
    output = net.forward(random_input)
    print(output)
    
    


