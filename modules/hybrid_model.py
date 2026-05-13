import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from .configs import HybridTTSConfig
from .diffusion_head.cfm import DiT
from .diffusion_head.cfm import DiT
from .output_dataclasses import HybridTTSOutput

class MLPAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.in_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.out_dim)
        )
    def forward(self, x):
        return self.net(x)

class HybridTTS(nn.Module):
    def __init__(self, config: HybridTTSConfig):
        super().__init__()
        self.config = config
        
        # Backbone
        bb_cfg = config.backbone_config
        if bb_cfg.pretrained:
            self.backbone = AutoModel.from_pretrained(bb_cfg.model_name_or_path)
        else:
            hf_config = AutoConfig.from_pretrained(bb_cfg.model_name_or_path)
            self.backbone = AutoModel.from_config(hf_config)
            
        hidden_size = self.backbone.config.hidden_size
        
        # Override config sizes based on actual backbone hidden size
        self.config.backbone_hidden_size = hidden_size
        self.config.token_head_config.in_dim = hidden_size
        if self.config.continuous_adapter_config is not None:
            self.config.continuous_adapter_config.out_dim = hidden_size
        self.config.diffusion_head_config.backbone_dim = hidden_size
            
        # Embeddings
        self.prompt_emb = nn.Embedding(bb_cfg.vocab_size, hidden_size)
        self.discrete_emb = nn.Embedding(config.discrete_token_vocab_size, hidden_size)
        
        # Optional Continuous Adapter
        if config.continuous_adapter_config is not None:
            self.continuous_adapter = MLPAdapter(config.continuous_adapter_config)
        else:
            self.continuous_adapter = nn.Linear(config.continuous_dim, hidden_size)
            
        # Normalizations
        self.norm_discrete = nn.LayerNorm(hidden_size)
        self.norm_continuous = nn.LayerNorm(hidden_size)
        
        # Special Tokens for Audio Delimitation
        self.start_audio_emb = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.end_audio_emb = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        # Output Heads
        self.token_head = nn.Linear(hidden_size, config.discrete_token_vocab_size)
        self.diffusion_head = DiT(config.diffusion_head_config)

    def forward(
        self, 
        prompt_ids: torch.Tensor, 
        discrete_tokens: torch.Tensor, 
        continuous_tokens: torch.Tensor,
        prompt_mask: torch.BoolTensor = None,
        padding_mask: torch.BoolTensor = None,
        target_continuous: torch.Tensor = None,
        **kwargs
    ):
        """
        Forward pass for training.
        """
        # 1. Prompt Embeddings
        p_emb = self.prompt_emb(prompt_ids)
        B, L_prompt, _ = p_emb.shape
        
        # 2. Input Representations
        d_emb = self.discrete_emb(discrete_tokens)
        c_emb = self.continuous_adapter(continuous_tokens)
        
        # 3. Normalization
        d_emb = self.norm_discrete(d_emb)
        c_emb = self.norm_continuous(c_emb)
        
        # Combine inputs for the audio part
        audio_emb = d_emb + c_emb
        L_audio = audio_emb.shape[1]
        
        # 4. Special Tokens
        start_emb = self.start_audio_emb.expand(B, -1, -1)
        end_emb = self.end_audio_emb.expand(B, -1, -1)
        
        # 5. Concatenate Sequence: [Prompt, START, Audio, END]
        inputs_embeds = torch.cat([p_emb, start_emb, audio_emb, end_emb], dim=1)
        
        # 6. Attention Mask
        if prompt_mask is None:
            prompt_mask = torch.ones((B, L_prompt), dtype=torch.bool, device=p_emb.device)
        if padding_mask is None:
            padding_mask = torch.zeros((B, L_audio), dtype=torch.bool, device=audio_emb.device)
            
        # In HF, attention_mask: 1 for attend, 0 for ignore
        prompt_attn = prompt_mask.long()
        start_attn = torch.ones((B, 1), dtype=torch.long, device=p_emb.device)
        audio_attn = (~padding_mask).long()
        end_attn = torch.ones((B, 1), dtype=torch.long, device=p_emb.device)
        
        attention_mask = torch.cat([prompt_attn, start_attn, audio_attn, end_attn], dim=1)
        
        # 7. Backbone Pass
        outputs = self.backbone(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        full_hidden_states = outputs.last_hidden_state
        
        # 8. Slicing Audio Hidden States for the Heads
        # The audio part starts after the prompt and the START token.
        audio_hidden_states = full_hidden_states[:, L_prompt + 1 : L_prompt + 1 + L_audio, :]
        
        # 9. Output Heads
        # Discrete Token Head
        token_logits = self.token_head(audio_hidden_states)
        
        # Diffusion Head (Continuous)
        # target_continuous is the actual continuous representation from VAE
        if target_continuous is None:
            target_continuous = continuous_tokens # Fallback if not provided separately
            
        diffusion_output = self.diffusion_head(
            target=target_continuous,
            target_padding_mask=padding_mask if padding_mask is not None else torch.zeros(target_continuous.shape[:2], dtype=torch.bool, device=target_continuous.device),
            context_vector=audio_hidden_states
        )
        
        return HybridTTSOutput(
            token_logits=token_logits,
            diffusion_loss=diffusion_output.loss,
            diffusion_output=diffusion_output
        )

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device
