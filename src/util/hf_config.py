from dataclasses import dataclass, field

# ==== CONFIGURATION ====
@dataclass 
class HFConfig:
    HF_TOKEN_UPLOAD : str
    HF_TOKEN_DOWNLOAD: str     
    HF_USERNAME : str      
    
hf_config = HFConfig(
    HF_TOKEN_UPLOAD="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",
    HF_TOKEN_DOWNLOAD="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",
    HF_USERNAME="mariakrissmer" 
)  
