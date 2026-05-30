#!/usr/bin/env python
from omnimesh import OmniMeshV2, ModelConfig

def main():
    config = ModelConfig()
    model = OmniMeshV2(config)
    
    response = model.generate(prompt="Jelaskan apa itu machine learning dalam 2 kalimat.")
    print("Response:\n", response)

if __name__ == "__main__":
    main()
