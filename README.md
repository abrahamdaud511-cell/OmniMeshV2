# OmniMeshV2 - Universal AI Model

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**OmniMeshV2** adalah model AI universal dengan kemampuan:
- Membaca berbagai format file (PDF, DOC, HTML, kode sumber)
- Tokenisasi kode dengan AST (Tree-sitter)
- Stabilitas training adaptif (ATSG) monitoring GPU/CPU
- Arsitektur Sparse-Dense Interleaved (48 blok + MoE + eRCD)
- Constitutional Safety Router v2 untuk output aman
- 3 mode training: ML Klasik, Expert (fine-tune), Scratch (from zero)

## 🚀 Quick Start

### Instalasi
```bash
git clone https://github.com/username/OmniMeshV2.git
cd OmniMeshV2
pip install -r requirements.txt
