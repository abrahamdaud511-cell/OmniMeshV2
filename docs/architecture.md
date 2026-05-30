
### 12. `docs/architecture.md`
```markdown
# Arsitektur OmniMeshV2

## Komponen Utama

1. **Universal Data Ingestion Engine (UDIE)**
   - Mendukung teks, PDF, DOC, HTML, kode (Python, JS, dll)
   - Menggunakan BPE tokenizer untuk teks dan AST parser untuk kode

2. **Adaptive Training Stability Governor (ATSG)**
   - Monitor GPU/CPU real-time
   - PID controller untuk menyesuaikan batch size dan top-k MoE
   - Throttle data loading jika overload

3. **Enhanced Recursive Context Distiller (eRCD)**
   - Memampatkan konteks panjang menjadi summary
   - Memory bank dengan pruning berdasarkan relevance score

4. **Sparse-Dense Interleaved Backbone**
   - 48 super block (Dense → Sparse MoE → eRCD)
   - 64 experts, top-k dinamis
   - Distillation setiap 4 blok

5. **Constitutional Safety Router v2 (CSR v2)**
   - 12 prinsip keamanan umum + 6 prinsip kode
   - Static analysis untuk kode (SQL injection, hardcoded secrets, dll)
   - Self-correction loop

## Alur Data
Input → UDIE → Embedding → Backbone (48x) → LM Head → CSR v2 → Output

## contoh input
text,label
Produk bagus,positif
Layanan buruk,negatif
