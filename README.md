This is an implementation of the paper 

Freitag, Markus, and Scott Roy. "Unsupervised natural language generation with denoising autoencoders." arXiv preprint arXiv:1804.07899 (2018).

### Organisation
- `./data/`: Contains a folder called e2e with E2E dataset and optionally, Out of domain data(news-commentary-v15.en)
- `./src/`: Source code
- `./res/`: Saved Models and results

### Results

# Unsupervised setup (with OOD): 
BLEU: 0.5653 
NIST: 7.0124
METEOR: 0.3928
ROUGE_L: 0.6040
CIDEr: 1.4143

# Supervised setup:
BLEU: 0.6982
NIST: 8.1302
METEOR: 0.4752
ROUGE_L: 0.7232
CIDEr: 2.1725
