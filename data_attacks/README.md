# Data Attacks

Notebooks exploring data poisoning and model-integrity attacks against ML classifiers, built for HTB AI Red Team training. Educational/lab context only.

## Notebooks

- **label_flipping.ipynb** / **label-flipping-student-template.ipynb** — Baseline label-flipping poisoning attack on a synthetic binary classification dataset (Logistic Regression), with an evaluator-facing student template.
- **targeted-label-student-template.ipynb** — Targeted label-flipping attack that pushes samples from one specific class into another chosen class, rather than flipping arbitrary labels.
- **clean_label_attacks.ipynb** / **clean-label-student-template.ipynb** — Clean-label poisoning: perturbing feature-space neighbors of a target point so it gets misclassified, without altering any labels.
- **skills-assessment-student-template.ipynb** — Combined assessment notebook applying a poisoning strategy end-to-end and submitting results to an evaluator API.
- **trojan_attack.ipynb** / **student_trojan_mnist.ipynb** — Backdoor/trojan attacks: training a CNN (GTSRB traffic signs, MNIST) that behaves normally on clean inputs but misclassifies inputs containing a trigger patch.
- **steganography_attack.ipynb** — Embeds an arbitrary payload inside a PyTorch model's floating-point weights via LSB steganography, demonstrating model-file-based payload smuggling (e.g. supply-chain risk of downloading untrusted checkpoints).

## Data / artifacts

`.npz`, `.pth`, and `.joblib` files alongside the notebooks are generated datasets and trained model checkpoints (clean and poisoned/trojaned) consumed by the corresponding notebook. They are reproducible from the notebook code and not meant to be hand-edited.

The `linkedin_*` files are writeups/screenshots for sharing results externally.
