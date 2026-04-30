# Exp1 three-backbone baseline summary (SDNET2018-D, CE)

- **Params / FLOPs / FPS (fwd)**: from `complexity.py` (thop, `attention=none`, input 224×224), file `complexity.json`.
- **FPS (test) / Lat_test**: measured during the test phase in each run's `results.json` (GPU/driver dependent).

| Model | Params | FLOPs | FPS (fwd) | Lat_fwd (ms) | FPS (test) | Lat_test (ms) | Train loss (last) | Val loss (last) | Test loss | Acc | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet18 | 11.18M | 1.82G | 928.3 | 1.08 | 719.4 | 1.39 | 0.01486 | 0.395198 | 0.35673 | 0.9276 | 0.7249 |
| MobileNetV3 | 916.05K | 54.90M | 426.4 | 2.35 | 284.53 | 3.515 | 0.048793 | 0.437471 | 0.450299 | 0.9066 | 0.6667 |
| EfficientNet-B0 | 3.97M | 384.54M | 240.6 | 4.16 | 211.36 | 4.731 | 0.002866 | 0.576922 | 0.542606 | 0.9325 | 0.7406 |

## Full columns (Precision / Recall / best_val_F1)

| Model | Params | FLOPs | FPS_test | Latency_test_ms | Train_loss_final | Val_loss_final | Test_loss | Accuracy | Precision | Recall | F1 | Best_val_F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet18 | 11.18M | 1.82G | 719.4 | 1.39 | 0.01486 | 0.395198 | 0.35673 | 0.9276 | 0.8333 | 0.6414 | 0.7249 | 0.7236 |
| MobileNetV3 | 916.05K | 54.90M | 284.53 | 3.515 | 0.048793 | 0.437471 | 0.450299 | 0.9066 | 0.71 | 0.6283 | 0.6667 | 0.6772 |
| EfficientNet-B0 | 3.97M | 384.54M | 211.36 | 4.731 | 0.002866 | 0.576922 | 0.542606 | 0.9325 | 0.864 | 0.648 | 0.7406 | 0.7269 |
