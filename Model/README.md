# GIGA-Piano XL Pre-Trained Model

***

## Trained upon ~420k Solo Piano MIDI compositions for ~1.2 epochs/~138k steps/~84 hours @ 2048 seq_len/8 batches on dual A6000 48GB GPUs

***

## Model Stats

### FLoss 1.2590 CE
### VLoss 0.6118 CE
### Acc: 0.8032 CE

***

### Encoding info:
### [dTime(0-126), Duration(1-126)+128, MIDI Pitch(1-126)+256, MIDI Velocity(1-126)+384]
### Compositions separator/Intro/Zero sequence: [126, 126+128, 0+256, 0+384]

***

![GIGA-Piano-XL-Training-Loss-Graph](https://user-images.githubusercontent.com/56325539/195238204-431763f6-3c03-4b05-81f0-96a60815243f.png)

***

![GIGA-Piano-XL-Positional-Embeddings-Plot](https://user-images.githubusercontent.com/56325539/195238241-e5991998-b55a-496a-a191-193485b21309.png)

***

### Project Los Angeles
### Tegridy Code 2022
