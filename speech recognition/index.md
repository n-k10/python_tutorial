# 音声認識超入門

## 目次

1. [音声認識とは？](./docs/01intro.md)
2. [音声認識の基礎知識](./docs/02basic.md)
3. [音声処理の基礎と特徴量抽出](./docs/03features.md)
4. [DPマッチングによる音声認識](./docs/04dp.md)
5. [GMM-HMMによる音声認識](./docs/05GMM-HMM.md)
6. [DNN-HMMによる音声認識](./docs/06DNN-HMM.md)
7. [End-to-Endモデルによる連続音声認識](./docs/07END2END.md)

## 下準備

本資料では Python によるプログラミングを行うため，Pythonが入っていない場合はインストールしてください．  
また，下記のライブラリをインストールしておくこと．

```sh
pip install numpy
pip install matplotlib
pip install sox
pip install pyyaml
```

また，機械学習を実際に行う関係で GPU および CUDA を使用します．  
そのため，GPU が入っていない環境の場合は Google Colaboratory を使用してください．  
これに加えて，[Pytorch](https://pytorch.org) をインストールしてください．

## 参考

- [ソースコードや章立ての参考](https://github.com/ry-takashima/python_asr/tree/main)
- [イラスト付きで(大学生が見て)分かりやすい](https://www.docswell.com/user/kyoto-kaira?page=5)

[-> 01 音声認識とは？](./docs/01intro.md)
