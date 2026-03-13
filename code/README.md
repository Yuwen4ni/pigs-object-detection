

本次物件偵測的作業是在colab環境上執行的，ipynb檔上有標記目綠，按照順序執行即可：

執行步驟如下：

(1)  由於我所使用的model 是YOLO ，所以請先下載相關套件

!pip install ultralytics -qq # YOLO套件

(2) 下載Kaggle競賽資料集

(3) 將現有的資料中的訓練集，再次切分成子訓練集跟子驗證集，並且標籤轉換成 YOLO 支援的格式

(4) 訓練model，在目錄上可以到那邊執行訓練，具體如下：

model = YOLO("yolov10l.yaml")
model.train(
    data="/content/drive/MyDrive/taica_hw1/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    optimizer="adamw",
    cos_lr=True,
    patience=20,
    amp=True,
    weight_decay=0.05,
    pretrained=False,
    project="/content/drive/MyDrive/taica_hw1/runs",
)

(5) 同理，預測model也在目錄上那邊執行預測，具體如下：

model = YOLO("/content/drive/MyDrive/taica_hw1/runs/train/weights/best.pt")

model.predict(
    source="/content/drive/MyDrive/taica_hw1/test/images",
    imgsz=640,
    conf=0.1,
    save=True,
    save_txt=True,
    save_conf=True,
    project="/content/drive/MyDrive/taica_hw1/preds",
    name="test"
)

(6) 最後將預測的結果轉成 Kaggle 競賽的格式，即可完成此次的作業



