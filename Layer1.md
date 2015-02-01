## 第1層 データマイニング手法からNeural Networkへ

この章では、読者が慣れ親しんだデータマイニング手法を、ニューラルネットワークに置き換えます。  
いきなりパーセプトロンがどうだよりはわかりやすかろうと。  

とりあえずKaggleの手書き文字認識をします。  
http://www.kaggle.com/c/digit-recognizer


### 1.1 Logistic Regression

データを読み込んで、trainからvalidデータ切り出して、モデルを10個作ります。  
0かそうでないかを判別するモデル、1かそうでないかを判別するモデル、・・・  

```R:setdata.R
# setwd("/Users/KeiHarada/Documents/digit")

traindata <- read.csv("ORG/train.csv")
testdata <- read.csv("ORG/test.csv")
train_label <- traindata$label[1:28000]
valid_label <- traindata$label[28001:42000]

# transform into array
train_array <- array(as.matrix(traindata[1:28000,-1]),dim=c(28000,28,28))
valid_array <- array(as.matrix(traindata[28001:42000,-1]),dim=c(14000,28,28))

# check
# image(train_array[1,,])
# image(valid_array[1,,])
```


28×28を説明変数にすると多すぎる気がするので、途中で7×7に減らします。  





