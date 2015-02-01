## 第1層 データマイニング手法からNeural Networkへ

この章では、読者が慣れ親しんだデータマイニング手法を、ニューラルネットワークに置き換えます。  
いきなりパーセプトロンがどうだよりはわかりやすかろうと。  

とりあえずKaggleの手書き文字認識をします。  
http://www.kaggle.com/c/digit-recognizer


### 1.1 Logistic Regression

データを読み込んで、trainからvalidデータ切り出して、モデルを10個作ります。  
つまり、0かそうでないかを判別するモデル、1かそうでないかを判別するモデル、・・・  

##### データ準備
まずはデータを読み込んで、使いやすい形に準備します。  

```
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

Arrayに変換しているのは半分趣味です。  
image(行列)で画像っぽくPlotに出力されます。適宜チェックしてください。  

##### データの圧縮
次に、28×28を説明変数にすると多すぎる気がするので、7×7に減らします。  
ついでにバイナリ(0/1)にしちゃいましょう。  
4×4の大きさのブロックに分割し(7×7個できます)、各ブロックの中に127より大きな値があれば1を、
そうでなければ0を出力とします。  

```
# resize(7*7)
# use max here
train_array_49 <- array(0,dim=c(28000,7,7))
for (i in seq(7)){
  for (j in seq(7)){
    for (row in seq(28000)){
      train_array_49[row,i,j] <- ifelse(max(train_array[row,((4*i-3):(4*i)),((4*j-3):(4*j))]) > 127,1,0)
    }
  }
}

# check
# image(train_array_49[1,,])
# image(train_array_49[2,,])
```

##### モデル構築
Rのデータフレームに変換して、glmで作ります。

```
train49 <- as.data.frame(cbind(train_label,array(train_array_49,dim=c(28000,49))))
# check
# image(matrix(as.numeric(train49[1,-1]),ncol=7))
# image(matrix(as.numeric(train49[2,-1]),ncol=7))


# logistic regression
logit0 <- glm((train_label==0)~.,data=train49)
logit1 <- glm((train_label==1)~.,data=train49)
logit2 <- glm((train_label==2)~.,data=train49)
logit3 <- glm((train_label==3)~.,data=train49)
logit4 <- glm((train_label==4)~.,data=train49)
logit5 <- glm((train_label==5)~.,data=train49)
logit6 <- glm((train_label==6)~.,data=train49)
logit7 <- glm((train_label==7)~.,data=train49)
logit8 <- glm((train_label==8)~.,data=train49)
logit9 <- glm((train_label==9)~.,data=train49)
```

