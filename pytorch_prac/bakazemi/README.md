#バカゼミ用
## ロリキャラorロリキャラでないの２値分類とちっぱいor巨乳の２値分類の実装をする
## ファイルの説明
+ detect.py : opencvアニメ顔検出のチュートリアル的なもの
+ filelist.py : pythonのosを利用してそのディレクトリ内のファイル名リストを出力するプログラム
+ test_detect.py : detect.pyを改良し、指定ディレクトリ内のすべてのファイルに対してアニメ顔検出を行い、アニメ顔のみの画像データを指定ディレクトリに出力する
+ rori_or_other.py : pytorchを利用したロリかロリじゃないかの分類学習=>fine_turningを利用していないもの
+ rori_or_other_fine.py : 上記のものにfine_turningを適応したものなお元のモデルは**resnet18**
+ weight.pth : 上記のrori_or_other_fine.pyで学習した重みを保存する
+ estimate.py : 引数に与えたフォルダにある画像データに顔検出と学習済みモデルを利用した推定を行い予測をする
