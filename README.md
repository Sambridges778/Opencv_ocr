# Opencv_ocr文字检测及识别
***
:house: 本次项目主要运用`Python`的`Opencv`和谷歌开源软件`tesseract`进行对文本的检测和识别，利用`Opencv`中带有的East扩展包实现对文本的检测框选，再利用`tesseract`实现对文本文字的具体识别，在`Python`终端输出文本识别效果。为了便于更容易操作，我采用`PyQt5`设计具体的GUI，这样可以不用再去看烦人:rage:的代码。


# 安装
***
### :earth_asia: 配置环境变量(Windows)
>当你下载好`Tesseract`后（安装路径不要有中文）,将你的`Tesseract`的路径添加到环境变量**Path**中,点击保存,退出。再次打开`Tesseract`的路径,找到**tessdata**,复制此路径，设置新的环境变量**TESSDATA_PREFIX**,点击保存。:bell:安装后的验证方法，Cmd 窗口，使用命令tesseract -v验证。


### :triangular_ruler:**IDE: Pycharm** 
### :sunny:**虚拟环境：** Anaconda
### :star:该项目使用的Python扩展包: `pytesseract` 、`tesseract` 、`Opencv4.5`、`imutils`、`numpy1.16`、`PIL`、`PyQt5`

  `pip install '扩展包'`

  Tesseract软件的下载地址(https://digi.bib.uni-mannheim.de/tesseract/)
  
  :bell:简体字识别包下载地址(https://raw.githubusercontent.com/tesseract-ocr/tessdata/4.00/chi_sim.traineddata)

# 使用
