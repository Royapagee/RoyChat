# RoyChat

受LingChat启发，目标是制作一个完备的，基于AI的Galgame的系统。

此项目为个人学习项目，难免石山，有问题还请多多鞭策指正。

## 项目的文件结构

```txt
ROYCHAT
├─.venv                #虚拟环境
│  └─Scripts
├─backend              #后端
│  ├─app               #软件后端
│  │  ├─api              #通信程序
│  │  |  ├─aichat          #ai大模型适配器
│  │  |  └─websoket.py     #后端与前端通信
│  │  └─services         #后端内部功能处理
│  ├─config            #各个功能的配置文件
│  └─models            #本地模型的存放
│      ├─chat            #聊天 
│      ├─embedding       #嵌入模型
│      ├─reranker        #重排序模型
│      ├─emotion         #情绪分类
│      ├─stt             #语音转文本
│      └─tts             #文本转语音
├─data                 #数据
│  ├─assets            #静态文件
│  │  ├─background       #背景
│  │  └─music            #音乐
│  ├─characters        #角色
│  │  ├─qinling
│  │  └─roy
│  │      └─pic
│  └─story             #剧本/故事
├─frontend             #前端
│  └─src                 #前端静态文件
│      ├─api             #后端交流
│      ├─assets          #前端数据
│      │  ├─fonts        #字体
│      │  └─ui           #图标
│      └─components    #交互组件
├─runtime              #嵌入python环境
│  └─python
└─storage              #用户对话数据
    ├─chathistory      #对话历史（纯文本）
    ├─task             #标记的日子/主动对话
    └─vector           #RAG存储
```
