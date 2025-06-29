# Information-content-security-python-crawler
# 人民网社会频道新闻分析系统 README

## 项目简介

本项目是一个基于 Python 的人民网社会频道新闻爬取与分析系统，能够自动爬取人民网社会频道的新闻文章，进行文本分析、情感分析和聚类分析，并生成可视化报告。

## 功能特点

  * **新闻爬取** ：从人民网社会频道首页获取有效新闻链接，并爬取每篇文章的标题、正文内容、发布时间及图片等信息。
  * **文本分析** ：对新闻文本进行预处理（清洗、分词、去停用词），通过 TF-IDF 算法提取关键词，利用 KMeans 算法对文章进行聚类分析，发现不同话题。
  * **情感分析** ：使用 SnowNLP 库对新闻内容进行情感倾向性分析，判断每篇文章的情感倾向（积极、中性、消极）并给出情感得分。
  * **可视化报告** ：生成包含话题分布饼图、情感分布饼图、新闻关键词云图和新闻聚类分布图的 HTML 报告，直观展示新闻分析结果。同时，报告中还会列出每篇新闻的详细信息。

## 环境要求

  * **Python** ：3.6 或以上版本
  * **库依赖** ：
    * requests：用于发送 HTTP 请求爬取网页
    * beautifulsoup4：用于解析 HTML 页面
    * numpy、scikit-learn：用于文本向量化和聚类分析
    * jieba、snownlp：用于中文文本处理和情感分析
    * matplotlib、wordcloud：用于生成可视化图表
    * flask：用于构建简单的 Web 服务（可选）

## 安装步骤

  1. **克隆项目仓库** ：将项目代码克隆到本地。
  2. **安装依赖库** ：在项目根目录下，运行 `pip install -r requirements.txt` 命令安装所需的库依赖。

## 使用方法

  1. **运行脚本** ：在项目根目录下，运行 `python main.py` 命令启动新闻分析系统。
  2. **查看结果** ：脚本运行完成后，会在项目目录下生成一个名为 “people_news” 的输出目录，其中包含爬取的新闻文本文件、图片文件、分析结果的 JSON 文件以及生成的 HTML 报告。打开 HTML 报告文件，即可在浏览器中查看详细的分析结果。

## 代码结构

  * **配置**参数 ：在脚本开头定义了基本配置，如目标 URL、输出目录、请求头信息等，可根据实际需求进行修改。
  * **爬虫函数** ：包括获取频道首页有效链接、解析文章页面、下载图片、保存文本数据等功能函数，实现新闻的爬取和存储。
  * **文本分析函数** ：包含文本预处理、向量化、聚类分析、情感分析等功能函数，用于对新闻文本进行深度分析。
  * **可视化函数** ：提供生成词云图、聚类分布图、话题分布饼图、情感分布饼图等功能函数，实现分析结果的可视化展示。
  * **HTML 报告生成函数** ：将分析结果整合成 HTML 格式的报告，方便用户查看和分享。

## 注意事项

  * 在运行脚本前，请确保已安装必要的库依赖，并且网络连接正常，以便能够成功爬取人民网的新闻数据。
  * 爬取过程中应遵循人民网的相关使用条款和机器人协议，避免对网站服务器造成过大压力。
  * 如果需要对爬取的新闻数量、分析的深度或可视化效果进行调整，可以修改相应的配置参数或函数逻辑。

## 扩展与维护

  * 可根据实际需求对爬虫功能进行扩展，例如增加对其他新闻频道或网站的支持。
  * 可进一步优化文本分析算法，提高分析的准确性和效率。
  * 如果在使用过程中遇到问题或发现代码缺陷，欢迎提交 issue 或贡献代码进行改进。
