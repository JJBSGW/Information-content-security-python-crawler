import os
import time
import hashlib
import re
import json
import requests
import numpy as np
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import jieba
import jieba.analyse
from snownlp import SnowNLP
import matplotlib.font_manager as fm
import matplotlib as mpl

# ========== 配置参数 ==========
BASE_URL = "http://society.people.com.cn/GB/index.html"
OUTPUT_DIR = "people_news"
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
DELAY = 2  # 爬取间隔

font_path = 'C:/Users/Lucky/Desktop/新建文件夹/PYCProject/Pa/SimHei.ttf'  # 根据实际路径修改
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False


# ========== 爬虫函数 ==========
def create_dirs():
    """创建存储目录"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)


def is_valid_url(url):
    """验证是否为本站有效链接"""
    parsed = urlparse(url)
    return parsed.netloc == 'society.people.com.cn' and 'n1' in url


def get_links():
    """获取频道首页所有有效链接"""
    response = requests.get(BASE_URL, headers=HEADERS)
    soup = BeautifulSoup(response.text, 'lxml')

    all_links = []
    for a in soup.select('a[href]'):
        full_url = urljoin(BASE_URL, a['href'])
        if is_valid_url(full_url):
            all_links.append(full_url)
    return list(set(all_links))  # 去重


def parse_article(url):
    """解析文章页面"""
    time.sleep(DELAY)
    response = requests.get(url, headers=HEADERS)

    # 显式指定网页编码为UTF-8
    response.encoding = 'UTF-8'

    soup = BeautifulSoup(response.text, 'html.parser')

    # 提取标题
    title = soup.find('h1').text.strip() if soup.find('h1') else '无标题'

    # 提取正文内容
    content_div = soup.find('div', class_='rm_txt_con')
    content = '\n'.join([p.text.strip() for p in content_div.find_all('p')]) if content_div else ""

    # 提取发布时间
    time_tag = soup.find('div', class_='channel')
    publish_time = ""
    if time_tag:
        time_str = time_tag.text
        match = re.search(r'\d{4}年\d{2}月\d{2}日\d{2}:\d{2}', time_str)
        if match:
            publish_time = match.group()

    # 提取并下载图片
    img_urls = []
    for img in soup.find_all('img'):
        img_url = urljoin(url, img.get('src'))
        if img_url.endswith(('.jpg', '.jpeg', '.png')):
            img_name = os.path.join(IMG_DIR, f"{hashlib.md5(img_url.encode()).hexdigest()}.jpg")
            download_image(img_url, img_name)
            img_urls.append(img_url)

    return {
        'title': title,
        'content': content,
        'url': url,
        'time': publish_time,
        'images': img_urls
    }


def download_image(url, save_path):
    """下载图片"""
    try:
        response = requests.get(url, stream=True, headers=HEADERS)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
    except Exception as e:
        print(f"下载图片失败：{url} - {str(e)}")


def save_text(data):
    """保存文本数据"""
    filename = hashlib.md5(data['url'].encode()).hexdigest() + ".txt"
    filepath = os.path.join(OUTPUT_DIR, filename)

    # 使用UTF-8编码保存
    with open(filepath, 'w', encoding='UTF-8') as f:
        f.write(f"标题：{data['title']}\n")
        f.write(f"发布时间：{data['time']}\n")
        f.write(f"原文链接：{data['url']}\n")
        f.write(f"\n正文内容：\n{data['content']}\n")
        f.write(f"\n相关图片：\n" + '\n'.join(data['images']))


# ========== 文本分析函数 ==========
def preprocess_text(text):
    """文本预处理：清洗、分词、去停用词"""
    # 清洗文本
    text = re.sub(r'\s+', ' ', text)  # 去除多余空格
    text = re.sub(r'[^\w\u4e00-\u9fff]', ' ', text)  # 去除非中文字符

    # 加载停用词表
    stopwords = set()
    try:
        # 尝试加载停用词表
        with open('stopwords.txt', 'r', encoding='utf-8') as f:
            for line in f:
                stopwords.add(line.strip())
    except:
        # 如果没有停用词表，使用默认列表
        stopwords = {'的', '了', '在', '是', '和', '就', '都', '而', '及', '与', '这', '那', '你', '我', '他'}

    # 分词和过滤
    words = jieba.cut(text)
    return ' '.join([word for word in words if word not in stopwords and len(word) > 1])


def vectorize_texts(texts):
    """文本向量化（TF-IDF）"""
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer


def cluster_articles(articles, n_clusters=5):
    """聚类分析（话题发现）"""
    # 预处理文本
    processed_texts = [preprocess_text(a['content']) for a in articles]

    # 向量化
    tfidf_matrix, vectorizer = vectorize_texts(processed_texts)

    # 自动确定聚类数量（最多不超过文章数量的1/3）
    n_clusters = min(n_clusters, max(2, len(articles) // 3))

    # KMeans聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(tfidf_matrix)

    # 获取每个聚类的关键词
    cluster_keywords = {}
    for i in range(n_clusters):
        cluster_indices = np.where(clusters == i)[0]
        cluster_text = ' '.join([processed_texts[idx] for idx in cluster_indices])
        keywords = jieba.analyse.extract_tags(cluster_text, topK=10)
        cluster_keywords[i] = keywords

    return clusters, cluster_keywords


def sentiment_analysis(articles):
    """情感倾向性分析"""
    results = []
    for article in articles:
        try:
            s = SnowNLP(article['content'])
            sentiment = s.sentiments  # 情感得分 (0-1)
            sentiment_label = "积极" if sentiment > 0.6 else "消极" if sentiment < 0.4 else "中性"
            results.append({
                'title': article['title'],
                'sentiment': sentiment,
                'label': sentiment_label
            })
        except:
            # 如果分析失败，使用中性作为默认值
            results.append({
                'title': article['title'],
                'sentiment': 0.5,
                'label': "中性"
            })
    return results


def visualize_clusters(articles, clusters):
    """聚类可视化（t-SNE降维）"""
    if len(articles) < 5:
        print("文章数量太少，跳过聚类可视化")
        return

    processed_texts = [preprocess_text(a['content']) for a in articles]
    tfidf_matrix, _ = vectorize_texts(processed_texts)

    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    points = tsne.fit_transform(tfidf_matrix.toarray())

    # 绘制散点图
    plt.figure(figsize=(12, 8))
    for i in range(max(clusters) + 1):
        cluster_points = points[clusters == i]
        if len(cluster_points) > 0:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'话题 {i + 1}')

    plt.title('新闻话题聚类分布')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'clusters.png'))
    plt.close()


def generate_wordcloud(articles, filename):
    """生成词云图"""
    if len(articles) == 0:
        print("没有文章内容，跳过生成词云")
        return

    all_text = ' '.join([preprocess_text(a['content']) for a in articles])

    if len(all_text.strip()) == 0:
        print("文本内容为空，跳过生成词云")
        return

    # 获取中文字体路径
    font_path = None
    try:
        for font in fm.findSystemFonts():
            if 'simhei' in font.lower() or 'simsun' in font.lower() or 'msyh' in font.lower():
                font_path = font
                break
    except:
        pass

    # 如果没有找到系统字体，尝试默认
    if not font_path:
        font_path = 'simhei.ttf'

    wc = WordCloud(
        font_path=font_path,
        background_color='white',
        width=800,
        height=600,
        max_words=200
    ).generate(all_text)

    wc.to_file(os.path.join(OUTPUT_DIR, filename))


# ========== HTML报告生成函数 ==========
def generate_html_report(articles, cluster_keywords, cluster_counts, sentiment_counts):
    """生成静态HTML报告"""
    report_path = os.path.join(OUTPUT_DIR, 'report.html')

    # 生成图表
    cluster_chart_path = os.path.join(OUTPUT_DIR, 'cluster_chart.png')
    sentiment_chart_path = os.path.join(OUTPUT_DIR, 'sentiment_chart.png')

    generate_cluster_chart(cluster_counts, cluster_chart_path)
    generate_sentiment_chart(sentiment_counts, sentiment_chart_path)

    # 构建HTML内容
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>新闻分析报告 - 人民网社会频道</title>
        <style>
            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
                font-family: 'Microsoft YaHei', Arial, sans-serif;
            }}
            body {{
                background-color: #f8f9fa;
                color: #333;
                line-height: 1.6;
                padding: 20px;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
                padding: 30px;
            }}
            header {{
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 1px solid #eee;
            }}
            h1 {{
                color: #1a365d;
                margin-bottom: 10px;
            }}
            .subtitle {{
                color: #666;
                font-size: 18px;
            }}
            .stats {{
                display: flex;
                justify-content: space-around;
                margin: 25px 0;
                text-align: center;
            }}
            .stat-card {{
                background: #f1f5f9;
                border-radius: 8px;
                padding: 20px;
                flex: 1;
                margin: 0 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            }}
            .stat-value {{
                font-size: 32px;
                font-weight: bold;
                color: #1a365d;
            }}
            .stat-label {{
                color: #64748b;
                font-size: 16px;
            }}
            .charts {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin-bottom: 30px;
            }}
            .chart-container {{
                background: white;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                padding: 20px;
                flex: 1;
                min-width: 300px;
            }}
            .chart-title {{
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 15px;
                color: #1a365d;
                text-align: center;
            }}
            .chart-img {{
                width: 100%;
                height: auto;
                border-radius: 5px;
            }}
            .section {{
                margin-bottom: 30px;
            }}
            .section-title {{
                font-size: 22px;
                color: #1a365d;
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 2px solid #e2e8f0;
            }}
            .keywords {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-bottom: 20px;
            }}
            .keyword {{
                background: #e0f2fe;
                padding: 8px 15px;
                border-radius: 20px;
                font-size: 14px;
                color: #0369a1;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }}
            .news-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }}
            .news-table th, .news-table td {{
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #e2e8f0;
            }}
            .news-table th {{
                background-color: #f1f5f9;
                font-weight: bold;
                color: #334155;
            }}
            .news-table tr:hover {{
                background-color: #f8fafc;
            }}
            .sentiment-positive {{
                color: #10b981;
                font-weight: bold;
            }}
            .sentiment-neutral {{
                color: #64748b;
                font-weight: bold;
            }}
            .sentiment-negative {{
                color: #ef4444;
                font-weight: bold;
            }}
            .wordcloud-container {{
                text-align: center;
                margin: 30px 0;
            }}
            footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #eee;
                color: #64748b;
                font-size: 14px;
            }}
            .cluster-section {{
                background: #f8fafc;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }}
            .cluster-header {{
                display: flex;
                align-items: center;
                margin-bottom: 15px;
            }}
            .cluster-id {{
                background: #1a365d;
                color: white;
                width: 36px;
                height: 36px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                margin-right: 15px;
                flex-shrink: 0;
            }}
            .cluster-title {{
                font-size: 20px;
                color: #1a365d;
            }}
            .cluster-count {{
                margin-left: auto;
                background: #dbeafe;
                padding: 5px 15px;
                border-radius: 20px;
                font-weight: bold;
                color: #1d4ed8;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>人民网社会频道新闻分析报告</h1>
                <p class="subtitle">基于网络爬虫与文本分析技术</p>
            </header>

            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value">{len(articles)}</div>
                    <div class="stat-label">分析文章数</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(cluster_counts)}</div>
                    <div class="stat-label">发现话题数</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{sentiment_counts['积极']}</div>
                    <div class="stat-label">积极文章</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{sentiment_counts['消极']}</div>
                    <div class="stat-label">消极文章</div>
                </div>
            </div>

            <div class="charts">
                <div class="chart-container">
                    <div class="chart-title">话题分布</div>
                    <img src="cluster_chart.png" alt="话题分布图" class="chart-img">
                </div>
                <div class="chart-container">
                    <div class="chart-title">情感分布</div>
                    <img src="sentiment_chart.png" alt="情感分布图" class="chart-img">
                </div>
            </div>

            <div class="wordcloud-container">
                <div class="chart-title">新闻关键词云</div>
                <img src="wordcloud.png" alt="关键词云图" style="max-width: 80%;">
            </div>

            <div class="section">
                <h2 class="section-title">话题关键词分析</h2>
                {"".join([generate_cluster_section(i, keywords, count)
                          for i, (cluster_id, keywords) in enumerate(cluster_keywords.items())
                          for count in [cluster_counts[cluster_id]]])}
            </div>

            <div class="section">
                <h2 class="section-title">新闻聚类分布图</h2>
                <img src="clusters.png" alt="新闻聚类分布" style="width: 100%;">
            </div>

            <div class="section">
                <h2 class="section-title">新闻列表</h2>
                <table class="news-table">
                    <thead>
                        <tr>
                            <th>标题</th>
                            <th>话题</th>
                            <th>情感倾向</th>
                            <th>情感值</th>
                            <th>发布时间</th>
                        </tr>
                    </thead>
                    <tbody>
 {                       "".join([generate_article_row(article, cluster_keywords) for article in articles])}
                    </tbody>
                </table>
            </div>

            <footer>
                <p>生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')} | 共分析 {len(articles)} 篇文章</p>
                <p>© 2023 新闻分析系统 | 基于Python网络爬虫与文本分析技术</p>
            </footer>
        </div>
    </body>
    </html>
    """

    # 保存HTML文件
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"报告已生成：{os.path.abspath(report_path)}")
    print(f"请用浏览器打开该文件查看完整报告")


def generate_cluster_section(index, keywords, count):
    """生成话题部分HTML"""
    color_classes = ["#dbeafe", "#ffe4e6", "#dcfce7", "#fef3c7", "#ede9fe"]
    bg_color = color_classes[index % len(color_classes)]

    # 生成话题名称（使用关键词的前几个词）
    topic_name = "、".join(keywords[:3])

    return f"""
    <div class="cluster-section">
        <div class="cluster-header">
            <div class="cluster-id">{index + 1}</div>
            <div class="cluster-title">话题 {index + 1}: {topic_name}</div>
            <div class="cluster-count">{count} 篇文章</div>
        </div>
        <div class="keywords">
            {"".join([f'<div class="keyword">{kw}</div>' for kw in keywords])}
        </div>
    </div>
    """


def generate_article_row(article, cluster_keywords):
    """生成新闻行HTML"""
    sentiment_class = f"sentiment-{article['sentiment_label']}"
    topic_keywords = cluster_keywords.get(article['cluster'], [])
    topic_name = "、".join(topic_keywords[:3]) if topic_keywords else f"话题 {article['cluster'] + 1}"

    return f"""
    <tr>
        <td><a href="{article['url']}" target="_blank">{article['title']}</a></td>
        <td>{topic_name}</td>
        <td class="{sentiment_class}">{article['sentiment_label']}</td>
        <td>{article['sentiment']:.2f}</td>
        <td>{article['time']}</td>
    </tr>
    """


def generate_cluster_chart(cluster_counts, save_path):
    """生成话题分布饼图"""
    plt.figure(figsize=(8, 8))

    # 准备数据
    labels = [f'话题{cluster_id + 1} ({count})' for cluster_id, count in cluster_counts.items()]
    sizes = list(cluster_counts.values())
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    # 绘制饼图
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('新闻话题分布', fontsize=16)
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()


def generate_sentiment_chart(sentiment_counts, save_path):
    """生成情感分布饼图"""
    plt.figure(figsize=(8, 8))

    # 准备数据
    labels = ['积极', '中性', '消极']
    sizes = [sentiment_counts['积极'], sentiment_counts['中性'], sentiment_counts['消极']]
    colors = ['#10b981', '#64748b', '#ef4444']

    # 绘制饼图
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('新闻情感分布', fontsize=16)
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()


# ========== 主函数 ==========
def main():
    # 0. 准备工作
    create_dirs()
    start_time = time.time()

    # 1. 爬取数据
    print("开始爬取人民网社会频道新闻...")
    links = get_links()
    print(f"发现有效文章链接：{len(links)}个")

    articles = []
    for i, url in enumerate(links):
        try:
            print(f"正在处理 [{i + 1}/{len(links)}]: {url}")
            article_data = parse_article(url)
            save_text(article_data)
            articles.append(article_data)
        except Exception as e:
            print(f"处理失败：{url} - {str(e)}")

    if not articles:
        print("未爬取到有效文章，程序终止")
        return

    # 2. 文本分析
    print("\n开始文本分析...")
    print(f"共爬取 {len(articles)} 篇文章")

    # 聚类分析
    clusters, cluster_keywords = cluster_articles(articles)
    for i, article in enumerate(articles):
        article['cluster'] = int(clusters[i])

    # 情感分析
    sentiment_results = sentiment_analysis(articles)
    for i, article in enumerate(articles):
        article['sentiment'] = sentiment_results[i]['sentiment']
        article['sentiment_label'] = sentiment_results[i]['label']

    # 统计计数
    cluster_counts = {}
    for article in articles:
        cluster_id = article['cluster']
        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

    sentiment_counts = {
        '积极': sum(1 for a in articles if a['sentiment_label'] == '积极'),
        '中性': sum(1 for a in articles if a['sentiment_label'] == '中性'),
        '消极': sum(1 for a in articles if a['sentiment_label'] == '消极')
    }

    # 3. 生成可视化图表
    print("生成可视化图表...")
    visualize_clusters(articles, clusters)
    generate_wordcloud(articles, 'wordcloud.png')

    # 4. 生成HTML报告
    print("生成HTML报告...")
    generate_html_report(articles, cluster_keywords, cluster_counts, sentiment_counts)

    # 5. 保存分析结果
    with open(os.path.join(OUTPUT_DIR, 'analysis.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'articles': articles,
            'cluster_keywords': cluster_keywords,
            'cluster_counts': cluster_counts,
            'sentiment_counts': sentiment_counts
        }, f, ensure_ascii=False, indent=2)

    # 6. 完成
    elapsed = time.time() - start_time
    print(f"\n分析完成！耗时: {elapsed:.2f}秒")
    print(f"结果保存在: {os.path.abspath(OUTPUT_DIR)}")
    print(f"请打开 {os.path.join(OUTPUT_DIR, 'report.html')} 查看完整报告")


if __name__ == "__main__":
    main()