# from urllib.request import urlopen
# import re
#
# # 如果是中文网站需要使用UTF-8编码方式来解码
# html = urlopen(
#     "https://mofanpy.com/static/scraping/basic-structure.html"
# ).read().decode('utf-8')
# print(html)
#
# res = re.findall(r"<title>(.+?)</title>", html)
# print("\nPage title is: ", res[0])
# print(res)
#####################
# from bs4 import BeautifulSoup
# from urllib.request import urlopen
#
# # if has Chinese, apply decode()
# html = urlopen("https://mofanpy.com/static/scraping/basic-structure.html").read().decode('utf-8')
# print(html)
#
# soup = BeautifulSoup(html, features='lxml')
# all_href = soup.find_all('a')
# print(all_href)
# for i in all_href:
#     print(i['href'])