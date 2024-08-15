# import urllib

from random import randint
import re
from time import sleep
from urllib.request import Request, urlopen
from warnings import warn


url_base = 'https://www.stickpng.com/cat/'


# def download_image(url, filename):
#     with urlopen(url) as response, open(filename, 'wb') as out_file:
#         data = response.read()  # a `bytes` object
#         out_file.write(data)


req = Request(
    url=url_base,
    headers={'User-Agent': 'Mozilla/5.0'}
)
page = urlopen(req, timeout=10).read().decode('utf8')

# page_num
# if pagenum_cat_curr == pagenum_cat_last:

queue = []
queue.extend(re.findall(r'<a\s*class=image\s*href=/cat/([^?>]*)', page))




# categories = re.findall(r'<a\s*class=image\s*href=/cat/([^?>]*)', page)
#
# subcategories = {}
# image_urls = []
# images = {}
# # dict
# # 'name'
# # 'nickname'
# # 'catgory'
# # 'subcategory'
# # 'code'
# # 'display_url'
# # 'url'
#
# max_pages = 100
# for category in categories:
#     subcategories[category] = []
#
#     pagenum_cat_curr = 0
#     pagenum_cat_last = 0
#     for p in range(max_pages):
#         pagenum_cat_curr = pagenum_cat_last + 1
#
#         sleep(randint(3, 30))
#
#         if url_base.endswith('/'):
#             url_catpage = url_base + category + '?page=' + str(pagenum_cat_curr)
#         else:
#             url_catpage = url_base + '/' + category + '?page=' + str(pagenum_cat_curr)
#
#         req = Request(
#             url=url_catpage,
#             headers={'User-Agent': 'Mozilla/5.0'}
#         )
#         page_cat = urlopen(req, timeout=10).read().decode('utf8')
#
#         if pagenum_cat_curr > 1:
#             pagenum_cat_check = re.findall(r'<title[^>]*>[^<>]*\s*Page\s*([0-9]+)\s*[^<>]*</title[^>]*>', page_cat)
#             if pagenum_cat_check:
#                 if pagenum_cat_check[0].isnumeric():
#                     pagenum_cat_check = float(pagenum_cat_check[0])
#                     if pagenum_cat_check.is_integer():
#                         pagenum_cat_check = int(pagenum_cat_check)
#                         pagenum_cat_curr = pagenum_cat_check
#         if pagenum_cat_curr == pagenum_cat_last:
#             break
#
#         subcategories[category].extend(re.findall(r'<a\s*class=image\s*href=/cat/' + category + '/([^?>]*)', page_cat))
#
#         pagenum_cat_last = pagenum_cat_curr
#         print('.', end='', flush=True)
#
#     pagenum_subcat_curr = 0
#     pagenum_subcat_last = 0
#     for subcategory in subcategories[category]:
#         # images[subcategory] = []
#
#         for p in range(max_pages):
#             pagenum_subcat_curr = pagenum_subcat_last + 1
#
#             sleep(randint(3, 30))
#
#             if url_base.endswith('/'):
#                 url_subcatpage = url_base + category + '/' + subcategory + '?page=' + str(pagenum_subcat_curr)
#             else:
#                 url_subcatpage = url_base + '/' + category + '/' + subcategory + '?page=' + str(pagenum_subcat_curr)
#
#             req = Request(
#                 url=url_subcatpage,
#                 headers={'User-Agent': 'Mozilla/5.0'}
#             )
#             page_subcat = urlopen(req, timeout=10).read().decode('utf8')
#
#             if pagenum_subcat_curr > 1:
#                 pagenum_subcat_check = re.findall(r'<title[^>]*>[^<>]*\s*Page\s*([0-9]+)\s*[^<>]*</title[^>]*>', page_subcat)
#                 if pagenum_subcat_check:
#                     if pagenum_subcat_check[0].isnumeric():
#                         pagenum_subcat_check = float(pagenum_subcat_check[0])
#                         if pagenum_subcat_check.is_integer():
#                             pagenum_subcat_check = int(pagenum_subcat_check)
#                             pagenum_subcat_curr = pagenum_subcat_check
#             if pagenum_subcat_curr == pagenum_subcat_last:
#                 break
#
#             # r'<a class="image pattern" href=/img/animals/elephants/elephant-back-view-close-up>'
#             image_urls.extend(re.findall(r'<a\s*class="image\s*pattern"\s*href=([^>]+)', page_subcat))
#             # images[subcategory].extend(re.findall(r'<a\s*class=image\s*href=/img/([^?>]*)', page_subcat))
#             # images[image_num]['display_url'] = re.findall(r'<a\s*class=image\s*href=/img/([^?>]*)', page_subcat)
#             # r'https://www.stickpng.com/img/animals/elephants/mother-and-baby-elephant'
#
#
# # first, build a list of all image display urls... all info is on those pages
#
#             pagenum_subcat_last = pagenum_subcat_curr
#             print('-', end='', flush=True)
#
#
#     if re.match(r'<a\s*href=/img/download/([a-zA-Z0-9]+)\s*class=button><i></i>Download</a>', page_subcat):
#         images[subcategory] = re.findall(r'<a\s*class=image\s*href=/img/([^?>]*)', page_subcat)
#
#
#     '<a href=/img/download/580b57fbd9996e24bc43bdf2 class=button><i></i>Download</a>'
#     '<a href=/cat>Categories</a><a href=/cat/bots-and-robots?page=1>Bots and Robots</a><span>Robot Warm Up</span></div>'
#
#     'https://assets.stickpng.com/images/580b57fbd9996e24bc43bdf2.png'
#
#
#
