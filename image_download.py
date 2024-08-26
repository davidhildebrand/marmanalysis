#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime, timezone
import json
import os
import pickle
from random import randint
import numpy as np
import re
import socket
from time import sleep
from urllib.request import Request, urlopen
from warnings import warn


url_base = 'https://www.stickpng.com/'
url_cat = url_base + 'cat/'
url_assets = 'https://assets.stickpng.com/images/'



skip_categories = ['icons-logos-emojis']
skip_subcategories = []


system_name = socket.gethostname()
if 'Galactica' in system_name:
    base_path = r'/Users/davidh/Data/Freiwald/ImageDatasets/stickpng'
    asset_path = base_path + os.path.sep + 'assets'
    tree_path = base_path + os.path.sep + 'category_tree'
    infosave_path = base_path + os.path.sep + 'info_saves'
elif 'marmostor' in system_name:
    base_path = r'/marmostor/DavidH/ImageDatasets/stickpng'
    asset_path = base_path + os.path.sep + r'assets'
    tree_path = base_path + os.path.sep + r'category_tree'
    infosave_path = base_path + os.path.sep + 'info_saves'
elif 'Obsidian' in system_name:
    base_path = r'F:\Data\stickpng'
    asset_path = base_path + os.path.sep + r'assets'
    tree_path = base_path + os.path.sep + r'category_tree'
    infosave_path = base_path + os.path.sep + 'info_saves'
elif 'Dobbin' in system_name:
    base_path = r'D:\Data\stickpng'
    asset_path = base_path + os.path.sep + r'assets'
    tree_path = base_path + os.path.sep + r'category_tree'
    infosave_path = base_path + os.path.sep + 'info_saves'
else:
    base_path = None
    asset_path = None
    tree_path = None
    infosave_path = None
    warn('Unknown system name, paths not set.')

os.makedirs(asset_path, exist_ok=True)
os.makedirs(tree_path, exist_ok=True)
os.makedirs(infosave_path, exist_ok=True)


def download_image(url, filename):
    dlreq = Request(url=url, headers={'User-Agent': 'Mozilla/5.0'})
    with urlopen(dlreq, timeout=10) as response, open(filename, 'wb') as out_file:
        data = response.read()
        out_file.write(data)


req = Request(url=url_cat, headers={'User-Agent': 'Mozilla/5.0'})
page = urlopen(req, timeout=10).read().decode('utf8')

queue = []
queue.extend(re.findall(r'<a\s*class=image\s*href=/cat/([^?>]*)', page))
# queue = np.unique(queue)

queue_sort_function = lambda x: (x.count('/'), x)
queue = sorted(queue, key=queue_sort_function)

image_info = {}

n_pages = 1
page_current = 1
n_total = 0
link_counter = 0
while len(queue) > 0:
    if len(queue) != len(np.unique(queue)):
        warn('Queue contains duplicate entries.')

    link = queue[0]
    print('Processing queued link: {} ({}/{})'.format(link, link_counter, len(queue)))
    while page_current <= n_pages:
        link_url = url_cat + link + '?page=' + str(page_current) \
            if url_cat.endswith('/') \
            else url_cat + '/' + link + '?page=' + str(page_current)
        req = Request(url=link_url, headers={'User-Agent': 'Mozilla/5.0'})
        sleep(randint(15, 80))
        link_page = urlopen(req, timeout=10).read().decode('utf8')

        # Update page counts using provided numbers.
        pattern_pageinfo = r'<section\s*id=pagination\s*data-pagination=\'([^\']*)\'></section>'
        page_info = re.search(pattern_pageinfo, link_page)
        if re.search(pattern_pageinfo, link_page) is not None:
            groups = re.search(pattern_pageinfo, link_page).groups()
            if len(groups) != 1:
                warn('Multiple page information entries found on page. Only the first will be used.')
            page_info = groups[0]
            page_info = json.loads(page_info)
            n_pages = page_info['pages']
            page_current = page_info['current']
        print('  page {}/{}'.format(page_current, n_pages))

        # Parse category code from URL.
        code_category = None
        code_subcategory = None
        code_full = None
        pattern_catcode = r'' + url_cat + '(([^/?]*)/([^?]*))?.*'
        if re.search(pattern_catcode, link_url) is not None:
            groups = re.search(pattern_catcode, link_url).groups()
            code_category = groups[1]
            code_subcategory = groups[2]
            code_full = groups[0]
        else:
            warn('Unable to parse category code from URL.')

        # # Typical example for a category page.
        # <section id=info>
        #     <div class=breadcrumb>
        #         <a href=/cat>Categories</a>
        #         <span>Animals</span>
        #     </div>
        # </section>
        # # Typical example for a sub-category page.
        # <section id=info>
        #     <div class=breadcrumb>
        #         <a href=/cat>Categories</a>
        #         <a href=/cat/animals?page=1>Animals</a>
        #         <span>Armadillos</span>
        #     </div>
        #     <h1>Download free Armadillos transparent PNGs</h1>
        # </section>
        # # Unusual example for a page with multiple sub-categories.
        # <section id=info>
        #     <div class=breadcrumb>
        #         <a href=/cat>Categories</a>
        #         <a href=/cat/sports?page=1>Sports</a>
        #         <a href=/cat/sports/ice-hockey?page=1>Ice Hockey</a>
        #         <span>American Hockey League</span>
        #     </div>
        #     <h1>Download free American Hockey League transparent PNGs</h1>
        # </section>
        # # Previous handling of category and subcategory information.
        # pattern_catinfo = r'<section\s*id=info><div\s*class=breadcrumb>' + \
        #                   r'<a\s*href=/cat>Categories</a>' + \
        #                   r'(<a\s*href=/cat/([^>]*)>([^<>]*)</a>)*' + \
        #                   r'<span>([^<>]*)</span>'
        # if re.search(pattern_catinfo, link_page) is not None:
        #     groups = re.search(pattern_catinfo, link_page).groups()
        #     if groups[0] is None and groups[1] is None and groups[2] is not None:
        #         link_category = groups[2]
        #         link_subcategory = None
        #     elif groups[0] is not None and groups[1] is not None and groups[2] is not None:
        #         link_category = groups[1]
        #         link_subcategory = groups[2]
        #     else:
        #         warn('Unrecognized category information found on page.')
        # else:
        #     warn('Unrecognized category information found on page.')
        link_category = None
        link_subcategory = None
        pattern_catsect = r'(?<=<section\sid=info>)(.*?)(?=</section>)'
        if re.search(pattern_catsect, link_page) is not None:
            groups = re.search(pattern_catsect, link_page).groups()
            if len(groups) != 1:
                warn('Multiple category information sections found on page. Only the first will be used.')
            category_search_result = groups[0]

            pattern_catinfo0 = r'<div\s*class=breadcrumb><a\s*href=/cat>Categories</a>'
            pattern_catinfo1 = r'(<a\s*href=/cat/([^?>]*)[^>]*>([^<>]*)</a>)(<span>([^<>]*)</span>)?'
            if (re.search(pattern_catinfo0, category_search_result) is not None and
                    re.search(pattern_catinfo1, category_search_result) is not None):
                category_items = re.findall(pattern_catinfo1, category_search_result)
                if len(category_items) > 0:
                    link_category = category_items[0][2]
                    if len(category_items) > 1:
                        link_subcategory = ''
                        for ci in range(1, len(category_items)):
                            link_subcategory = category_items[ci][2] if link_subcategory == '' else(
                                    link_subcategory + ', ' + category_items[ci][2])
                            if category_items[ci][4] != '':
                                link_subcategory = link_subcategory + ', ' + category_items[ci][4]
                else:
                    warn('Unrecognized category information found on page.')

        # Search for grid section containing links to relevant subcategory or image pages.
        pattern_gridsect = r'(?<=<section\sid=results\sclass="grid\sgrid-flexible\sclearfix">)(.*?)(?=</section>)'
        if re.search(pattern_gridsect, link_page) is not None:
            groups = re.search(pattern_gridsect, link_page).groups()
            if len(groups) != 1:
                warn('Multiple grid sections found on page. Only the first will be used.')
            grid_search_result = groups[0]

            # Match within-category pages (i.e. lists of subcategories) and add them to the queue.
            pattern_subcat0 = r'<div\s*class=item><a\s*class=image\s*href=/cat/'
            pattern_subcat1 = r'<img\s*src=https://[^/]*/categories/[^/]*'
            if (re.search(pattern_subcat0, grid_search_result) is not None and
                    re.search(pattern_subcat1, grid_search_result) is not None):
                new_items = re.findall(r'<a\s*class=image\s*href=/cat/(' + link + '/[^?>]*)', grid_search_result)
                queue.extend(new_items)
                n_total = n_total + len(new_items)
                print('  extended queue with {} new items'.format(len(queue)))
                del new_items

            # Match within-subcategory pages (i.e. lists of images) and add them to the image list.
            pattern_image0 = r'<div\s*class=item><a\s*class="image\s*pattern"\s*href=/img/'
            pattern_image1 = r'<img\s*src=https://[^/]*/thumbs/[^/]*'
            if (re.search(pattern_image0, grid_search_result) is not None and
                    re.search(pattern_image1, grid_search_result) is not None):
                #     <div class="grid grid-flexible clearfix">
                #         <div class=item>
                #             <a class="image pattern" href=/img/animals/armadillos/armadillo>
                #                 <img src=https://assets.stickpng.com/thumbs/5c7f956c72f5d9028c17ecb1.png alt=Armadillo>
                #             </a>
                #             <div class=title>Armadillo</div>
                #         </div>
                pattern_imageinfo = r'<a\s*class="image pattern"\s*href=/(img/' + link + '/([^>]*))>' + \
                                    r'<img\s*src=https://[^/]*/thumbs/([^/\.]*)\.[^\s]*\s*alt=([^>]*)>'
                image_search_result = re.findall(pattern_imageinfo, grid_search_result)
                for ii in image_search_result:
                    if ii[2] in image_info:
                        warn('Previously processed image code found on page ({}), overwriting.'.format(ii[2]))
                    image_info[ii[2]] = {
                        'title': ii[1],
                        'name': ii[3].strip('"'),
                        'category': link_category,
                        'subcategory': link_subcategory,
                        'code': ii[2],
                        'code_category': code_category,
                        'code_subcategory': code_subcategory,
                        'code_category_full': code_full,
                        'url_info': url_base + ii[0],
                        'url_image': url_assets + ii[2] + '.png',
                    }
        else:
            warn('No grid section found on page.')

        # Reset page counts before moving on to next link in queue.
        if page_current == n_pages:
            n_pages = 1
            page_current = 1
            break
        if 'next' in page_info:
            if page_info['next'] != (page_info['current'] + 1):
                warn('Next listed page count is not one more than the current page count.')
            page_current = page_info['next']

    # Save a snapshot of the collected information.
    if link_counter % 100 == 0 and link_counter > 0:
        now = datetime.now(timezone.utc)
        datetime_str = now.strftime('%Y%m%dd%H%M%StUTC')
        infosave_filename = datetime_str + '_stickpng_image_info_partial.pickle'
        infosave_filepath = os.path.join(infosave_path, infosave_filename)
        with open(infosave_filepath, 'wb') as file:
            pickle.dump([image_info, queue, n_pages, page_current, page_info],
                        file,
                        protocol=pickle.HIGHEST_PROTOCOL)

    link_counter += 1
    queue.remove(link)
    queue = sorted(queue, key=queue_sort_function)


# Download images.
for ii in image_info:
    image_path = os.path.join(asset_path, image_info[ii]['code'] + '.png')
    if not os.path.isfile(image_path):
        sleep(randint(30, 120))
        download_image(image_info[ii]['url_image'], image_path)
        print('Downloaded image {}.png ({}.png).'.format(image_info[ii]['code'], image_info[ii]['title']))
    linkdir_path = os.path.join(tree_path, image_info[ii]['code_category_full'])
    os.makedirs(linkdir_path, exist_ok=True)
    link_path = os.path.join(linkdir_path, image_info[ii]['title'] + '.png')
    if not os.path.islink(link_path):
        os.symlink(image_path, link_path)
        print('Linked {}/{}.png to image asset {}.png.'.format(image_info[ii]['code_category_full'],
                                                               image_info[ii]['title'],
                                                               image_info[ii]['code']))


# # Temporary converter to add details.
# iminf = {}
# for ii in image_info:
#     code_category = None
#     code_subcategory = None
#     code_full = None
#     pattern_catcode = r'^' + url_base + 'img/((([^/]*)/?(.*)?)/([^/]*))$'
#     if re.search(pattern_catcode, image_info[ii]['url_info']) is not None:
#         groups = re.search(pattern_catcode, image_info[ii]['url_info']).groups()
#         code_category = groups[2]
#         code_subcategory = groups[3] if groups[3] != '' else None
#         code_full = groups[1]
#     else:
#         warn('Unable to parse category code from URL for image_info {}.'.format(ii))
#
#     if image_info[ii]['code'] in iminf:
#         warn('Duplicate image code ({}) found in image_info, overwriting.'.format(image_info[ii]['code']))
#     iminf[image_info[ii]['code']] = {
#         'title': image_info[ii]['title'],
#         'name': image_info[ii]['name'],
#         'category': image_info[ii]['category'],
#         'subcategory': image_info[ii]['subcategory'],
#         'code': image_info[ii]['code'],
#         'code_category': code_category,
#         'code_subcategory': code_subcategory,
#         'code_category_full': code_full,
#         'url_info': image_info[ii]['url_info'],
#         'url_image': image_info[ii]['url_image'],
#     }



    # # Within-category pages:
    # <section id=info>
    #     <div class=breadcrumb>
    #         <a href=/cat>Categories</a>
    #         <span>Animals</span>
    #     </div>
    #     <h1>Download free Animals transparent PNGs</h1>
    #     <div class=description>People[...]. </div>
    # </section>
    # <section>
    #     <div class="stickad header-thick">
    #         <div data-fuse=22411526620></div>
    #     </div>
    # </section>
    # <section id=results class="grid grid-flexible clearfix">
    #     <div class="grid grid-flexible clearfix">
    #         <div class=item>
    #             <a class=image href=/cat/animals/armadillos?page=1>
    #                 <img src=https://assets.stickpng.com/categories/7639.png alt=Armadillos>
    #             </a>
    #             <div class=title>Armadillos</div>
    #         </div>
    #         <div class=item>
    #             <a class=image href=/cat/animals/baboons?page=1>
    #                 <img src=https://assets.stickpng.com/categories/2188.png alt=Baboons>
    #             </a>
    #             <div class=title>Baboons</div>
    #         </div>
    #         [...]
    #     </div>
    # </section>
    # <section>
    #     <div class="stickad header-thick"><div data-fuse=22411526629></div></div>
    # </section>
    # <section id=pagination data-pagination='{"pages":9,"current":1,"next":2}'></section>

    # # Image page:
    # <section id=info>
    #     <div class=breadcrumb>
    #         <a href=/cat>Categories</a>
    #         <a href=/cat/animals?page=1>Animals</a>
    #         <span>Armadillos</span>
    #     </div>
    #     <h1>Download free Armadillos transparent PNGs</h1>
    # </section>
    # <section>
    #     <div class="stickad header-thick">
    #         <div data-fuse=22411526620></div>
    #     </div>
    # </section>
    # <section id=results class="grid grid-flexible clearfix">
    #     <div class="grid grid-flexible clearfix">
    #         <div class=item>
    #             <a class="image pattern" href=/img/animals/armadillos/armadillo>
    #                 <img src=https://assets.stickpng.com/thumbs/5c7f956c72f5d9028c17ecb1.png alt=Armadillo>
    #             </a>
    #             <div class=title>Armadillo</div>
    #         </div>
    #         <div class=item>
    #             <a class="image pattern" href=/img/animals/armadillos/armadillo-front-view>
    #                 <img src=https://assets.stickpng.com/thumbs/5c7f954772f5d9028c17ecac.png alt="Armadillo Front View">
    #             </a>
    #             <div class=title>Armadillo Front View</div>
    #         </div>
    #     </div>
    #     [...]
    # </section>
    # <section id=pagination data-pagination='{"pages":1,"current":1}'></section>


    #
    # for p in range(max_pages):
    #     link_page_count_curr = link_page_count_prev + 1
    #     print('p {} curr {} prev {}'.format(p, link_page_count_curr, link_page_count_prev))
    #     if url_base.endswith('/'):
    #         link_url = url_base + link + '?page=' + str(link_page_count_curr)
    #     else:
    #         link_url = url_base + '/' + link + '?page=' + str(link_page_count_curr)
    #     req = Request(url=link_url, headers={'User-Agent': 'Mozilla/5.0'})
    #     # print('{}/{}'.format(i_link, len(queue)), end='', flush=True)
    #     print('{}/{}'.format(i_link, len(queue)))
    #     sleep(randint(3, 20))
    #     link_page = urlopen(req, timeout=10).read().decode('utf8')
    #
    #     if link_page_count_curr > 1:
    #         link_page_num = re.findall(r'<title[^>]*>[^<>]*\s*Page\s*([0-9]+)\s*[^<>]*</title[^>]*>', link_page)
    #         if link_page_num:
    #             if link_page_num[0].isnumeric():
    #                 link_page_num = float(link_page_num[0])
    #                 if link_page_num.is_integer():
    #                     link_page_num = int(link_page_num)
    #                     link_page_count_curr = link_page_num
    #                     print('updated curr {}'.format(link_page_count_curr))
    #     if link_page_count_curr == link_page_count_prev:
    #         print('broke')
    #         break
    #
    #     # if re.match(r'<a\s*href=/img/download/([a-zA-Z0-9]+)\s*class=button><i></i>Download</a>', page_subcat):
    #     #     images[subcategory] = re.findall(r'<a\s*class=image\s*href=/img/([^?>]*)', page_subcat)
    #     #
    #     #
    #     #     '<a href=/img/download/580b57fbd9996e24bc43bdf2 class=button><i></i>Download</a>'
    #     #     '<a href=/cat>Categories</a><a href=/cat/bots-and-robots?page=1>Bots and Robots</a><span>Robot Warm Up</span></div>'
    #     #
    #     #     'https://assets.stickpng.com/images/580b57fbd9996e24bc43bdf2.png'
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
    #     queue.extend(re.findall(r'<a\s*class=image\s*href=/cat/(' + link + '/[^?>]*)', link_page))
    #     # queue = list(set(queue))
    #
    #     link_page_count_prev = link_page_count_curr
    #     print('updated last {}'.format(link_page_count_curr))
    #     # print('.', end='', flush=True)
    #
    # queue.remove(link)






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
