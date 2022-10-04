# Scrapping www.bebelereisim.com for Turkish names.
import requests
from bs4 import BeautifulSoup
import re
import csv


def main():
    # https://stackoverflow.com/questions/61968521/python-web-scraping-request-errormod-security
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0',
    }
    URL = "https://www.bebelereisim.com/isimler/a-z/"
    last_page_num = 624

    names = []
    for i in range(1, last_page_num + 1):
        print(f"Page {i} of {last_page_num}")
        # Content is under:
        # <main id="main">
        #   <div class="container">
        #     <ol class="row boxed">
        #       <li> 
        #         <a>
        #           "Name " 
        #           <small>Gender</small>
        page_url = URL + str(i)
        page = requests.get(page_url, headers=headers)
        soup = BeautifulSoup(page.content, 'html.parser')
        main = soup.find(id='main')
        container = main.find(class_='container')
        ol = container.find(class_='row boxed')
        lis = ol.find_all('li')
        for li in lis:
            a = li.find('a')
            # Remove the <small> tag
            a.small.decompose()
            name = a.text.strip()
            # Add the name to the list
            names.append(name)

    # Write the names to a csv file
    with open('names_new.csv', 'w') as f:
        writer = csv.writer(f)
        for name in names:
            writer.writerow([name])


if __name__ == '__main__':
    main()
