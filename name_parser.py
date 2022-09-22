# Cleaning and creating a csv file from publicly available data from: 
# https://kerteriz.net/tum-turkce-erkek-ve-kadin-isimleri-cinsiyet-listesi-veritabani/
#
# The original data is in the form of a text file with names and genders
# in the form ('Aba', 'K').

import csv
import re


def main():
    with open('makemore/isimler.txt', 'r') as f:
        with open('makemore/names.csv', 'w') as g:
            writer = csv.writer(g)
            # Add header (Name, Gender)
            writer.writerow(['Name', 'Gender'])
            for line in f:
                line = line.replace('(', '').replace(')', '').replace("'", '').replace("\n", '')
                line = line.split(', ')
                writer.writerow(line)


if __name__ == '__main__':
    main()